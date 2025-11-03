# train_stonytinylm.py
# From-scratch tiny GPT trained via streaming on HuggingFaceFW/finewiki (English subset).
# - Dataset: load_dataset("HuggingFaceFW/finewiki", name="en", split="train", streaming=True)
# - No evaluation/validation (train-only), simple & fast
# - DDP-ready, AMP, cosine LR, AdamW
# - Token packing from a continuous stream
# - Falls back across potential text fields: ["text", "text_md", "markdown"]

# torchrun --standalone --nproc_per_node=4 train_stonytinylm.py --dataset_name HuggingFaceFW/finewiki --dataset_config en


import os, math, time, inspect, itertools, argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# -----------------------------
# CLI 
# -----------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_per_step", type=int, default=131072, help="global tokens per optimization step")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--micro_bsz", type=int, default=32)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--max_lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--dataset_name", type=str, default="HuggingFaceFW/finewiki")
    ap.add_argument("--dataset_config", type=str, default="en")  # English subset
    ap.add_argument("--text_keys", type=str, default="text,text_md,markdown")
    ap.add_argument("--log_dir", type=str, default="log")
    return ap.parse_args()

args = get_args()

# -----------------------------
# DDP / device
# -----------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

device_type = "cuda" if str(device).startswith("cuda") else ("mps" if str(device).startswith("mps") else "cpu")
print(f"[Init] device: {device}")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# -----------------------------
# Tokenizer (GPT-2 BPE via tiktoken)
# -----------------------------
import tiktoken
enc = tiktoken.get_encoding("gpt2")  # base vocab 50257; we pad to 50304 for matmul perf

# -----------------------------
# Streaming dataloader (FineWiki EN)
# -----------------------------
from datasets import load_dataset

class StreamingLoader:
    """
    Streams 'HuggingFaceFW/finewiki' English subset (train-only) and packs tokens.
    - Dataset fields may vary; will try several text keys (text, text_md, markdown).
    - DDP sharding via stride: each rank takes every k-th example.
    - Continuous token stream => minimal padding.
    """
    def __init__(self, B, T, process_rank=0, num_processes=1,
                 dataset_name="HuggingFaceFW/finewiki",
                 dataset_config="en",
                 text_keys=("text","text_md","markdown")):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.text_keys = list(text_keys)

        self.ds = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
        self.it = itertools.islice(self.ds, process_rank, None, num_processes)

        self.buf = torch.empty(0, dtype=torch.long)
        if master_process:
            print(f"[Data] Streaming {dataset_name} ({dataset_config}) split=train rank={process_rank}/{num_processes}")
            print(f"[Data] Will try text fields in order: {self.text_keys}")

    def _extract_text(self, ex):
        for k in self.text_keys:
            if k in ex and isinstance(ex[k], str) and ex[k]:
                return ex[k]
        return None

    def _fill_buf(self, min_needed):
        toks, total = [], 0
        while total < min_needed:
            try:
                ex = next(self.it)
            except StopIteration:
                # Restart stream
                self.it = itertools.islice(self.ds, self.process_rank, None, self.num_processes)
                ex = next(self.it)
            text = self._extract_text(ex)
            if not text:
                continue
            ids = enc.encode_ordinary(text)  # fast path; no special tokens
            if not ids:
                continue
            t = torch.tensor(ids, dtype=torch.long)
            toks.append(t)
            total += t.numel()
        if toks:
            cat = torch.cat(toks)
            self.buf = torch.cat([self.buf, cat])

    def next_batch(self):
        needed = self.B * self.T + 1
        if self.buf.numel() < needed:
            self._fill_buf(needed)
        buf = self.buf[:needed]
        self.buf = self.buf[needed:]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1: ].view(self.B, self.T)
        return x, y

    def reset(self):
        self.buf = torch.empty(0, dtype=torch.long)

# -----------------------------
# Model
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # pad 50257 up to multiple of 64 for matmul perf
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 320

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f= nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # tie weights
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for n, p in params.items() if p.dim() >= 2]
        nodecay = [p for n, p in params.items() if p.dim() < 2]
        groups = [{'params': decay, 'weight_decay': weight_decay},
                  {'params': nodecay, 'weight_decay': 0.0}]
        if master_process:
            print(f"[Optim] decay_tensors={len(decay)}, nodecay_tensors={len(nodecay)}")
        fused_avail = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avail and device_type == "cuda"
        if master_process:
            print(f"[Optim] fused AdamW: {use_fused}")
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

# -----------------------------
# Hparams & scheduler
# -----------------------------
total_batch_size = args.tokens_per_step
B = args.micro_bsz
T = args.seq_len
assert total_batch_size % (B * T * ddp_world_size) == 0, "tokens_per_step must be divisible by B*T*world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"[Train] tokens/step={total_batch_size}, grad_accum_steps={grad_accum_steps}")

max_steps = args.max_steps
max_lr = args.max_lr
min_lr = args.min_lr
warmup_steps = args.warmup_steps

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)

torch.set_float32_matmul_precision('high')

# -----------------------------
# Data
# -----------------------------
text_keys = tuple([k.strip() for k in args.text_keys.split(",") if k.strip()])
train_loader = StreamingLoader(
    B=B, T=T,
    process_rank=ddp_rank, num_processes=ddp_world_size,
    dataset_name=args.dataset_name,
    dataset_config=args.dataset_config,
    text_keys=text_keys
)

# -----------------------------
# Model & Optimizer
# -----------------------------
cfg = GPTConfig(block_size=T, vocab_size=50304, n_layer=8, n_head=8, n_embd=320)
model = GPT(cfg).to(device)

use_compile = args.compile and hasattr(torch, "compile")
if use_compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)

# -----------------------------
# Logging / checkpoints
# -----------------------------
log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
open(log_file, "w").close()

if master_process:
    total_params = sum(p.numel() for p in raw_model.parameters())
    print(f"[Model] params: {total_params:,}")

# -----------------------------
# Train loop
# -----------------------------
for step in range(max_steps):
    t0 = time.time()
    last = (step == max_steps - 1)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=(torch.bfloat16 if device_type != "cpu" else torch.float32)):
            _, loss = model(x, y)
            loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize()

    dt = time.time() - t0
    tokens = B * T * grad_accum_steps * ddp_world_size
    tps = tokens / dt

    if master_process:
        print(f"step {step:5d} | loss {loss_accum.item():.6f} | lr {lr:.4e} | gn {grad_norm:.4f} | {dt*1000:.0f}ms | tok/s {tps:.0f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
        if step > 0 and (step % 5000 == 0 or last):
            ckpt = {
                "model": raw_model.state_dict(),
                "config": raw_model.config,
                "step": step,
            }
            path = os.path.join(log_dir, f"model_{step:05d}.pt")
            torch.save(ckpt, path)
            print(f"[CKPT] saved {path}")

if ddp:
    destroy_process_group()
