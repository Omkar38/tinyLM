# tinyLM
StonyTinyLM — tiny GPT from scratch

From-scratch decoder-only Transformer (~26.3M params) trained by streaming English Wikipedia via HuggingFaceFW/finewiki (name="en"). Single Python file; DDP + AMP; token packing; cosine LR.

Quickstart
pip install -r requirements.txt
# single GPU
python train_stonytinylm.py --dataset_name HuggingFaceFW/finewiki --dataset_config en
# multi-GPU (e.g., 4x)
torchrun --standalone --nproc_per_node=4 train_stonytinylm.py \
  --dataset_name HuggingFaceFW/finewiki --dataset_config en

Model (Micro GPT)

Params: ~26.3M

Config: n_layer=8, n_head=8, n_embd=320, seq_len=1024, vocab=50304

Arch: GPT-2 style blocks (Multi-Head Attention, MLP, residuals, LayerNorm), weight-tied embeddings.

Dataset

Source: HuggingFaceFW/finewiki (English subset; name="en", split="train", streaming=True)

Extracted from August 2025 Wikipedia HTML; templates rendered; non-article pages removed; math/tables preserved; rich metadata.

Note: This repo does not redistribute data; please follow the dataset’s license/terms.

Features

Streaming data pipeline (no preprocessing)

DDP-ready (torchrun), AMP (bf16/fp16), fused AdamW (when available)

Continuous token packing for minimal padding

Cosine LR with warmup, gradient clipping, checkpointing

Files

train_stonytinylm.py — training script (from scratch)

log/ — checkpoints model_XXXXX.pt, log.txt (created at runtime)

Results (fill as you train)

Loss curve screenshot

Tokens/sec (1x / multi-GPU)

Sample generations (after adding a sampling script)
