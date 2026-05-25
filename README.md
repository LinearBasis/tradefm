# TradeFM

Decoder-only transformer trained on MOEX OrderLog event data (inspired by [TradeFM, arXiv:2602.23784](notes/2602.23784v1.pdf)). Learns latent market-microstructure dynamics via scale-invariant tokenization of trade/order flow.

## Pipeline

1. **Tokenize** raw OrderLog CSVs → per-instrument per-day token sequences (vocab 16384, ADV-bucketed liquidity).
2. **Train** `OrderFlowTransformer` (Llama-style: SwiGLU, RMSNorm, RoPE, GQA) with Muon-hybrid optimizer + composite cross-entropy.
3. **Evaluate** via closed-loop rollout against a minimal LOB simulator → stylized facts + K-S / W₁ distributional fidelity.

## Quick start

```bash
uv sync

# 1. Tokenize
uv run python -m src.data.pipeline \
    --raw-dir data/raw --output-dir data/processed \
    --top-n-instruments 20 --n-jobs 8 --polars-threads 4

# 2. Train (multi-GPU DDP)
uv run torchrun --standalone --nproc-per-node=8 \
    -m scripts.train_transformer --config configs/base_50m.json \
    --num-workers 8 --amp bf16

# 3. Evaluate
uv run python -m scripts.eval_rollout \
    --checkpoint checkpoints/transformer_50m/best.pt \
    --tokenizer data/processed/tokenizer.json \
    --val-sequences data/processed/sequences \
    --all-instruments --n-rollouts 10 --n-events 1024 \
    --output runs/eval/xfmr_50m
```

## Configs

- `configs/base_50m.json` — 52M params, production.
- `configs/base_20m.json` — 19M, fast iteration.
- `configs/smoke.json` — CPU smoke for debug.
