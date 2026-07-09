# Low-Rank Compression of Weak MoE Experts (SVD)

Compress the **weakest experts** of a Mixture-of-Experts LLM by **low-rank (SVD) approximation** and
measure the accuracy cost on **GSM8K** (math reasoning) and **MMLU** (broad knowledge), across MoE
models. The results tables also cite **pruning (REAP)** and **merging (REAM)** numbers as external
baselines for context; those are separate published methods and are **not** implemented in this repo.

## Methods (implemented here)

| Method | What it does |
|---|---|
| **SVD (plain)** | Truncated-SVD reconstruction of each weak expert's `gate`/`up`/`down`, minimising Frobenius weight error `||W - W_r||`. Data-agnostic. |
| **SVD (activation-aware)** | Initialise `U,V` from the plain SVD, then **refine `U,V` by alternating least squares** to minimise the **output** error `||XW - X.U.V^T||` over gate-weighted calibration activations (NNCF `lora_correction`-style). Same rank/shape as plain SVD; the difference is that `U,V` are data-fitted, not the raw SVD factors. |
| **SVD (activation-aware + mean-baseline)** | Activation-aware as above, but first strip a fixed rank-1 baseline `B = (1/H)*ones([O,H])` from `W` (it maps each token to the mean of its input channels). `U,V` then fit only the residual `W - B`, so all `rank` directions model the deviation from the mean instead of re-learning it; reconstruction adds `B` back: `W_r = (U V)^T + B`. Enabled with `--mean-baseline`. |

Both reconstruct **in place** (same shape as the original weight), so they measure the *accuracy
ceiling* of a rank-r approximation. Realising the memory saving would require storing `U,V` separately
and changing the forward pass (out of scope here).

"Weakest" = lowest **saliency** (router-softmax-weighted L2 norm of expert output over routed tokens,
following Cerebras REAP arXiv:2510.13999 Eq. 9), computed via the real `model.forward()` with hooks.

**External baselines in the tables (not implemented here):** REAP (prune weak experts, arXiv:2510.13999)
and REAM (merge weak experts, arXiv:2604.04356). Shown only for comparison.

## Calibration data

The calibration set drives **two** things: which experts are ranked "weakest" (saliency), and what
activations the activation-aware fit is optimised for. So its **domain mix matters** and is held
identical across models here. Both pre-built batches in `calibration/` use the **same recipe**:

| File | Model | Domain mix (c4 / math / code) | Size |
|---|---|---|---|
| `paper_calib_qwen36.pt` | Qwen3.6-35B | **~8 / 68 / 24** by tokens (REAM paper Table-1: c4 512x128, math 1024x512, code 512x512) | 2048 seqs, ~846k tokens |
| `gemma_calib.pt` | Gemma-4-26B | **~8 / 68 / 24** by tokens (same recipe, Gemma tokenizer) | 2048 seqs, ~845k tokens |

Build your own with `svd_compression/calibration.py`:
- `build_calibration(tokenizer)` gives the default **paper 8/68/24** c4/math/code mix (`PAPER_RECIPE`), or
- `build_calibration(tokenizer, single_domain="math")` for a single-domain calibration.

The two batches use the same 8/68/24 recipe and the same 2048-sequence size (differing only by
tokenizer), so results are comparable across the two models. Match the calibration mix to your
deployment domain, and to any runs you compare against.

## Models

| Model | Layers | Experts | top-k | Expert intermediate `I` | Notes |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 40 | 256 | 8 | 512 | hybrid SSM + full-attn; tiny shared expert |
| Gemma-4-26B-A4B-it | 30 | 128 | 8 | 704 | **parallel dense MLP** per layer (a capability floor) |

Ranks are 128.

## Results

REAP and REAM rows are external baselines (prune / merge), shown for comparison, not implemented in this
repo. SVD rows are this repo's methods.

GSM8K is reported as `strict/flexible` exact-match. MMLU is **0-shot** no-chat (with forced BOS on Gemma;
Qwen is trained without BOS so plain-text scoring is correct). Gemma also has a **chat** column (5-shot
chat-template). Cells marked `eval` are still running. `mean-baseline` = the activation-aware variant that
strips a fixed rank-1 `(1/H)*ones` term before fitting.

### Qwen3.6-35B-A3B (8/68/24 calibration)

| Method | Frac | Rank | GSM8K str/flex | MMLU 0-shot |
|---|---|---|---|---|
| Baseline (256) | -- | -- | 87.87/89.08 | 83.73 |
| REAP | 25% | -- | 86.73/87.87 | 78.09 |
| REAM | 25% | -- | 87.64/88.78 | 68.58 |
| SVD plain | 25% | 128 | 87.49/89.16 | **80.12** |
| SVD-activation aware | 25% | 128 | **88.25/89.31** | 79.80 |
| SVD-activation aware + mean-baseline | 25% | 128 | 87.95/89.61 | 23.02 |
| SVD-activation aware + mean-baseline | 25% | 256 | 32.07/33.06 | eval |
| REAP | 50% | -- | 21.30/21.91 | **61.67** |
| REAM | 50% | -- | 79.23/80.44 | 51.64 |
| SVD plain | 50% | 64 | 68.08/68.92 | 22.95 |
| SVD-activation aware | 50% | 64 | 36.54/37.00 | 22.95 |
| SVD plain | 50% | 128 | **83.70/84.38** | 22.95 |
| SVD-activation aware | 50% | 128 | 72.86/73.09 | 23.02 |
| SVD-activation aware + mean-baseline | 50% | 128 | 71.27/71.42 | eval |

### Gemma-4-26B-A4B-it (8/68/24 calibration)

| Method | Frac | Rank | GSM8K str/flex | MMLU chat | MMLU 0-shot |
|---|---|---|---|---|---|
| Baseline (128) | -- | -- | 86.96/88.32 | 83.83 | 66.66 |
| REAP | 25% | -- | 84.76/85.67 | 78.88 | **70.80** |
| REAM | 25% | -- | 86.20/87.57 | 71.36 | 59.86 |
| SVD plain | 25% | 176 | 85.14/86.58 | **80.99** | 68.43 |
| SVD-activation aware | 25% | 176 | **86.28/87.72** | 80.35 | 66.04 |
| SVD plain | 25% | 128 | 84.15/86.13 | -- | 69.16 |
| SVD-activation aware | 25% | 128 | 83.09/84.31 | -- | 67.25 |
| SVD-activation aware + mean-baseline | 25% | 128 | 84.38/85.82 | -- | 68.36 |
| SVD-activation aware + mean-baseline | 25% | 256 | 84.46/85.75 | -- | 66.33 |
| REAP | 50% | -- | 73.69/74.68 | 59.46 | 51.42 |
| REAM | 50% | -- | **86.20/86.96** | 49.56 | 41.77 |
| SVD plain | 50% | 88 | 84.08/85.37 | **64.14** | 55.94 |
| SVD-activation aware | 50% | 88 | 81.80/82.79 | 61.44 | 52.24 |
| SVD plain | 50% | 128 | 85.29/86.28 | -- | **58.69** |
| SVD-activation aware | 50% | 128 | 84.99/85.60 | -- | 54.25 |
| SVD-activation aware + mean-baseline | 50% | 128 | 82.34/82.94 | -- | 52.23 |

## Usage

```bash
pip install -r requirements.txt

# 1. (once per model) build calibration + saliency  [pre-built copies are in calibration/ and saliency/]
python -m svd_compression.compress --model $MODEL --method svd --frac 0.25 --rank 128 \
    --calib calibration/paper_calib_qwen36.pt --saliency saliency/qwen36_expert_saliency.json \
    --build-saliency --label build --out /tmp/scratch.json     # builds saliency then runs

# 2. run a config (svd | actaware) and eval
python -m svd_compression.compress --model $MODEL --method actaware --frac 0.25 --rank 128 \
    --calib calibration/paper_calib_qwen36.pt --saliency saliency/qwen36_expert_saliency.json \
    --label qwen36-actaware-w25-r128 --out results/qwen36.json --eval gsm8k mmlu --shard
```

## Eval protocol (important)
- **GSM8K**: 5-shot, chat-template, greedy, 1024 max gen tokens. Uniform across all models.
- **MMLU** has a BOS subtlety on Gemma:
  - Qwen (`bos_token_id is None`, trained without BOS): **0-shot, no chat template** (paper `eval_mc.py`) is correct.
  - Gemma (requires a leading `<bos>`): plain-text `tok(prompt)` sets `add_bos_token=False`, so 0-shot no-chat *silently omits* the BOS the model was trained on, giving a uniform ~random ~48% artifact. Two valid fixes are reported: **5-shot chat-template** (the template string starts with `<bos>`), and **0-shot with `add_bos_token=True` forced** (Qwen-comparable). The forced-BOS 0-shot works (baseline 66.66, not random) but sits below chat-template because this is an instruct model.
- `svd_compression/evaluate.py` auto-picks the protocol by `tokenizer.bos_token_id`. The Gemma tables report both.

## Layout
```
svd_compression/
  lowrank.py       # plain_svd + activation_aware_svd  (the two core functions)
  models.py        # per-model adapters (module paths, router math)
  saliency.py      # weak-expert saliency + precompute (self-contained)
  calibration.py   # build calibration batches
  compress.py      # CLI: low-rank-compress weak experts + eval
  evaluate.py      # GSM8K + MMLU with per-model MMLU protocol
saliency/          # pre-built saliency (qwen36, gemma) on the 8/68/24 calibration
calibration/       # pre-built calibration batches (.pt, via Git LFS)
results/           # result JSONs behind the tables above
```
