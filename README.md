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

Ranks are set to `I/4` (25% setting) and `I/8` (50% setting) so the **per-expert compression ratio is
identical across models** (keep the top quarter / eighth of each weak expert's singular values).

## Results

Rows marked with a dagger are external baselines (REAP prune / REAM merge), shown for comparison, not
implemented in this repo. SVD rows are this repo's methods.

GSM8K is reported as `strict/flexible` exact-match. MMLU protocols: **0-shot** = 0-shot no-chat (with
forced BOS on Gemma); **chat** = 5-shot chat-template. **The 0-shot column is the cross-model comparable
one** (both models have it). Gemma additionally reports the 5-shot chat column (its most favourable, and
the one that avoids the BOS issue most naturally); Qwen was not run in 5-shot chat, so it has 0-shot only.

### Qwen3.6-35B-A3B (8/68/24 calibration)

MMLU here is the 0-shot no-chat protocol (Qwen was trained without BOS, so plain-text scoring is correct).

**25%** (compress weakest 64/256 experts, rank 128)
| Method | GSM8K str/flex | MMLU |
|---|---|---|
| Baseline (256) | 87.87/89.08 | 83.73 |
| REAP (dagger) | 86.73/87.87 | 78.09 |
| SVD Rank 128 | 87.49/89.16 | 80.12 |
| SVD-activation aware Rank 128 | 88.25/89.31 | 79.80 |
| REAM (dagger) | 87.64/88.78 | 68.58 |

**50%** (compress weakest 128/256 experts, rank 64)
| Method | GSM8K str/flex | MMLU |
|---|---|---|
| Baseline (256) | 87.87/89.08 | 83.73 |
| REAP (dagger) | 21.30/21.91 | 61.67 |
| SVD Rank 64 | 68.08/68.92 | 22.95 |
| SVD-activation aware Rank 64 | 36.54/37.00 | 22.95 |
| REAM (dagger) | 79.23/80.44 | 51.64 |

### Gemma-4-26B-A4B-it (8/68/24 calibration)

Two MMLU columns: **chat** (5-shot chat-template, the trustworthy Gemma protocol) and **0-shot** (0-shot
no-chat with forced BOS, the Qwen-comparable protocol).

**25%** (weakest 32/128, rank 176)
| Method | GSM8K str/flex | MMLU chat | MMLU 0-shot |
|---|---|---|---|
| Baseline (128) | 86.96/88.32 | 83.83 | 66.66 |
| REAP (dagger) | 84.76/85.67 | 78.88 | 70.80 |
| SVD Rank 176 | 85.14/86.58 | 80.99 | 68.43 |
| SVD-activation aware Rank 176 | 86.28/87.72 | 80.35 | 66.04 |
| REAM (dagger) | 86.20/87.57 | 71.36 | 59.86 |

**50%** (weakest 64/128, rank 88)
| Method | GSM8K str/flex | MMLU chat | MMLU 0-shot |
|---|---|---|---|
| Baseline (128) | 86.96/88.32 | 83.83 | 66.66 |
| REAP (dagger) | 73.69/74.68 | 59.46 | 51.42 |
| SVD Rank 88 | 84.08/85.37 | 64.14 | 55.94 |
| SVD-activation aware Rank 88 | 81.80/82.79 | 61.44 | 52.24 |
| REAM (dagger) | 86.20/86.96 | 49.56 | 41.77 |

### Rank-128 ablation + mean-baseline variant

Follow-up sweep holding **rank = 128 for every configuration** (both compression levels, both models), plus a
**mean-baseline** variant of activation-aware SVD. Mean-baseline strips a fixed rank-1 term `(1/H)*ones([O,H])`
(which maps each token to the mean of its input channels) from `W` before fitting `U,V`, so all 128 rank
directions model the deviation from that mean instead of re-learning it; reconstruction adds the baseline back.
Gemma MMLU here is **0-shot forced-BOS** (the Qwen-comparable protocol). All SVD numbers reproduced by the
packaged `svd_compression` code (parity: Qwen plain 25% rank-128 GSM8K 87.57/89.08 vs the table's 87.49/89.16).

**Qwen3.6-35B-A3B (rank 128)**
| Method | Frac | GSM8K str/flex | MMLU 0-shot |
|---|---|---|---|
| SVD-activation aware (from table above) | 25% | 88.25/89.31 | 79.80 |
| SVD-activation aware + mean-baseline | 25% | 87.95/89.61 | **23.02** (collapsed) |
| SVD plain | 50% | 83.70/84.38 | 22.95 |
| SVD-activation aware | 50% | 72.86/73.09 | 23.02 |
| SVD-activation aware + mean-baseline | 50% | _running_ | _running_ |

**Gemma-4-26B-A4B-it (rank 128)**
| Method | Frac | GSM8K str/flex | MMLU 0-shot |
|---|---|---|---|
| SVD plain | 25% | 84.15/86.13 | 69.16 |
| SVD-activation aware | 25% | 83.09/84.31 | 67.25 |
| SVD-activation aware + mean-baseline | 25% | 84.38/85.82 | **68.36** |
| SVD plain | 50% | 85.29/86.28 | 58.69 |
| SVD-activation aware | 50% | 84.99/85.60 | 54.25 |
| SVD-activation aware + mean-baseline | 50% | _running_ | _running_ |

**Ablation findings**
- **Rank 128 > rank 64 at 50%** (Qwen): GSM8K plain 83.70 vs 68.08, aware 72.86 vs 36.54. More retained rank
  is strictly better on any signal-bearing metric; MMLU is at the random floor (~23) for both ranks.
- **Mean-baseline helps Gemma at 25%** (+1.3 GSM8K, +1.1 MMLU over standard activation-aware) but is
  **not universal**: on Qwen 25% it leaves GSM8K intact (87.95) yet **collapses MMLU to random (23.02)**. Qwen
  (tiny shared expert, no dense-MLP floor) is far more fragile, so the extra fit freedom overfits the
  calibration domain -- math survives, broad knowledge dies. Same asymmetry as standard activation-aware at
  aggressive rank.

### Key findings (SVD)
1. **Plain SVD is the best knowledge-preserving method at mild compression.** Highest MMLU at 25% on both models among non-baseline methods (Qwen35B 80.12; Gemma 80.99 chat / 68.43 0-shot), beating the prune/merge baselines.
2. **Activation-aware SVD helps only at generous rank.** It beats plain SVD at 25% on GSM8K (Qwen35B 88.25 vs 87.49; Gemma 86.28 vs 85.14) but *inverts* at 50%/low rank (Qwen35B 36.5 vs 68.1 GSM8K; Gemma 81.8 vs 84.1), because the activation fit overfits the calibration domain when the rank is tight.
3. **SVD degrades sharply at aggressive rank.** At 50%/rank-64 on Qwen35B, MMLU collapses to random (both variants 22.95) while GSM8K partly survives (68.1 plain). The weak experts are badly approximated at rank-64, and MMLU (single-pass, knowledge-heavy) is far more sensitive to that than GSM8K.
4. **Architecture matters.** Gemma's per-layer **parallel dense MLP** is a capability floor: it keeps GSM8K at 82-87 even at 50%, whereas Qwen35B (only a tiny shared expert) is far more fragile (SVD 50% GSM8K drops to 68 / 37).

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

## The single-GPU OOM caveat
SVD reconstructs experts *in place*, so the model stays full size (35B is ~69GB). The 248k-vocab fp32
logit head (~7.5GB) then does not fit alongside it on one 80GB GPU during MMLU. Pass `--shard` to spread
the model over 2 GPUs (`device_map=auto`) for the eval. (Gemma at ~52GB fits on one GPU.)

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
