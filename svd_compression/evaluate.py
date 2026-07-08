"""GSM8K + MMLU evaluation, with the per-model MMLU protocol baked in.

GSM8K: always 5-shot, chat-template (task default), max_gen_toks=1024.

MMLU protocol differs by model and this matters (see README "Eval protocol"):
  - Qwen (bos_token_id=None, trained without BOS): 0-shot, NO chat template -> the paper eval_mc.py
    protocol. Plain-text scoring is correct because nothing is missing.
  - Gemma (REQUIRES a leading <bos> the tokenizer does not add for plain text): 0-shot no-chat is
    SILENTLY BROKEN (uniform ~random). Use 5-shot chat-template multiturn, which prepends <bos>.
Detection is by tokenizer.bos_token_id (None -> Qwen path, else -> Gemma/BOS path).
"""
import json
import time
from pathlib import Path

import lm_eval
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


def append_result(output_json, row):
    path = Path(output_json)
    rows = json.loads(path.read_text()) if path.exists() else []
    rows.append(row)
    path.write_text(json.dumps(rows, indent=2))
    print(json.dumps(row, indent=2), flush=True)
    print("saved", output_json, flush=True)


def run_evals(model, tokenizer, adapter, tasks, label, output_json, method=None, frac=None, rank=None,
              gsm8k_batch_size=16, mmlu_batch_size=4):
    requires_bos = tokenizer.bos_token_id is not None       # Gemma-style: BOS required for correct MMLU
    language_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=gsm8k_batch_size, max_gen_toks=1024,
                          **({"enable_thinking": False} if not requires_bos else {}))
    result_base = {"label": label, "method": method, "frac": frac, "rank": rank}

    if "gsm8k" in tasks:
        print("GSM8K 5-shot chat...", flush=True)
        start_time = time.time()
        gsm8k_result = lm_eval.simple_evaluate(model=language_model, tasks=["gsm8k"], num_fewshot=5,
                                               apply_chat_template=True)["results"]["gsm8k"]
        append_result(output_json, {
            **result_base, "phase": "gsm8k",
            "gsm8k_strict": round(gsm8k_result.get("exact_match,strict-match", 0) * 100, 2),
            "gsm8k_flexible": round(gsm8k_result.get("exact_match,flexible-extract", 0) * 100, 2),
            "seconds": round(time.time() - start_time)})

    if "mmlu" in tasks:
        language_model._batch_size = mmlu_batch_size
        start_time = time.time()
        if requires_bos:
            # Gemma: 5-shot chat-template multiturn (prepends <bos>); 0-shot no-chat is BOS-broken.
            print("MMLU 5-shot chat-template multiturn (BOS-safe)...", flush=True)
            mmlu_result = evaluator.simple_evaluate(model=language_model, tasks="mmlu", num_fewshot=5,
                                                    batch_size=mmlu_batch_size, apply_chat_template=True,
                                                    fewshot_as_multiturn=True)
            phase_key = "mmlu_5shot_chat"
        else:
            # Qwen: 0-shot no-chat (paper eval_mc protocol). Fixed seeds for reproducibility.
            print("MMLU 0-shot no-chat (paper protocol)...", flush=True)
            mmlu_result = evaluator.simple_evaluate(model=language_model, tasks="mmlu", num_fewshot=None,
                                                    batch_size=mmlu_batch_size, random_seed=0,
                                                    numpy_random_seed=1234, torch_random_seed=1234)
            phase_key = "mmlu_0shot"
        accuracy = mmlu_result["results"]["mmlu"]["acc,none"] * 100
        append_result(output_json, {**result_base, "phase": "mmlu", phase_key: round(accuracy, 2),
                                    "seconds": round(time.time() - start_time)})
