"""Compress the weakest MoE experts of a model by LOW-RANK approximation, then evaluate GSM8K + MMLU.

Methods:
  svd       : plain truncated-SVD reconstruction of each weak expert's gate/up/down  (data-agnostic)
  actaware  : activation-aware low-rank (gate-weighted least squares over calibration activations)

"weakest" = lowest saliency (router-softmax-weighted L2 norm of expert output; see saliency.py).
Compress the bottom `frac` fraction of experts per layer. `rank` is the retained rank
(typically intermediate_dim / 4 for -25%, intermediate_dim / 8 for -50%).

(Pruning/merging baselines like REAP and REAM are separate published methods and are NOT implemented
here; the results/ tables cite their numbers only for comparison.)

Example:
  python -m svd_compression.compress \
      --model /path/to/Qwen3.6-35B-A3B --method actaware --frac 0.25 --rank 128 \
      --saliency saliency/qwen36_expert_saliency.json --calib calibration/paper_calib_qwen36.pt \
      --label qwen36-actaware-w25-r128 --out results/qwen36.json --eval gsm8k mmlu

Notes:
  - SVD reconstructs IN PLACE (keeps full model size). For the 35B this means the 248k-vocab logit head
    OOMs a single 80GB GPU during MMLU -> pass --shard to spread the model over 2 GPUs.
  - Qwen has bos_token_id=None -> MMLU is 0-shot no-chat (paper eval_mc protocol). Gemma REQUIRES <bos>
    -> its plain-text MMLU is broken; evaluate.py uses 5-shot chat-template for Gemma automatically.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lowrank import plain_svd, activation_aware_svd
from .models import build_adapter
from .saliency import precompute_saliency
from . import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["svd", "actaware"], required=True)
    parser.add_argument("--frac", type=float, default=0.25, help="fraction of weakest experts to compress")
    parser.add_argument("--rank", type=int, default=128, help="retained rank for svd/actaware")
    parser.add_argument("--saliency", required=True, help="path to saliency json (built by --build-saliency)")
    parser.add_argument("--calib", default=None, help="path to calibration .pt (needed for actaware / building saliency)")
    parser.add_argument("--build-saliency", action="store_true", help="compute+cache saliency then continue")
    parser.add_argument("--num-calib-sequences", type=int, default=64, help="calib sequences used for the actaware fit")
    parser.add_argument("--num-iterations", type=int, default=3, help="actaware alternating-least-squares iterations")
    parser.add_argument("--max-tokens-per-layer", type=int, default=20000, help="cap collected tokens per layer")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eval", nargs="*", default=[], choices=["gsm8k", "mmlu"], help="which evals to run")
    parser.add_argument("--shard", action="store_true", help="device_map=auto (spread over all visible GPUs)")
    parser.add_argument("--gsm8k-batch-size", type=int, default=16)
    parser.add_argument("--mmlu-batch-size", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def collect_activations(model, adapter, calibration, num_calib_sequences, max_tokens_per_layer, device):
    """Collect per-layer MoE-input hidden states over the calibration batch (for the actaware fit).
    Returns {layer_index: [hidden_states_chunk, ...]} on CPU, capped at max_tokens_per_layer per layer."""
    decoder_layers = adapter.decoder_layers()
    captured_hidden = {}
    collected_hidden = {layer_index: [] for layer_index in range(len(decoder_layers))}

    def make_hook(layer_index):
        def hook(module, args, kwargs):
            captured_hidden[layer_index] = (args[0] if args else kwargs.get("hidden_states")).detach()
        return hook

    # experts pre-hook input is the MoE-block input for Gemma (experts called directly) and the
    # already-flattened hidden states for Qwen; both reshape to [num_tokens, hidden] below.
    hooks = [adapter.expert_module(decoder_layers[layer_index]).register_forward_pre_hook(
                make_hook(layer_index), with_kwargs=True)
             for layer_index in range(len(decoder_layers))]
    input_ids = calibration["input_ids"][:num_calib_sequences]
    attention_mask = calibration["attention_mask"][:num_calib_sequences]
    for batch_start in range(0, num_calib_sequences, 8):
        captured_hidden.clear()
        model(input_ids=input_ids[batch_start:batch_start + 8].to(device),
              attention_mask=attention_mask[batch_start:batch_start + 8].to(device), use_cache=False)
        for layer_index in range(len(decoder_layers)):
            hidden_states = captured_hidden[layer_index].reshape(-1, captured_hidden[layer_index].shape[-1])
            if sum(chunk.shape[0] for chunk in collected_hidden[layer_index]) < max_tokens_per_layer:
                collected_hidden[layer_index].append(hidden_states.cpu())
    for hook in hooks:
        hook.remove()
    return collected_hidden


def main():
    args = parse_args()
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    load_kwargs = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = (AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", **load_kwargs)
             if args.shard else AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(device))
    model.eval()
    adapter = build_adapter(model)
    num_experts = adapter.num_experts
    intermediate_dim = adapter.intermediate_dim
    num_layers = adapter.num_layers
    num_weak_experts = int(round(num_experts * args.frac))
    print(f"{model.__class__.__name__}: layers={num_layers} experts={num_experts} "
          f"intermediate_dim={intermediate_dim} top_k={adapter.top_k} | method={args.method} "
          f"frac={args.frac} num_weak={num_weak_experts} rank={args.rank}", flush=True)

    # --- saliency (build or load) ---
    if args.build_saliency:
        assert args.calib, "--calib required to build saliency"
        calibration = torch.load(args.calib)
        precompute_saliency(model, adapter, calibration, args.saliency,
                            num_calib_sequences=args.num_calib_sequences, device=device)
    saliency_by_layer = json.loads(Path(args.saliency).read_text())
    weak_experts_by_layer = {layer_index: set(saliency_by_layer[str(layer_index)]["weakest_order"][:num_weak_experts])
                             for layer_index in range(num_layers)}

    # --- collect activations for actaware ---
    collected_hidden = None
    if args.method == "actaware":
        assert args.calib, "--calib required for actaware"
        calibration = torch.load(args.calib)
        collected_hidden = collect_activations(model, adapter, calibration, args.num_calib_sequences,
                                               args.max_tokens_per_layer, device)
        print("collected activations for actaware fit", flush=True)

    # --- apply low-rank compression to the weakest experts (in place) ---
    start_time = time.time()
    decoder_layers = adapter.decoder_layers()
    with torch.no_grad():
        for layer_index, layer in enumerate(decoder_layers):
            experts = adapter.expert_module(layer)
            if args.method == "actaware":
                expert_device = experts.gate_up_proj.device
                layer_activations = torch.cat(collected_hidden[layer_index], 0).to(expert_device)
                if layer_activations.shape[0] > 6144:
                    subset = torch.randperm(layer_activations.shape[0], device=expert_device)[:6144]
                    layer_activations = layer_activations[subset]
                router_probs = functional.softmax(adapter.router_logits(layer, layer_activations).float(), dim=-1)
            for expert_index in weak_experts_by_layer[layer_index]:
                gate_up_weight = experts.gate_up_proj.data[expert_index]
                gate_weight, up_weight = gate_up_weight[:intermediate_dim], gate_up_weight[intermediate_dim:]
                down_weight = experts.down_proj.data[expert_index]
                if args.method == "svd":
                    experts.gate_up_proj.data[expert_index] = torch.cat(
                        [plain_svd(gate_weight, args.rank), plain_svd(up_weight, args.rank)], 0)
                    experts.down_proj.data[expert_index] = plain_svd(down_weight, args.rank)
                else:
                    gate_probs_for_expert = router_probs[:, expert_index]
                    new_gate = activation_aware_svd(gate_weight, layer_activations, args.rank,
                                                    args.num_iterations, gate_weights=gate_probs_for_expert)
                    new_up = activation_aware_svd(up_weight, layer_activations, args.rank,
                                                  args.num_iterations, gate_weights=gate_probs_for_expert)
                    intermediate_activations = (experts.act_fn(functional.linear(layer_activations, gate_weight))
                                                * functional.linear(layer_activations, up_weight))
                    new_down = activation_aware_svd(down_weight, intermediate_activations, args.rank,
                                                    args.num_iterations, gate_weights=gate_probs_for_expert)
                    experts.gate_up_proj.data[expert_index] = torch.cat([new_gate, new_up], 0)
                    experts.down_proj.data[expert_index] = new_down
            if args.method == "actaware":
                collected_hidden[layer_index] = None
                del layer_activations, router_probs
    gc.collect(); torch.cuda.empty_cache()   # free reconstruction-loop cache before the logit-heavy eval
    print(f"applied {args.method} in {time.time()-start_time:.0f}s; "
          f"gpu mem {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # --- eval ---
    if args.eval:
        evaluate.run_evals(model, tokenizer, adapter, args.eval, args.label, args.out,
                           method=args.method, frac=args.frac, rank=args.rank,
                           gsm8k_batch_size=args.gsm8k_batch_size, mmlu_batch_size=args.mmlu_batch_size)


if __name__ == "__main__":
    main()
