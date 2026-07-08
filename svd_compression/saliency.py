"""Per-expert REAP saliency (following Cerebras REAP, arXiv:2510.13999 Eq. 9).

REAP saliency of expert i = mean over calibration tokens of  gate_i(x) * ||expert_i(x)||_2 , where
gate_i is the router-softmax probability and the mean is over the top-k-routed tokens. Low saliency =
weak expert (rarely / weakly routed, small output) -> the first candidate for compression.

precompute_saliency() runs the REAL model.forward() over a calibration batch with hooks on each MoE
layer (this is the faithful path used to pick experts; for hybrid-SSM Qwen3.6 it avoids a hand-rolled
forward that mishandles the SSM layers). Saliency is computed inside the hook and only the tiny
per-expert vector is kept, so holding all layers' per-token expert outputs at once (tens of GB) is
avoided.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


def reap_saliency(router_logits, expert_outputs, top_k=8):
    """REAP saliency (Cerebras REAP arXiv:2510.13999 Eq. 9). Faithful to the reference implementation.

    Copied from https://github.com/SamsungSAILMontreal/ream/blob/main/ream/saliency.py (function `reap`;
    renamed here, otherwise unchanged for the top_k>0 path).

    router_logits:  [B, S, E] router logits.
    expert_outputs: [E, B*S, H] per-token output of every expert.
    Returns [E] saliency: for each expert, the mean over the tokens ROUTED to it (top-k) of
    gate_probability * ||output||_2. The mean is over that expert's routed-token set, so saliency is the
    average gated output magnitude WHEN the expert is used.
    """
    num_experts = router_logits.shape[-1]
    assert expert_outputs.dim() == 3 and expert_outputs.shape[0] == num_experts, \
        (expert_outputs.shape, router_logits.shape)
    gate = functional.softmax(router_logits.view(-1, num_experts), dim=-1, dtype=torch.float)   # [B*S, E]
    gate, selected = torch.topk(gate, k=top_k, dim=-1)
    expert_mask = functional.one_hot(selected, num_classes=num_experts).permute(2, 1, 0)         # [E, B*S, top_k]
    routed_experts = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    saliency = torch.zeros(num_experts)
    for expert_idx in routed_experts:
        expert_idx = expert_idx.item()
        token_idx, slot_idx = torch.where(expert_mask[expert_idx])          # tokens routed to this expert
        routed_output = expert_outputs[expert_idx][None, token_idx].reshape(-1, expert_outputs.shape[-1])
        saliency[expert_idx] += (routed_output.norm(dim=-1) * gate[token_idx, slot_idx]).mean().item()
    return saliency


@torch.no_grad()
def precompute_saliency(model, adapter, calibration, output_json,
                        num_calib_sequences=128, batch_size=4, device="cuda:0"):
    """Compute + cache per-expert REAP saliency for every MoE layer via the real forward.
    Writes {layer_index: {"saliency": [...], "weakest_order": [ascending expert index]}} to output_json."""
    decoder_layers = adapter.decoder_layers()
    num_experts, top_k = adapter.num_experts, adapter.top_k
    input_ids = calibration["input_ids"][:num_calib_sequences]
    attention_mask = calibration["attention_mask"][:num_calib_sequences]
    num_batches = max(1, num_calib_sequences // batch_size)
    accumulated_saliency = {layer_index: 0.0 for layer_index in range(len(decoder_layers))}
    captured_router_logits = {}

    def make_router_hook(layer_index):
        def hook(module, inputs, output):
            hidden_states = inputs[0]
            flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
            captured_router_logits[layer_index] = adapter.router_logits(decoder_layers[layer_index], flat_hidden)
        return hook

    def make_experts_hook(layer_index):
        def hook(module, inputs, output):
            flat_hidden = inputs[0].reshape(-1, inputs[0].shape[-1])
            router_logits = captured_router_logits[layer_index]
            gate_up_proj, down_proj = module.gate_up_proj, module.down_proj
            expert_outputs = None
            for expert in range(num_experts):
                gate_activation, up_activation = functional.linear(flat_hidden, gate_up_proj[expert]).chunk(2, dim=-1)
                expert_output = functional.linear(module.act_fn(gate_activation) * up_activation, down_proj[expert])
                if expert_outputs is None:
                    expert_outputs = torch.zeros(num_experts, flat_hidden.shape[0], expert_output.shape[1],
                                                 dtype=expert_output.dtype, device=expert_output.device)
                expert_outputs[expert] = expert_output
            accumulated_saliency[layer_index] = (
                accumulated_saliency[layer_index]
                + reap_saliency(router_logits, expert_outputs, top_k=top_k) / num_batches)
            del expert_outputs
        return hook

    # hook the router first (fires before the experts) then the experts module
    hooks = []
    for layer_index, layer in enumerate(decoder_layers):
        router_module = getattr(layer, "router", None) or layer.mlp.gate
        hooks.append(router_module.register_forward_hook(make_router_hook(layer_index)))
        hooks.append(adapter.expert_module(layer).register_forward_hook(make_experts_hook(layer_index)))

    start_time = time.time()
    for batch_start in range(0, num_calib_sequences, batch_size):
        captured_router_logits.clear()
        model(input_ids=input_ids[batch_start:batch_start + batch_size].to(device),
              attention_mask=attention_mask[batch_start:batch_start + batch_size].to(device), use_cache=False)
        if batch_start % (batch_size * 8) == 0:
            print(f"  saliency chunk {batch_start}/{num_calib_sequences} ({time.time()-start_time:.0f}s)", flush=True)
    for hook in hooks:
        hook.remove()

    saliency_by_layer = {}
    for layer_index in range(len(decoder_layers)):
        saliency = np.asarray(accumulated_saliency[layer_index].cpu())
        saliency_by_layer[str(layer_index)] = {"saliency": saliency.tolist(),
                                               "weakest_order": np.argsort(saliency).tolist()}
    Path(output_json).write_text(json.dumps(saliency_by_layer))
    print(f"saved saliency -> {output_json} ({time.time()-start_time:.0f}s)", flush=True)
    return saliency_by_layer
