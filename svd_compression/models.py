"""Model adapters: the ONLY architecture-specific bits of the pipeline.

Each adapter exposes a uniform interface over a MoE model so compress.py / saliency.py stay
model-agnostic:
  - decoder_layers()                   : list of decoder layers
  - expert_module(layer)               : packed-expert module (gate_up_proj[E,2I,H], down_proj[E,H,I])
  - router_weight(layer)               : [num_experts, hidden] matrix used to score experts
  - router_logits(layer, hidden)       : [num_tokens, num_experts] router logits (includes router math)
  - top_k, num_experts, num_layers, intermediate_dim
  - num_experts_in(layer)              : expert count of a specific layer

Supported:
  Qwen3.5/3.6 MoE (qwen3_5_moe): standard layer.mlp.{gate(Linear), experts(packed)} + a small shared_expert.
  Gemma-4 MoE (gemma4):          layer.{experts, router} where router = norm+proj+scale+per_expert_scale,
                                 and a PARALLEL dense mlp per layer (left untouched).
"""
import torch
import torch.nn.functional as functional


def build_adapter(model):
    """Return the right adapter for the given loaded MoE model."""
    class_name = model.__class__.__name__.lower()
    architectures = " ".join(getattr(model.config, "architectures", None) or []).lower()
    if "gemma" in class_name or "gemma" in architectures:
        return GemmaAdapter(model)
    return Qwen35Adapter(model)          # default: standard Qwen-style MoE


class Qwen35Adapter:
    """Qwen3.5/3.6 hybrid MoE. MoE block at layer.mlp: .gate (nn.Linear [num_experts, hidden])
    plus .experts (packed)."""
    def __init__(self, model):
        self.model = model
        text_config = model.config.get_text_config()
        self.top_k = text_config.num_experts_per_tok
        self.num_experts = text_config.num_experts
        self._decoder_layers = model.model.layers
        first_layer_experts = self._decoder_layers[0].mlp.experts
        self.intermediate_dim = first_layer_experts.gate_up_proj.shape[1] // 2
        self.num_layers = len(self._decoder_layers)

    def decoder_layers(self):
        return self._decoder_layers

    def expert_module(self, layer):
        return layer.mlp.experts

    def router_weight(self, layer):
        return layer.mlp.gate.weight            # [num_experts, hidden]

    def router_logits(self, layer, hidden_states):
        return functional.linear(hidden_states.float(), layer.mlp.gate.weight.float())   # [num_tokens, num_experts]

    def num_experts_in(self, layer):
        experts = layer.mlp.experts
        return experts.num_experts if hasattr(experts, "num_experts") else experts.gate_up_proj.shape[0]


class GemmaAdapter:
    """Gemma-4 text MoE. Routed experts at layer.experts; router at layer.router (norm+proj+scale).
    Each layer also runs a PARALLEL dense mlp that we never touch (it is the model's capability floor)."""
    def __init__(self, model):
        self.model = model
        text_config = model.config.get_text_config()
        self.top_k = text_config.top_k_experts
        self._decoder_layers = model.model.language_model.layers
        first_layer_experts = self._decoder_layers[0].experts
        self.num_experts = first_layer_experts.num_experts
        self.intermediate_dim = first_layer_experts.gate_up_proj.shape[1] // 2
        self.num_layers = len(self._decoder_layers)

    def decoder_layers(self):
        return self._decoder_layers

    def expert_module(self, layer):
        return layer.experts

    def router_weight(self, layer):
        return layer.router.proj.weight         # [num_experts, hidden]

    def router_logits(self, layer, hidden_states):
        router = layer.router
        normalised = router.norm(hidden_states) * router.scale * router.scalar_root_size
        return router.proj(normalised)          # [num_tokens, num_experts]

    def num_experts_in(self, layer):
        return layer.experts.num_experts
