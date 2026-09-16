# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# Granite iOS (chunked-static / ANE) model class.
#
# This is the DENSE `granite` model_type (GraniteForCausalLM). It is NOT granitemoe,
# granitemoeshared or granitemoehybrid -- 4.0-h and 4.0-micro are Mamba hybrids and have no
# path to the ANE at all.
#
# Ported from `ios/mistral.py`. Granite is llama plus four scalar knobs, and all four are
# implemented here even though granite-4.2-3b sets three of them to 1.0, because granite-3.x
# does not and a silently-ignored multiplier is the exact class of failure Rule 2b describes:
# the bundle compiles, verifies 30/30 and decodes confident nonsense.
#
#   attention_multiplier -> the SDPA scale, REPLACING 1/sqrt(head_dim).
#                           granite-4.2-3b uses 0.015625 = 1/64 where 1/sqrt(64) = 0.125,
#                           an 8x difference, so this one is never a no-op.
#   embedding_multiplier -> token embeddings scaled before the first block.
#   residual_multiplier  -> both residual adds per block: h = x + r * mult.
#   logits_scaling       -> logits DIVIDED by it on the way out (HF: "main diff with Llama").
#
# Reference: transformers/models/granite/modeling_granite.py (self.scaling = attention_
# multiplier; hidden_states = residual + hidden_states * self.residual_multiplier;
# inputs_embeds = inputs_embeds * self.embedding_multiplier; logits = logits / logits_scaling).
#
# wangqi modified 2026-09-16

import torch
import torch.nn as nn
from transformers.models.granite.modeling_granite import (
    GraniteConfig,
)
from transformers.models.granite.modeling_granite import (
    GraniteForCausalLM as HFGraniteForCausalLM,
)

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.mlp import MLP
from coreai_models.primitives.ios.quantization import (
    dequantize_per_tensor,
    quantize_per_tensor,
)
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import RoPECache, apply_rope
from coreai_models.primitives.ios.sdpa import SDPA


class Attention(nn.Module):
    def __init__(self, config: GraniteConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", None) or dim // n_heads
        attention_bias = getattr(config, "attention_bias", False)

        self.q_proj = nn.Conv2d(dim, n_heads * head_dim, kernel_size=1, bias=attention_bias)
        self.k_proj = nn.Conv2d(dim, n_kv_heads * head_dim, kernel_size=1, bias=attention_bias)
        self.v_proj = nn.Conv2d(dim, n_kv_heads * head_dim, kernel_size=1, bias=attention_bias)

        self.o_proj = nn.Conv2d(n_heads * head_dim, dim, kernel_size=1, bias=False)

        # The one knob that is never a no-op. SDPA already accepts an explicit scale; without
        # it the default is head_dim ** -0.5, which granite does not use.
        attention_multiplier = getattr(config, "attention_multiplier", None)
        self.sdpa = SDPA(head_dim=self.head_dim, scale=attention_multiplier)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _, hidden_size = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        x = x.transpose(-3, -1)
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = (
            query.transpose(-3, -1)
            .reshape(batch_size, query_len, n_heads, self.head_dim)
            .transpose(-2, -3)
        )
        key = (
            key.transpose(-3, -1)
            .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
            .transpose(-2, -3)
        )

        seq_len = rope_cos.shape[1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)

        query = apply_rope(query, rope_cos, rope_sin)
        key = apply_rope(key, rope_cos, rope_sin)

        query = (
            query.transpose(-2, -3)
            .reshape(batch_size, query_len, 1, n_heads * self.head_dim)
            .transpose(-3, -1)
        )
        key = (
            key.transpose(-3, -2)
            .reshape(batch_size, query_len, 1, n_kv_heads * self.head_dim)
            .transpose(-3, -1)
        )

        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx,
                in_step,
                key,
                value,
                query_len,
            )

        output = self.sdpa(query, key, value, causal_mask)
        output = self.o_proj(output)
        return output.transpose(-3, -1)


class TransformerBlock(nn.Module):
    def __init__(self, config: GraniteConfig, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(dim=hidden_size, hidden_dim=config.intermediate_size)

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        self.residual_multiplier = float(getattr(config, "residual_multiplier", 1.0) or 1.0)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(
            self.input_layernorm(x),
            rope_cos,
            rope_sin,
            in_step,
            causal_mask,
            cache,
        )
        h = x + r * self.residual_multiplier
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r * self.residual_multiplier


class GraniteModel(nn.Module):
    def __init__(self, config: GraniteConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            token_embeddings = layer(
                token_embeddings,
                rope_cos,
                rope_sin,
                in_step,
                causal_mask,
                cache,
            )
        return self.norm(token_embeddings)


class GraniteExtend(nn.Module):
    def __init__(self, config: GraniteConfig):
        super().__init__()
        self.model = GraniteModel(config)
        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.emb_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)

        self.prefill_mode = False

        self.embedding_multiplier = float(getattr(config, "embedding_multiplier", 1.0) or 1.0)
        self.logits_scaling = float(getattr(config, "logits_scaling", 1.0) or 1.0)

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        self.kv_cache = KVCacheHandler(config.num_hidden_layers, config.hidden_size)

        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        rope_theta = resolve_rope_theta(config)
        self.rope = RoPECache(head_dim, config.max_position_embeddings, rope_theta)

    def forward(
        self,
        transformer_input: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        embedding_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.kv_cache.register_kv_cache(key_cache, value_cache)
        rope_cos, rope_sin = self.rope.gather_cos_sin(position_ids)

        batch_size, seq_len, _, hidden_dim = transformer_input.shape

        # HF scales inputs_embeds before the first decoder layer; transformer_input IS the
        # gathered (and already dequantized) token embeddings on this path.
        if self.embedding_multiplier != 1.0:
            transformer_input = transformer_input * self.embedding_multiplier

        out = self.model(
            transformer_input,
            rope_cos,
            rope_sin,
            in_step,
            causal_mask,
            self.kv_cache,
        )
        if self.prefill_mode:
            # No logits are produced in prefill mode, so logits_scaling does not apply here.
            return self.kv_cache.k_cache[0, 0, 0, 0, 0] + self.kv_cache.v_cache[0, 0, 0, 0, 0]

        if self.lm_head is not None:
            logits = self.lm_head(out.transpose(-2, -3))
            if self.logits_scaling != 1.0:
                logits = logits / self.logits_scaling
            return logits

        if embedding_table.dtype == torch.int8:
            embedding_table = dequantize_per_tensor(
                embedding_table,
                self.emb_scale,
                self.emb_zero_point,
                out.dtype,
            )

        embedding_table = embedding_table.reshape(
            embedding_table.shape[1], embedding_table.shape[0], embedding_table.shape[2]
        )

        out = out.transpose(-3, -1).reshape(batch_size, 1, hidden_dim, seq_len)
        logits = (embedding_table @ out).permute(0, 3, 1, 2)
        if self.logits_scaling != 1.0:
            logits = logits / self.logits_scaling
        return logits


class GraniteForCausalLMForiOS(BaseForCausalLMForiOS):
    _HF_MODEL_CLASS = HFGraniteForCausalLM

    def _init_model(self, config: GraniteConfig) -> None:
        self.extend = GraniteExtend(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = self.gather_embeddings(input_ids, self.load_embeddings.embedding_table)
        return self.extend(
            token_embeddings,
            position_ids,
            in_step,
            causal_mask,
            key_cache,
            value_cache,
            self.load_embeddings.embedding_table,
        )

    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        max_layer = -1
        for k in state_dict:
            name_split = k.split(".")
            if len(name_split) != 6:
                continue
            if not k.startswith("model.layers."):
                continue
            max_layer = max(max_layer, int(name_split[2]))

        if max_layer < 0:
            err = "invalid state_dict"
            raise ValueError(err)

        for i in range(max_layer + 1):
            # Reshape attention weights for Conv2d
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                weight_key = f"model.layers.{i}.self_attn.{proj}.weight"
                state_dict[weight_key] = state_dict[weight_key].unsqueeze(-1).unsqueeze(-1)

            # Reshape MLP weights for Conv2d
            for proj in ["up_proj", "gate_proj", "down_proj"]:
                weight_key = f"model.layers.{i}.mlp.{proj}.weight"
                state_dict[weight_key] = state_dict[weight_key].unsqueeze(-1).unsqueeze(-1)

            if not getattr(self.config, "attention_bias", False):
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    state_dict.pop(f"model.layers.{i}.self_attn.{proj}.bias", None)

        # Handle embeddings
        embedding_table = state_dict["model.embed_tokens.weight"].unsqueeze(1)
        if not self.disable_embedding_quantization:
            embedding_table, scale, zero_point = quantize_per_tensor(
                embedding_table, nbits=8, symmetric=True
            )
        else:
            scale = torch.tensor(1.0, dtype=embedding_table.dtype)
            zero_point = torch.tensor(0, dtype=torch.int8)

        state_dict["load_embeddings.embedding_table"] = embedding_table
        state_dict["gather_embeddings.scale"] = scale
        state_dict["gather_embeddings.zero_point"] = zero_point
        state_dict["extend.emb_scale"] = scale
        state_dict["extend.emb_zero_point"] = zero_point

        state_dict.pop("model.embed_tokens.weight")

        # GraniteModel is held inside GraniteExtend — add "extend." prefix
        new_state_dict = {}
        keys_to_pop = set()

        for k, _v in state_dict.items():
            if k.startswith("model.") and "gather_embeddings" not in k:
                new_key = f"extend.{k}"
                new_state_dict[new_key] = state_dict[k]
                keys_to_pop.add(k)

        for k in keys_to_pop:
            state_dict.pop(k)
        state_dict.update(new_state_dict)

        if not self.config.tie_word_embeddings:
            state_dict["extend.lm_head.weight"] = state_dict["lm_head.weight"]

        state_dict.pop("lm_head.weight", None)
