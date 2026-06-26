# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


class LoRALinear(nn.Module):
    """Low-rank update injected alongside a frozen projection (q/k/v of GPT2's c_attn).

    delta(x) = (x @ A @ B) * (alpha / rank). B is zero-initialized so the adapter
    starts as a no-op and the pretrained model's behavior is unchanged at step 0.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A @ self.lora_B) * self.scaling


def reset_lora_parameters(model: nn.Module) -> None:
    """Re-initialize every LoRALinear adapter in place.

    `from_pretrained`'s fast-init path materializes any parameter missing from the
    checkpoint (lora_A/lora_B are never in the GPT2 checkpoint) straight from the
    meta device without going through LoRALinear.__init__, since GPT2's _init_weights
    doesn't recognize LoRALinear and leaves it untouched. Without this call, lora_A
    ends up as near-zero garbage instead of its intended Kaiming init, so the adapter
    never escapes a zero/zero deadlock. Call this once right after from_pretrained.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.reset_parameters()


class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, rank, alpha, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        # LoRA adapters for the query, key and value projections of c_attn/q_attn
        self.lora_q = LoRALinear(self.embed_dim, self.embed_dim, rank, alpha)
        self.lora_k = LoRALinear(self.embed_dim, self.embed_dim, rank, alpha)
        self.lora_v = LoRALinear(self.embed_dim, self.embed_dim, rank, alpha)

    # edit the forward method to implement LoRa
    # from transformers 4.38.0
    # https://github.com/huggingface/transformers/blob/v4.38.0/src/transformers/models/gpt2/modeling_gpt2.py
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states) + self.lora_q(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            key = key + self.lora_k(encoder_hidden_states)
            value = value + self.lora_v(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            query = query + self.lora_q(hidden_states)
            key = key + self.lora_k(hidden_states)
            value = value + self.lora_v(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2_LoRA(GPT2LMHeadModel):
    def __init__(self, *model_args, rank, alpha, **model_kwargs):
        super().__init__(*model_args, **model_kwargs)
        # substitute every block's attn with a LoRA-augmented instance,
        # keeping the (pretrained) c_attn/c_proj weights via load_state_dict
        for block in self.transformer.h:
            old_attn = block.attn
            new_attn = CustomGPT2Attention(self.config, rank=rank, alpha=alpha)
            new_attn.load_state_dict(old_attn.state_dict(), strict=False)
            block.attn = new_attn

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
