import torch
from transformers.models.llama.modeling_llama import *
from transformers.models.gpt_neox.modeling_gpt_neox import *
import numpy as np
import torch.nn as nn
import math
from typing import Optional, Tuple

from .selfextend_flash_attn import self_extend_flash_forward

from transformers.utils import is_flash_attn_2_available
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input

from transformers.cache_utils import Cache

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin) if not q is None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if not k is None else None
    return q_embed, k_embed


def flash_self_extend_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 1024,
    scale_base: Optional[int] = -1,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
        Require updating tansformers to >= 4.38.1, flash_attn to the newest version

        a. Only support causal mask.
        b. Don't support atttention_mask.
        c. Never test it with batch size > 1.
        d. Only support q_len = 1 or q_len = seq_len.
    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop("padding_mask")

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if scale_base > 0:
        scaled_query = query_states * (((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1)).to(query_states.dtype)
        #scaled_query = query_states * (((0.1*(((position_ids+1)[:, None, :, None]/scale_base).log())+1)**2).clip(1)).to(query_states.dtype)
    else:
        scaled_query = query_states
    
    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len) # only consider bsz=1 for now.

    attn_dropout = self.config.attention_dropout if self.training else 0.0

    if q_len == 1:
        neighbor_key_position = position_ids[:, -1] - key_position
        _re_group_size_2 = 0 if position_ids.max() < group_size_2 else group_size_2
        group_key_position = position_ids[:, -1]//group_size_1 - key_position//group_size_1 + (_re_group_size_2 - _re_group_size_2//group_size_1)
        decode_key_position = torch.cat([group_key_position[:, :-group_size_2], neighbor_key_position[:,-group_size_2:]], dim=1)
        
        decode_k_cos, decode_k_sin = self.rotary_emb(value_states, decode_key_position, seq_len=None)
        decode_query_states = scaled_query.transpose(1,2).contiguous() # position 0: cos 0 = 1, sin 0 = 0
        _, decode_key_states = apply_rotary_pos_emb(None, key_states, decode_k_cos, -decode_k_sin, decode_key_position) 

        decode_key_states = repeat_kv(decode_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        decode_value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        
        attn_output = flash_attn_func(decode_query_states,
                                      decode_key_states,
                                      decode_value_states,
                                      attn_dropout, 
                                      softmax_scale=None, 
                                      causal=True)
    elif q_len == kv_seq_len:
        neighbor_q_cos, neighbor_q_sin = self.rotary_emb(value_states, query_position, seq_len=None)
        neighbor_k_cos, neighbor_k_sin = self.rotary_emb(value_states, key_position, seq_len=None)


        _re_group_size_2 = 0 if query_position.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
        group_query_position = query_position // group_size_1 + _re_group_size_2 - _re_group_size_2 / group_size_1
        group_key_position = key_position // group_size_1

        group_q_cos, group_q_sin = self.rotary_emb(value_states, group_query_position, seq_len=None)
        group_k_cos, group_k_sin = self.rotary_emb(value_states, group_key_position, seq_len=None)



        neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, neighbor_q_cos, neighbor_q_sin, None)
        _, neighbor_key_states = apply_rotary_pos_emb(None, key_states, neighbor_k_cos, neighbor_k_sin, None)
        group_query_states, _ = apply_rotary_pos_emb(scaled_query, None, group_q_cos, group_q_sin, None)
        _, group_key_states = apply_rotary_pos_emb(None, key_states, group_k_cos, group_k_sin, None)
        

        neighbor_query_states = neighbor_query_states.transpose(1, 2)
        neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups).transpose(1, 2)
        group_query_states = group_query_states.transpose(1, 2)
        group_key_states = repeat_kv(group_key_states, self.num_key_value_groups).transpose(1, 2)
        value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2)


        attn_dropout = self.config.attention_dropout if self.training else 0.0


        attn_output = self_extend_flash_forward(self,
                                                query_position,
                                                group_size_2,
                                                neighbor_query_states,
                                                neighbor_key_states,
                                                group_query_states,
                                                group_key_states,
                                                value_states,
                                                attention_mask,
                                                bsz,
                                                q_len,
                                                kv_seq_len,
                                                attn_dropout,
                                            )
    else:
        raise ValueError("q_len should be 1 or seq_len.")

    attn_output = attn_output.contiguous()
    attn_output = attn_output.view(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
       

def self_extend_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 1024,
    scale_base: Optional[int] = -1,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if scale_base > 0:
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype) # log scale 
        #scaled_query = query_states * (((0.1*(((position_ids+1)[:, None, :, None]/scale_base).log())+1)**2).clip(1)).to(query_states.dtype) # Yarn scale 
    else:
        scaled_query = query_states
    
    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len) # only consider bsz=1 for now.



    neighbor_q_cos, neighbor_q_sin = self.rotary_emb(value_states, query_position, seq_len=None)
    neighbor_k_cos, neighbor_k_sin = self.rotary_emb(value_states, key_position, seq_len=None)


    _re_group_size_2 = 0 if query_position.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
    group_query_position = query_position // group_size_1 + _re_group_size_2 - _re_group_size_2 / group_size_1
    group_key_position = key_position // group_size_1

    group_q_cos, group_q_sin = self.rotary_emb(value_states, group_query_position, seq_len=None)
    group_k_cos, group_k_sin = self.rotary_emb(value_states, group_key_position, seq_len=None)



    neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, neighbor_q_cos, neighbor_q_sin, None)
    _, neighbor_key_states = apply_rotary_pos_emb(None, key_states, neighbor_k_cos, neighbor_k_sin, None)
    group_query_states, _ = apply_rotary_pos_emb(scaled_query, None, group_q_cos, group_q_sin, None)
    _, group_key_states = apply_rotary_pos_emb(None, key_states, group_k_cos, group_k_sin, None)



    neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups)
    group_key_states = repeat_kv(group_key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)



    neighbor_attn_weights = torch.matmul(neighbor_query_states, neighbor_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    group_attn_weights = torch.matmul(group_query_states, group_key_states.transpose(2, 3)) / math.sqrt(self.head_dim) 


    if attention_mask is not None:  # no matter the length, we just slice it
        if cache_position is not None:
            causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
        else:
            causal_mask = attention_mask
        group_attn_weights = group_attn_weights + causal_mask
        neighbor_attn_weights = neighbor_attn_weights + causal_mask

    if q_len == 1:
        neighbor_attention_mask = torch.zeros((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        neighbor_attention_mask[:, -group_size_2:] = 1
    elif q_len == kv_seq_len:
        neighbor_attention_mask = torch.ones((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        neighbor_attention_mask = torch.tril(neighbor_attention_mask)
        if q_len-group_size_2 > 0:
            group_attention_mask =  torch.tril(torch.ones((q_len-group_size_2, kv_seq_len-group_size_2), device=group_attn_weights.device))
            neighbor_attention_mask[group_size_2:, :-group_size_2] -= group_attention_mask
    else:
        raise ValueError("q_len should be 1 or seq_len.")
    
    neighbor_attention_mask = neighbor_attention_mask.bool()
    attn_weights = torch.where(neighbor_attention_mask, neighbor_attn_weights, group_attn_weights)
    
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)


    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
