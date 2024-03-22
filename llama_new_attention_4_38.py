import torch
from transformers.models.llama.modeling_llama import *
from transformers.models.gpt_neox.modeling_gpt_neox import *
import numpy as np
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F
import transformers
from flash_attn import flash_attn_func, flash_attn_varlen_func
from qq_flash_attention import self_extend_flash_forward 
from triton_self_extend import prefill_flash_forward



if transformers.__version__ >= '4.36':
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



def self_extend_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 1024,
    log_scale_base: Optional[int] = -1,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()


    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if log_scale_base > 0:
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(log_scale_base)).clip(1).to(query_states.dtype)
    else:
        scaled_query = query_states
    
    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(bsz, kv_seq_len)


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

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

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
    log_scale_base: Optional[int] = -1,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
        Require updating tansformers to >= 4.38.1, flash_attn to the newest version

        a. Only support causal mask.
        b. Don't support atttention_mask.
        c. Never test it with batch size > 1.
        d. Only support q_len = 1 or q_len = seq_len.
        e. Add support for log-scale, https://arxiv.org/abs/2202.12172 ; https://kexue.fm/archives/8823
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

    if log_scale_base > 0:
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(log_scale_base)).clip(1).to(query_states.dtype)
        #scaled_query = query_states * (((0.1*(((position_ids+1)[:, None, :, None]/log_scale_base).log())+1)**2).clip(1)).to(query_states.dtype)
    else:
        scaled_query = query_states
    
    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    # only consider bsz=1 for now. 
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len)
    attn_dropout = self.config.attention_dropout if self.training else 0.0
    if q_len == 1:
        # We implement the case q_len == 1 separately, by manipulating positions.
        # for our flash implementation doesnot work for  decoding stage at the releasing time.

        neighbor_key_position = position_ids[:, -1] - key_position
        _re_group_size_2 = 0 if position_ids.max() < group_size_2 else group_size_2
        group_key_position = position_ids[:, -1]//group_size_1 - key_position//group_size_1 + (_re_group_size_2 - _re_group_size_2//group_size_1)
        decode_key_position = torch.cat([group_key_position[:, :-group_size_2], neighbor_key_position[:,-group_size_2:]], dim=1)
        
        decode_k_cos, decode_k_sin = self.rotary_emb(value_states, decode_key_position, seq_len=None)
        #import pdb; pdb.set_trace()
        #neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, cos, sin, query_position_ids) 
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
        # set correct position_ids & apply RoPE.
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
        

        #neighbor_query_states = neighbor_query_states.transpose(1, 2).contiguous()
        #neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        #group_query_states = group_query_states.transpose(1, 2).contiguous()
        #group_key_states = repeat_kv(group_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        #value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()

        """
        if query_position.max() >= group_size_2:
            '''
            neighbor_attn_output, neighbor_softmax_lse_right_padded, _ = self._flash_attention_forward(
                neighbor_query_states,
                neighbor_key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=attn_dropout,
                window_size=[group_size_2 - 1, 0],
                # right dim here does not matter and can be -1, or > 0 due to causal mask
                return_attn_probs=True,
            )

            group_attention_len = (
                kv_seq_len - group_size_2
            )  # here we should use kv_seq_len rather than max_kv_len since we have paddings in qkv and attention_mask
            group_attention_mask = attention_mask[:, :group_attention_len] if not attention_mask is None else None
            group_attn_output, group_softmax_lse_right_padded, _ = self._flash_attention_forward(
                group_query_states[:, -group_attention_len:, :, :],
                group_key_states[:, :group_attention_len, :, :],
                value_states[:, :group_attention_len, :, :],
                group_attention_mask,
                group_query_states[:, -group_attention_len:, :, :].shape[1],
                dropout=attn_dropout,
                window_size=[-1, -1],
                return_attn_probs=True,
            )  # note that kv and q's indexing are different! also query size could be different from kv length and very small during generation compared to prefilling

            # compute each seq length after removing the window, some could be 0
            #neighbor_seq_length = torch.sum(attention_mask, axis=1, keepdim=True)  # [batch_size, 1]
            #group_seq_length = torch.sum(attention_mask[:, :group_attention_len], axis=1, keepdim=True)  # [batch_size, 1]

            # compute each seq length after removing the window, some could be 0
            neighbor_seq_length = torch.Tensor([kv_seq_len,]).long().expand(bsz, 1) if attention_mask is None else torch.sum(attention_mask, axis=1, keepdim=True)  # [batch_size, 1]
            group_seq_length = torch.Tensor([group_attention_len,]).long().expand(bsz, 1) if attention_mask is None else torch.sum(attention_mask[:, :group_attention_len], axis=1, keepdim=True)  # [batch_size, 1]

            # do exp to convert logsumexp to sumexp
            neighbor_softmax_lse_right_padded = torch.exp(neighbor_softmax_lse_right_padded)
            group_softmax_lse_right_padded = torch.exp(group_softmax_lse_right_padded)

            # convert align left to align right and convert exp(0) to 0
            neighbor_softmax_lse = torch.zeros_like(neighbor_softmax_lse_right_padded)
            group_softmax_lse = torch.zeros_like(group_softmax_lse_right_padded)
            #import pdb; pdb.set_trace()
            for idx in range(bsz):
                if neighbor_seq_length[idx] > 0:
                    neighbor_softmax_lse[idx, :, -neighbor_seq_length[idx] :] = neighbor_softmax_lse_right_padded[
                        idx, :, : neighbor_seq_length[idx]
                    ]
                if group_seq_length[idx] > 0:
                    group_softmax_lse[idx, :, -group_seq_length[idx] :] = group_softmax_lse_right_padded[
                        idx, :, : group_seq_length[idx]
                    ]

            # attn_output size is [batch_size, max_seq_len (not the true one), query_length, dim]
            true_neighbor_seq_max_length = neighbor_softmax_lse.shape[
                -1
            ]  # it could be smaller than query_length due to the attention_mask
            true_group_seq_max_length = group_softmax_lse.shape[
                -1
            ]  # it could be smaller than group_query_layer[:, -group_attention_len:, :, :].shape[1] due to the attention_mask[:, :group_attention_len]

            neighbor_softmax_lse = neighbor_softmax_lse.transpose(1, 2).unsqueeze(
                -1
            )  # [batch_size, true_neighbor_seq_max_length, self.num_heads, 1]
            group_softmax_lse = group_softmax_lse.transpose(1, 2).unsqueeze(
                -1
            )  # [batch_size, true_group_seq_max_length, self.num_heads, 1]
            rescale_lse = torch.empty_like(neighbor_softmax_lse).copy_(neighbor_softmax_lse)

            rescale_lse[:, -true_group_seq_max_length:, :, :] += group_softmax_lse


            # recale the output
            neighbor_attn_output[:, -true_neighbor_seq_max_length:, ...] = (
                neighbor_attn_output[:, -true_neighbor_seq_max_length:, ...] * neighbor_softmax_lse
            )
            group_attn_output[:, -true_group_seq_max_length:, ...] = (
                group_attn_output[:, -true_group_seq_max_length:, ...] * group_softmax_lse
            )
            attn_output = torch.empty_like(neighbor_attn_output).copy_(
                neighbor_attn_output
            )  # might be slightly faster than clone
            attn_output[:, group_size_2-kv_seq_len:, ...] += group_attn_output
            attn_output[:, -true_neighbor_seq_max_length:, ...] /= rescale_lse
            attn_output = torch.nan_to_num(attn_output, nan=0)  # rescale_lse could be 0 for padding tokens causing nan
            '''
            neighbor_attn_output, neighbor_softmax_lse_right_padded, neighbor_prob = self._flash_attention_forward(
                neighbor_query_states,
                neighbor_key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=attn_dropout,
                window_size=[group_size_2 - 1, 0],
                # right dim here does not matter and can be -1, or > 0 due to causal mask
                return_attn_probs=True,
            )

            group_attention_len = (
                kv_seq_len - group_size_2
            )  # here we should use kv_seq_len rather than max_kv_len since we have paddings in qkv and attention_mask

            group_attention_mask = attention_mask[:, :group_attention_len] if not attention_mask is None else None
            group_attn_output, group_softmax_lse_right_padded, group_prob = self._flash_attention_forward(
                group_query_states[:, -group_attention_len:, :, :],
                group_key_states[:, :group_attention_len, :, :],
                value_states[:, :group_attention_len, :, :],
                group_attention_mask,
                group_query_states[:, -group_attention_len:, :, :].shape[1],
                dropout=attn_dropout,
                window_size=[-1, -1],
                return_attn_probs=True,
            )  # note that kv and q's indexing are different! also query size could be different from kv length and very small during generation compared to prefilling


            # compute each seq length after removing the window, some could be 0
            neighbor_seq_length = torch.Tensor([kv_seq_len,]).long().expand(bsz, 1) if attention_mask is None else torch.sum(attention_mask, axis=1, keepdim=True)  # [batch_size, 1]
            group_seq_length = torch.Tensor([group_attention_len,]).long().expand(bsz, 1) if attention_mask is None else torch.sum(attention_mask[:, :group_attention_len], axis=1, keepdim=True)  # [batch_size, 1]

            # convert align left to align right and convert exp(0) to 0
            neighbor_softmax_lse = torch.zeros_like(neighbor_softmax_lse_right_padded)
            group_softmax_lse = torch.zeros_like(group_softmax_lse_right_padded)
            for idx in range(bsz):
                if neighbor_seq_length[idx] > 0:
                    neighbor_softmax_lse[idx, :, -neighbor_seq_length[idx] :] = neighbor_softmax_lse_right_padded[
                        idx, :, : neighbor_seq_length[idx]
                    ]
                if group_seq_length[idx] > 0:
                    group_softmax_lse[idx, :, -group_seq_length[idx] :] = group_softmax_lse_right_padded[
                        idx, :, : group_seq_length[idx]
                    ]

            # attn_output size is [batch_size, max_seq_len (not the true one), query_length, dim]
            true_neighbor_seq_max_length = neighbor_softmax_lse.shape[
                -1
            ]  # it could be smaller than query_length due to the attention_mask
            true_group_seq_max_length = group_softmax_lse.shape[
                -1
            ]  # it could be smaller than group_query_layer[:, -group_attention_len:, :, :].shape[1] due to the attention_mask[:, :group_attention_len]

            neighbor_softmax_lse = neighbor_softmax_lse.transpose(1, 2).unsqueeze(
                -1
            )  # [batch_size, true_neighbor_seq_max_length, self.num_heads, 1]
            group_softmax_lse = group_softmax_lse.transpose(1, 2).unsqueeze(
                -1
            )  # [batch_size, true_group_seq_max_length, self.num_heads, 1]

            '''
            half_lse_gap =  neighbor_softmax_lse[:, -true_group_seq_max_length:, :, :] / 2 - group_softmax_lse / 2 

            neighbor_softmax_lse[:, -true_group_seq_max_length:, :, :] = torch.exp(half_lse_gap)
            neighbor_softmax_lse[:, :-true_group_seq_max_length, :, :] = torch.exp(neighbor_softmax_lse[:, :-true_group_seq_max_length, :, :])
            group_softmax_lse = torch.exp(-half_lse_gap)

            rescale_lse = torch.empty_like(neighbor_softmax_lse).copy_(neighbor_softmax_lse)
            rescale_lse[:, -true_group_seq_max_length:, :, :] += group_softmax_lse
            if  torch.isinf(rescale_lse).any() or torch.isnan(rescale_lse).any():
                import pdb; pdb.set_trace()

            # recale the output
            neighbor_attn_output[:, -true_neighbor_seq_max_length:, ...] = (
                neighbor_attn_output[:, -true_neighbor_seq_max_length:, ...] * neighbor_softmax_lse
            )
            group_attn_output[:, -true_group_seq_max_length:, ...] = (
                group_attn_output[:, -true_group_seq_max_length:, ...] * group_softmax_lse
            )
            attn_output = torch.empty_like(neighbor_attn_output).copy_(
                neighbor_attn_output
            )  # might be slightly faster than clone
            #attn_output[:, group_size_2:, ...] += group_attn_output # will not work for decoding, due to : kv_seq_len > query)
            attn_output[:, group_size_2 - kv_seq_len :, ...] += group_attn_output 
            attn_output[:, -true_neighbor_seq_max_length:, ...] /= rescale_lse
            attn_output = torch.nan_to_num(attn_output, nan=0)  # rescale_lse could be 0 for padding tokens causing nan

            '''

            
            lse_gap = group_softmax_lse - neighbor_softmax_lse[:, -true_group_seq_max_length:, :, :]
            if  torch.isinf(neighbor_softmax_lse).any() or torch.isnan(neighbor_softmax_lse).any():
                import pdb; pdb.set_trace()
            neighbor_softmax_lse[:, -true_group_seq_max_length:, :, :] = 1 / (1 + torch.exp(lse_gap))
            neighbor_softmax_lse[:, :-true_group_seq_max_length, :, :] = 1.
            group_softmax_lse = 1 / (1 + torch.exp(-lse_gap))



            neighbor_attn_output[:, -true_neighbor_seq_max_length:, ...] = (
                neighbor_attn_output[:, -true_neighbor_seq_max_length:, ...] * neighbor_softmax_lse
            )
            group_attn_output[:, -true_group_seq_max_length:, ...] = (
                group_attn_output[:, -true_group_seq_max_length:, ...] * group_softmax_lse
            )
            attn_output = torch.empty_like(neighbor_attn_output).copy_(
                neighbor_attn_output
            )  # might be slightly faster than clone
            #attn_output[:, group_size_2:, ...] += group_attn_output
            attn_output[:, group_size_2-kv_seq_len:, ...] += group_attn_output
            attn_output = torch.nan_to_num(attn_output, nan=0)  
            
            '''
            lse_gap = group_softmax_lse - neighbor_softmax_lse[:, -true_group_seq_max_length:, :, :]
            if  torch.isinf(neighbor_softmax_lse).any() or torch.isnan(neighbor_softmax_lse).any():
                import pdb; pdb.set_trace()
            neighbor_softmax_lse = torch.exp(neighbor_softmax_lse)
            #neighbor_softmax_lse[:, -true_group_seq_max_length:, :, :] = torch.exp(torch.tensor(1))  / (torch.exp(torch.tensor(1)) + torch.exp(lse_gap+1))
            #neighbor_softmax_lse[:, :-true_group_seq_max_length, :, :] = 1.
            #group_softmax_lse = torch.exp(torch.tensor(1))  / (torch.exp(torch.tensor(1))   + torch.exp(-lse_gap + 1))
            group_softmax_lse = torch.exp(group_softmax_lse)
            rescale_lse = torch.empty_like(neighbor_softmax_lse).copy_(neighbor_softmax_lse)
            rescale_lse[:, -true_group_seq_max_length:, :, :] += group_softmax_lse

            neighbor_attn_output[:, -true_neighbor_seq_max_length:, ...] = (
                neighbor_attn_output[:, -true_neighbor_seq_max_length:, ...] * neighbor_softmax_lse
            )
            group_attn_output[:, -true_group_seq_max_length:, ...] = (
                group_attn_output[:, -true_group_seq_max_length:, ...] * group_softmax_lse
            )
            attn_output = torch.empty_like(neighbor_attn_output).copy_(
                neighbor_attn_output
            )  # might be slightly faster than clone
            #attn_output[:, group_size_2:, ...] += group_attn_output
            attn_output[:, group_size_2-kv_seq_len:, ...] += group_attn_output
            attn_output[:, -true_neighbor_seq_max_length:, ...] /= rescale_lse
            attn_output = torch.nan_to_num(attn_output, nan=0)  
            '''
        else:
            attn_output = self._flash_attention_forward(
                neighbor_query_states,
                neighbor_key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=attn_dropout,
                window_size=[-1, -1],
            )
        """
        
        neighbor_query_states = neighbor_query_states.transpose(1, 2).contiguous()
        neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        group_query_states = group_query_states.transpose(1, 2).contiguous()
        group_key_states = repeat_kv(group_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()
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
        
        '''
        neighbor_query_states = neighbor_query_states.contiguous()
        neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups).contiguous()
        group_query_states = group_query_states.contiguous()
        group_key_states = repeat_kv(group_key_states, self.num_key_value_groups).contiguous()
        value_states = repeat_kv(value_states, self.num_key_value_groups).contiguous()

        attn_output = prefill_flash_forward(neighbor_query_states, 
                                            neighbor_key_states, 
                                            group_query_states, 
                                            group_key_states, 
                                            value_states, 
                                            q_len, 
                                            kv_seq_len, 
                                            group_size_2, 
                                            sm_scale=None)
        attn_output = attn_output.contiguous().transpose(1, 2)
        '''
    else:
        raise ValueError("q_len should be 1 or seq_len.")
    
    attn_output = attn_output.contiguous()
    attn_output = attn_output.view(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value

 