from flash_attn import flash_attn_func, flash_attn_varlen_func
import torch

# must replace orginal flash forward method with the following one first, to enbale the window feature.
def flash_attention2_forward_with_window_size(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0.0,
    softmax_scale=None,
    window_size=[-1, -1],
    return_attn_probs=False,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`int`, *optional*):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        window_size ([Int, Int])
            The left & right window size for Flash Attention. Default to [-1, -1] which means no window size is used.
        return_attn_probs (`bool`, *optional*):
            Whether to return the attention softmax logssumexp and probabilities. Default to False.
    """
    if not self._flash_attn_uses_top_left_mask:
        causal = self.is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
        causal = self.is_causal and query_length != 1

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
        attn_output_unpad, softmax_lse, S_dmask = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_attn_probs=True,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        attn_output, softmax_lse, S_dmask = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_attn_probs=True,
        )

    if return_attn_probs:
        return attn_output, softmax_lse, S_dmask
    else:
        return attn_output

def self_extend_flash_forward(
        model_self,
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
    ):
    
    if query_position.max() >= group_size_2:
        neighbor_attn_output, neighbor_softmax_lse_right_padded, neighbor_prob = model_self._flash_attention_forward(
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
        group_attn_output, group_softmax_lse_right_padded, group_prob = model_self._flash_attention_forward(
            group_query_states[:, -group_attention_len:, :, :],
            group_key_states[:, :group_attention_len, :, :],
            value_states[:, :group_attention_len, :, :],
            group_attention_mask,
            group_query_states[:, -group_attention_len:, :, :].shape[1],
            dropout=attn_dropout,
            window_size=[-1, -1],
            return_attn_probs=True,
        )  # note that kv and q's indexing are different! also query size could be different from kv length and very small during generation compared to prefilling


        # normalize lse first
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

        lse_gap = group_softmax_lse - neighbor_softmax_lse[:, -true_group_seq_max_length:, :, :]
        #if  torch.isinf(neighbor_softmax_lse).any() or torch.isnan(neighbor_softmax_lse).any():
        #    import pdb; pdb.set_trace()
        
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
    
    else:
        attn_output = model_self._flash_attention_forward(
            neighbor_query_states,
            neighbor_key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
            window_size=[-1, -1],
        )

    return attn_output
