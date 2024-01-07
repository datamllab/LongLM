

def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        group_size_1: int = 4,
        group_size_2: int = 512,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    # [batch_size, seq_length, 3 x hidden_size]
    fused_qkv = self.query_key_value(hidden_states)

    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_states, key_states, value_states) = self._split_heads(fused_qkv)

    if self.qk_layernorm:
        query_states = self.q_layernorm(query_states)
        key_states = self.k_layernorm(key_states)

    # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, num_heads, head_dim]
    query_states = query_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    # Partial rotary embedding
    query_rot, query_pass = (
        query_states[..., : self.rotary_emb.dim],
        query_states[..., self.rotary_emb.dim :],
    )
    key_rot, key_pass = (
        key_states[..., : self.rotary_emb.dim],
        key_states[..., self.rotary_emb.dim :],
    )
    # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
    query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

    # [batch_size, seq_length, num_heads, head_dim]
    query_states = torch.cat((query_rot, query_pass), dim=-1)
    key_states = torch.cat((key_rot, key_pass), dim=-1)

    if past_key_value is not None:
        # Specific to RoPE models with partial rotation
        cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_emb.dim}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query_states.dtype)
    attn_weights = self.attention_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.dense(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value        