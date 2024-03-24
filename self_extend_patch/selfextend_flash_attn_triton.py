import math
import torch

import triton
import triton.language as tl


def self_extend_flash_forward_triton(
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

    o = _self_extend_flash_forward_triton(q=neighbor_query_states,
                        k=neighbor_key_states,
                        q1=group_query_states,
                        k1=group_key_states,
                        v=value_states,
                        causal=(q_len == kv_seq_len),
                        sm_scale=1. / math.sqrt(neighbor_query_states.shape[-1]),
                        window=group_size_2)
    o = o.transpose(1, 2).contiguous()
    # print("o", o.shape)
    return o






def _self_extend_flash_forward_triton(q, k, q1, k1, v, causal, sm_scale, window):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        
        device = torch.cuda.device_of(q)
        with torch.cuda.device(device):
            o = torch.empty_like(q)
            BLOCK_M = 128
            BLOCK_N = 32
            grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])
            L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
            _fwd_kernel[grid](
                q, 
                k,
                q1,
                k1,
                v, 
                sm_scale,
                L,
                o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                q.shape[0], 
                q.shape[1], 
                q.shape[2],
                k.shape[2],
                BLOCK_M=BLOCK_M, 
                BLOCK_N=BLOCK_N, 
                BLOCK_DMODEL=Lk,
                IS_CAUSAL=causal,
                WINDOW=window,
                num_warps=8,
                num_stages=2)

        return o




@triton.heuristics(
    {
        "EVEN_M": lambda args: args["Q_CTX"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["KV_CTX"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_kernel(
    Q, 
    K,
    Q1,
    K1,
    V, 
    sm_scale,
    L,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, 
    H, 
    Q_CTX,
    KV_CTX,
    BLOCK_M: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr

):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # qvk_offset = off_hz * stride_qh
    q_offset = off_hz * stride_qh
    vk_offset = off_hz * stride_kh
    # vk_offset = q_offset

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(Q_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + vk_offset,
        shape=(KV_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    Q1_block_ptr = tl.make_block_ptr(
        base=Q1 + q_offset,
        shape=(Q_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K1_block_ptr = tl.make_block_ptr(
        base=K1 + vk_offset,
        shape=(KV_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + vk_offset,
        shape=(KV_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.4426950408889634
    
    # load q: it will stay in SRAM throughout
    if EVEN_M:
        q = tl.load(Q_block_ptr)
        q1 = tl.load(Q1_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(1,0))
        q1 = tl.load(Q1_block_ptr, boundary_check=(1,0))

    q = (q * qk_scale).to(tl.bfloat16)
    q1 = (q1 * qk_scale).to(tl.bfloat16)


    # Dot I trick: it converts q1, q2 into mma layout and saves shared memory
    # better way to generate a eye matrix. avoid casting from bool
    offs_k = tl.arange(0, BLOCK_DMODEL)
    I = tl.where(offs_k[:, None] == offs_k,
                 tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=tl.bfloat16),
                 tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=tl.bfloat16))
    q = tl.dot(q, I).to(tl.bfloat16)
    q1 = tl.dot(q1, I).to(tl.bfloat16)


    # loop over k, v and update accumulator
    lo = 0
    if IS_CAUSAL:
        hi = tl.minimum(KV_CTX, (start_m + 1) * BLOCK_M)
    else:
        hi = KV_CTX

    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        if EVEN_N:
            k = tl.load(K_block_ptr)
            k1 = tl.load(K1_block_ptr)
            v = tl.load(V_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(1,0))
            k1 = tl.load(K1_block_ptr, boundary_check=(1,0))
            v = tl.load(V_block_ptr, boundary_check=(1,0))
        
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Window masking
        mask = ( KV_CTX - Q_CTX + offs_m[:, None]) >= (start_n + offs_n[None, :] + WINDOW)
        qk += tl.where(mask, tl.dot(q1, tl.trans(k1)), tl.dot(q, tl.trans(k)))

        # if not EVEN_N:
        #     mask = (start_n + offs_n) < KV_CTX
        #     qk = tl.where(mask, qk, float("-inf"))

        if IS_CAUSAL:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))
        # qk += tl.dot(q, k)
        
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.bfloat16), v)
        
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        K1_block_ptr = tl.advance(K1_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    

    # write back l and m        
    acc = acc * (1.0 / l_i[:, None])
    l_ptrs = L + off_hz * Q_CTX + offs_m
    
    mask_m = offs_m < Q_CTX
    l_i = m_i  + tl.math.log2(l_i)
    if EVEN_M:
        tl.store(l_ptrs, l_i)
    else:
        tl.store(l_ptrs, l_i, mask=mask_m)

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(Q_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    if EVEN_M:
        tl.store(O_block_ptr, acc.to(tl.bfloat16))
    else:
        tl.store(O_block_ptr, acc.to(tl.bfloat16), boundary_check=(1,0))
