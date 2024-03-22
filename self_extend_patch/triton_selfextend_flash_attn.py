import torch
import math
import triton
import triton.language as tl




# We don't run auto-tuning every time to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
#@triton.autotune(
#    configs=[
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
#        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
#        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
#        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=7, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=7, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=6, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=5, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=6, num_warps=4),
#    ],
#    key=['N_CTX'],
#)
@triton.jit
def _attn_fwd_prefill(Q1, K1, Q2, K2, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H,  #
              Q_CTX: tl.constexpr,  #
              N_CTX: tl.constexpr,  #
              WINDOW: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q1_block_ptr = tl.make_block_ptr(
        base=Q1 + qvk_offset,
        shape=(Q_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    Q2_block_ptr = tl.make_block_ptr(
        base=Q2 + qvk_offset,
        shape=(Q_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K1_block_ptr = tl.make_block_ptr(
        base=K1 + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    K2_block_ptr = tl.make_block_ptr(
        base=K2 + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(Q_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.442695040888963#1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    #q = tl.load(Q_block_ptr)
    if start_m * BLOCK_M + BLOCK_M > Q_CTX:
        q1 = tl.load(Q1_block_ptr, boundary_check=(0,), padding_option='zero')
        q2 = tl.load(Q2_block_ptr, boundary_check=(0,), padding_option='zero')
    else:
        q1 = tl.load(Q1_block_ptr)
        q2 = tl.load(Q2_block_ptr)
    #q1 = (q1 * qk_scale).to(tl.float16)
    #q2 = (q2 * qk_scale).to(tl.float16)
                
    lo = 0
    hi = (start_m + 1) * BLOCK_M 
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        #qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) #?
        #qk = qk.to(tl.float16)
        # if use condition, qk has to be float32, then convert to float16...
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if start_n + BLOCK_N - 1 > start_m * BLOCK_M - 1:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, -1.0e6)#float("-inf"))
        
        #qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # -- compute qk ----
        #k = tl.load(K_block_ptr)
        # case 1: only need group attention: q2, k2
        if BLOCK_N + start_n <= (start_m * BLOCK_M - WINDOW + 1):
            if BLOCK_N + start_n >= N_CTX:
                k2 = tl.load(K2_block_ptr, boundary_check=(1,), padding_option='zero')
                v = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero')
            else:
                k2 = tl.load(K2_block_ptr)
                v = tl.load(V_block_ptr)
            #qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            #qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
            qk += tl.dot(q2, k2)#, out_dtype=tl.float16)
        else:
            #case 2: only need neighbor attention: q1, k1
            if start_n >= (start_m+1) * BLOCK_M - WINDOW:
                if BLOCK_N + start_n >= N_CTX:
                    k1 = tl.load(K1_block_ptr, boundary_check=(1,), padding_option='zero')
                    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero')
                else:
                    k1 = tl.load(K1_block_ptr)
                    v = tl.load(V_block_ptr)
                #qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                #qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
                qk += tl.dot(q1, k1)#, out_dtype=tl.float16)
            else:
                #case 3: need both q1, k1 and q2, k2
                if BLOCK_N + start_n >= N_CTX:
                    k1 = tl.load(K1_block_ptr, boundary_check=(1,), padding_option='zero')
                    k2 = tl.load(K2_block_ptr, boundary_check=(1,), padding_option='zero')
                    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero')
                else:
                    k1 = tl.load(K1_block_ptr)
                    k2 = tl.load(K2_block_ptr)
                    v = tl.load(V_block_ptr)
                #qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                #qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
                qk1 = tl.dot(q1, k1)#, out_dtype=tl.float16)
                qk2 = tl.dot(q2, k2)#, out_dtype=tl.float16)
                #merge_mask = tl.abs((offs_m[:, None] - (start_n + offs_n[None, :]))) >= WINDOW
                #qk += tl.where(merge_mask, qk2, qk1)
                qk += tl.where(tl.abs(offs_m[:, None] - (start_n + offs_n[None, :])) < WINDOW, qk1, qk2)

        qk *= qk_scale

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        #v = tl.load(V_block_ptr)
        #v = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero')
        acc += tl.dot(p.to(tl.float16), v)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K1_block_ptr = tl.advance(K1_block_ptr, (0, BLOCK_N))
        K2_block_ptr = tl.advance(K2_block_ptr, (0, BLOCK_N))

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * Q_CTX + offs_m
    if start_m * BLOCK_M + BLOCK_M >= Q_CTX:
        tl.store(m_ptrs, m_i, mask=offs_m < Q_CTX)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,))
    else:
        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def prefill_flash_forward(q1, k1, q2, k2, v, q_len, seq_len, window, sm_scale=None):
    # shape constraints
    Lq, Lk, Lv = q1.shape[-1], k1.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    assert q_len == seq_len or q_len == 1 
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(Lq) # the default scale factor.
    o = torch.empty_like(q1, device=q1.device)
    block_m = 128
    block_n = 64 # if Lk <= 64 else 32
    num_stages = 4 if Lk <= 64 else 3
    num_warps = 4
    # Tuning for H100
    if torch.cuda.get_device_capability()[0] == 9:
        num_warps = 8
        num_stages = 7 if Lk >= 64 else 3
    grid = (triton.cdiv(q1.shape[2], block_m), q1.shape[0] * q1.shape[1], 1)
    M = torch.empty((q1.shape[0], q1.shape[1], q1.shape[2]), device=q1.device, dtype=torch.float32)
    with torch.cuda.device(v.device.index):
        # https://github.com/Dao-AILab/flash-attention/commit/9795159082f6e6c847db2bf4284fd17326c31fbd
        # to avoid the device issue .
        _attn_fwd_prefill[grid](
            q1, k1, q2, k2, v, sm_scale, M, o,  #
            q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),  #
            k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q1.shape[0], q1.shape[1],  #
            Q_CTX=q_len, 
            N_CTX=seq_len,  #
            BLOCK_M=block_m,  #
            BLOCK_N=block_n,  #
            WINDOW=window,
            BLOCK_DMODEL=Lk,  #
            num_warps=num_warps,  #
            num_stages=num_stages  #
        )

    return o
