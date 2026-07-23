# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except (ImportError, OSError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except (ImportError, OSError):
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    # Several upstream Wan call sites invoke this low-level helper directly.
    # Fall back here (not only in ``attention``) when a FlashAttention wheel is
    # absent or ABI-incompatible with the host glibc.
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            fa_version=version,
        )

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lengths = [lq] * b
        q_lens = torch.tensor(
            q_lengths, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q_lens = torch.as_tensor(q_lens, dtype=torch.int32).reshape(-1)
        if q_lens.numel() != b:
            raise ValueError("q_lens must contain one length per batch item")
        q_lengths = q_lens.detach().cpu().tolist()
        if any(length < 0 or length > lq for length in q_lengths):
            raise ValueError(f"q_lens entries must be in [0, {lq}]")
        q = half(torch.cat([u[:length] for u, length in zip(q, q_lengths)]))
        q_lens = q_lens.to(device=q.device, non_blocking=True)

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lengths = [lk] * b
        k_lens = torch.tensor(
            k_lengths, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k_lens = torch.as_tensor(k_lens, dtype=torch.int32).reshape(-1)
        if k_lens.numel() != b:
            raise ValueError("k_lens must contain one length per batch item")
        k_lengths = k_lens.detach().cpu().tolist()
        if any(length < 0 or length > lk for length in k_lengths):
            raise ValueError(f"k_lens entries must be in [0, {lk}]")
        k = half(torch.cat([u[:length] for u, length in zip(k, k_lengths)]))
        v = half(torch.cat([u[:length] for u, length in zip(v, k_lengths)]))
        k_lens = k_lens.to(device=k.device, non_blocking=True)

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0]
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic)

    # FlashAttention returns packed queries. Restore the documented padded
    # [B, Lq, ...] layout instead of assuming every q_lens entry equals Lq.
    if all(length == lq for length in q_lengths):
        x = x.unflatten(0, (b, lq))
    else:
        padded = x.new_zeros((b, lq, *x.shape[1:]))
        offset = 0
        for batch_index, length in enumerate(q_lengths):
            padded[batch_index, :length] = x[offset:offset + length]
            offset += length
        x = padded

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        out_dtype = q.dtype
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        if q_scale is not None:
            q = q * q_scale
        if softmax_scale is not None:
            q = q * (softmax_scale * q.shape[-1] ** 0.5)

        batch_size, _, query_length, _ = q.shape
        key_length = k.shape[2]
        attn_mask = None
        needs_explicit_mask = (
            q_lens is not None
            or k_lens is not None
            or window_size != (-1, -1)
        )
        if needs_explicit_mask:
            device = q.device
            query_positions = torch.arange(query_length, device=device)
            key_positions = torch.arange(key_length, device=device)
            if q_lens is None:
                q_lens_tensor = torch.full(
                    (batch_size,), query_length, device=device, dtype=torch.long
                )
            else:
                q_lens_tensor = torch.as_tensor(
                    q_lens, device=device, dtype=torch.long
                ).reshape(-1)
            if k_lens is None:
                k_lens_tensor = torch.full(
                    (batch_size,), key_length, device=device, dtype=torch.long
                )
            else:
                k_lens_tensor = torch.as_tensor(
                    k_lens, device=device, dtype=torch.long
                ).reshape(-1)
            if q_lens_tensor.numel() != batch_size or k_lens_tensor.numel() != batch_size:
                raise ValueError(
                    "q_lens and k_lens must contain one length per batch item"
                )

            valid_queries = query_positions.unsqueeze(0) < q_lens_tensor.unsqueeze(1)
            valid_keys = key_positions.unsqueeze(0) < k_lens_tensor.unsqueeze(1)
            allowed = valid_queries.unsqueeze(2) & valid_keys.unsqueeze(1)

            # FlashAttention uses a bottom-right aligned causal mask when query
            # and key lengths differ. Equal-length self-attention is unchanged.
            relative_key = key_positions.view(1, 1, -1)
            relative_query = query_positions.view(1, -1, 1)
            # FlashAttention bottom-right alignment is defined per unpadded
            # sequence, not from the common padded tensor dimensions.
            causal_limit = relative_query + (
                k_lens_tensor - q_lens_tensor
            ).view(batch_size, 1, 1)
            if causal:
                allowed &= relative_key <= causal_limit
            left_window, right_window = window_size
            if left_window >= 0:
                allowed &= relative_key >= causal_limit - left_window
            if right_window >= 0:
                allowed &= relative_key <= causal_limit + right_window
            attn_mask = allowed.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=causal and attn_mask is None,
            dropout_p=dropout_p,
            enable_gqa=q.shape[1] != k.shape[1],
        )

        if attn_mask is not None:
            # PyTorch versions differ in how fully-masked query rows are
            # represented. Force padding queries to the documented zero output.
            valid_rows = attn_mask.any(dim=-1, keepdim=True)
            out = torch.where(valid_rows, out, torch.zeros_like(out))

        out = out.transpose(1, 2).contiguous()
        return out.to(out_dtype)
