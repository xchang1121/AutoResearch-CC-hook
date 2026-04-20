"""Seed kernel for the hyper-connection backward.

Four output gradients to produce (all read from `grad_updated` plus one
other input):
    grad_h_res        = grad_updated @ streams.T
    grad_streams_part = h_res.T @ grad_updated
    grad_h_post       = sum_h(grad_updated * sublayer_out)      ← this kernel
    grad_sublayer_out = sum_r(grad_updated * h_post)

This seed implements only `grad_h_post` in Triton-Ascend (a simple per-row
dot-product with a broadcast sublayer row). The two matmuls and the other
reduction stay in PyTorch via tensor methods (`.matmul()`, `.sum()`) — they
are deliberate targets for autoresearch:

  1. Replace `.matmul()` calls with real Triton batched-matmul kernels.
  2. Fuse the two rate-dim reductions (`grad_h_post`, `grad_sublayer_out`)
     into a single pass over `grad_updated`, amortizing the largest tensor's
     HBM traffic.
  3. Fuse everything that reads `grad_updated` into one kernel — the most
     aggressive endpoint, where the [seq, batch, 4*hidden] tensor streams
     through global memory exactly once.

NOTE: The worker's verify pipeline imports `ModelNew` by name — do not rename.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

try:
    import torch_npu
except ImportError:
    torch_npu = None


@triton.jit
def grad_h_post_kernel(
    gu_ptr,         # float32 [N_rows, H]     = grad_updated_view flattened to (seq*batch*rate, hidden)
    sl_ptr,         # float32 [N_pairs, H]    = sublayer_out flattened to (seq*batch, hidden)
    out_ptr,        # float32 [N_rows]        = grad_h_post (will be reshaped to [seq, batch, rate, 1])
    N_rows, H,
    rate: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """Per-row elementwise-mul + reduce-over-hidden.

    Each output slot reduces H elements; we parallelize across N_rows via
    stride-NUM_CORES looping (the canonical Ascend pattern). `pair = row //
    rate` maps a (seq, batch, r) row back to its (seq, batch) sublayer row.
    """
    pid = tl.program_id(0)
    for row in range(pid, N_rows, NUM_CORES):
        pair = row // rate
        row_base = row * H
        pair_base = pair * H
        acc = 0.0
        for h_start in range(0, H, BLOCK_H):
            offs = h_start + tl.arange(0, BLOCK_H)
            mask = offs < H
            gu = tl.load(gu_ptr + row_base + offs, mask=mask, other=0.0)
            sl = tl.load(sl_ptr + pair_base + offs, mask=mask, other=0.0)
            acc += tl.sum(gu * sl, axis=0)
        tl.store(out_ptr + row, acc)


class ModelNew(nn.Module):
    def __init__(self, rate=4, hidden_size=3584, input_dtype=torch.bfloat16):
        super().__init__()
        self.rate = rate
        self.hidden_size = hidden_size
        self.input_dtype = input_dtype

    def forward(self, h_res, h_post, original_streams, sublayer_out, grad_updated):
        seq_len, batch_size, packed = original_streams.shape
        R = self.rate
        H = self.hidden_size
        assert packed == R * H

        # Cast to fp32 up-front for numerical parity with the reference.
        x_streams = original_streams.float().view(seq_len, batch_size, R, H)
        grad_updated_view = grad_updated.float().view(seq_len, batch_size, R, H)
        h_res_f = h_res.float()
        h_post_f = h_post.float()
        sublayer_f = sublayer_out.float()

        # --- grad_h_res: [seq, batch, R, R] = grad_updated_view @ x_streams.T
        grad_h_res = grad_updated_view.matmul(x_streams.transpose(-1, -2))

        # --- grad_streams_part: [seq, batch, R, H] = h_res.T @ grad_updated_view
        grad_x_from_res = h_res_f.transpose(-1, -2).matmul(grad_updated_view)

        # --- grad_h_post: triton reduction over hidden dim
        N_rows = seq_len * batch_size * R
        gu_flat = grad_updated_view.contiguous().view(N_rows, H)
        sl_flat = sublayer_f.contiguous().view(seq_len * batch_size, H)
        grad_h_post_buf = torch.empty(N_rows, dtype=torch.float32,
                                      device=grad_updated.device)

        NUM_CORES = 20
        BLOCK_H = 256
        grad_h_post_kernel[(NUM_CORES,)](
            gu_flat, sl_flat, grad_h_post_buf,
            N_rows, H,
            rate=R, BLOCK_H=BLOCK_H, NUM_CORES=NUM_CORES,
        )
        grad_h_post = grad_h_post_buf.view(seq_len, batch_size, R, 1)

        # --- grad_sublayer_out: [seq, batch, H] = sum over rate of gu * h_post
        grad_sublayer_out = (grad_updated_view * h_post_f).sum(dim=2)

        return (
            grad_h_res.to(h_res.dtype),
            grad_h_post.to(h_post.dtype),
            grad_x_from_res.reshape_as(original_streams).to(original_streams.dtype),
            grad_sublayer_out.to(sublayer_out.dtype),
        )
