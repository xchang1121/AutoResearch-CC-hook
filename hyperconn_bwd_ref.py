"""PyTorch reference for the hyper-connection output cell backward.

Computes four gradients from the forward

    updated = stack([h_res_r @ streams_r + h_post_r * sublayer_out
                     for r in range(rate)], dim=2)

given `grad_updated`, via manual differentiation:

    grad_h_res[s, b, r, i] = sum_h grad_updated[s, b, r, h] * streams[s, b, i, h]
    grad_streams[s, b, i, h] = sum_r h_res[s, b, r, i] * grad_updated[s, b, r, h]
    grad_h_post[s, b, r, 0] = sum_h grad_updated[s, b, r, h] * sublayer_out[s, b, h]
    grad_sublayer_out[s, b, h] = sum_r grad_updated[s, b, r, h] * h_post[s, b, r, 0]

Shape table (rate=4, hidden=3584, seq=2048, batch=2):
    h_res:           [seq, batch, rate, rate]
    h_post:          [seq, batch, rate, 1]
    original_streams:[seq, batch, rate*hidden]  (viewed as [..., rate, hidden])
    sublayer_out:    [seq, batch, hidden]
    grad_updated:    [seq, batch, rate*hidden]  (viewed as [..., rate, hidden])
Outputs match their respective input shapes and dtypes.
"""
import torch
import torch.nn as nn


def _sinkhorn_knopp(logits, iters, eps):
    logits = logits.float()
    logits_max = logits.amax(dim=-1, keepdim=True)
    matrix = torch.exp(logits - logits_max)
    for _ in range(iters):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


class Model(nn.Module):
    def __init__(self, rate=4, hidden_size=3584, input_dtype=torch.bfloat16):
        super().__init__()
        self.rate = rate
        self.hidden_size = hidden_size
        self.input_dtype = input_dtype

    def forward(self, h_res, h_post, original_streams, sublayer_out, grad_updated):
        seq_len, batch_size, packed_hidden_size = original_streams.shape
        assert packed_hidden_size == self.rate * self.hidden_size

        x_streams = original_streams.float().view(
            seq_len, batch_size, self.rate, self.hidden_size
        )
        sublayer_view = sublayer_out.float().view(
            seq_len, batch_size, 1, self.hidden_size
        )
        grad_updated = grad_updated.float().view(
            seq_len, batch_size, self.rate, self.hidden_size
        )

        grad_h_res = torch.matmul(grad_updated, x_streams.transpose(-1, -2))
        grad_x_from_res = torch.matmul(h_res.float().transpose(-1, -2), grad_updated)
        grad_h_post = (grad_updated * sublayer_view).sum(dim=-1, keepdim=True)
        grad_sublayer_out = (grad_updated * h_post.float()).sum(dim=2)

        return (
            grad_h_res.to(h_res.dtype),
            grad_h_post.to(h_post.dtype),
            grad_x_from_res.reshape_as(original_streams).to(original_streams.dtype),
            grad_sublayer_out.to(sublayer_out.dtype),
        )


def get_inputs():
    torch.manual_seed(2033)

    h_res_logits = torch.randn(2048, 2, 4, 4, dtype=torch.float32)
    h_res = _sinkhorn_knopp(h_res_logits, 20, 1e-6).to(torch.bfloat16)

    h_post = (2.0 * torch.sigmoid(
        torch.randn(2048, 2, 4, 1, dtype=torch.float32)
    )).to(torch.bfloat16)

    original_streams = torch.randn(2048, 2, 14336, dtype=torch.float32).to(torch.bfloat16)
    sublayer_out = torch.randn(2048, 2, 3584, dtype=torch.float32).to(torch.bfloat16)
    grad_updated = torch.randn(2048, 2, 14336, dtype=torch.float32).to(torch.bfloat16)

    return [h_res, h_post, original_streams, sublayer_out, grad_updated]


def get_init_inputs():
    return [4, 3584, torch.bfloat16]
