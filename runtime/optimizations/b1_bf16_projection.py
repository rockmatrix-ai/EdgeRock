import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from model import inference_kernels as kernels

original_linear = F.linear


@triton.jit
def bf16_matvec(X, W, Y, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * 2 + tl.arange(0, 2)
    columns = tl.arange(0, BK)
    x = tl.load(X + columns, columns < K, 0).to(tl.float32)
    weight = tl.load(W + rows[:, None] * K + columns[None, :],
        (rows[:, None] < N) & (columns[None, :] < K), 0).to(tl.float32)
    result = tl.sum(weight * x[None, :], axis=1)
    tl.store(Y + rows, result, rows < N)


def vector_linear(values, weight, bias=None):
    eligible = (values.is_cuda and values.dtype == torch.bfloat16
        and values.numel() == values.shape[-1] and values.stride(-1) == 1
        and weight.dtype == torch.bfloat16 and weight.is_contiguous() and bias is None)
    if not eligible:
        return original_linear(values, weight, bias)
    n, k = weight.shape
    output = torch.empty((*values.shape[:-1], n), device=values.device, dtype=values.dtype)
    bf16_matvec[triton.cdiv(n, 2),](values, weight, output, n, k,
        triton.next_power_of_2(k), num_warps=4, num_stages=1)
    return output


def install():
    F.linear = vector_linear
