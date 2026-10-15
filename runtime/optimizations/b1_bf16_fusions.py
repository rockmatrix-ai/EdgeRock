import sys
from types import MethodType
import torch
import triton
import triton.language as tl
from model import inference_kernels as kernels
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_model
from . import b1_bf16_projection as projection_patch

original_rope = hf_model.apply_rotary_pos_emb
original_query = {}


@triton.jit
def fast_norm(X, U, G, Y, R, WIDTH: tl.constexpr, EPS: tl.constexpr,
              BLOCK: tl.constexpr, ADD: tl.constexpr, ROUND_ADD: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + column, column < WIDTH, 0).to(tl.float32)
    if ADD:
        x += tl.load(U + row * WIDTH + column, column < WIDTH, 0).to(tl.float32)
        if ROUND_ADD:
            x = x.to(X.dtype.element_ty).to(tl.float32)
        tl.store(R + row * WIDTH + column, x, column < WIDTH)
    inv = tl.rsqrt(tl.sum(x * x, 0) / WIDTH + EPS)
    gamma = tl.load(G + column, column < WIDTH, 0)
    tl.store(Y + row * WIDTH + column, x * inv * gamma, column < WIDTH)


def norm(values, gamma, eps, update=None, round_update=True):
    values = values.contiguous()
    width = values.shape[-1]
    rows = values.numel() // width
    if rows == 0:
        return values, values
    output = torch.empty_like(values)
    summed = torch.empty_like(values) if update is not None else values
    fast_norm[rows,](values, values if update is None else update, gamma, output, summed,
        width, float(eps), triton.next_power_of_2(width), update is not None, round_update,
        num_warps=4, enable_fp_fusion=False)
    return summed, output



@triton.jit
def rope_kernel(X, C, S, Y, TOTAL: tl.constexpr, D: tl.constexpr,
                 ROT: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, column = index // D, index % D
    x = tl.load(X + row * STRIDE + column, index < TOTAL, 0).to(tl.float32)
    paired = tl.where(column < ROT // 2, column + ROT // 2, column - ROT // 2)
    other = tl.load(X + row * STRIDE + paired, (index < TOTAL) & (column < ROT), 0).to(tl.float32)
    other = tl.where(column < ROT // 2, -other, other)
    cos = tl.load(C + column, column < ROT, 0).to(tl.float32)
    sin = tl.load(S + column, column < ROT, 0).to(tl.float32)
    first = (x * cos).to(X.dtype.element_ty).to(tl.float32)
    second = (other * sin).to(X.dtype.element_ty).to(tl.float32)
    rotated = tl.where(column < ROT, first + second, x)
    tl.store(Y + index, rotated, index < TOTAL)


def rotate(values, cos, sin):
    d = values.shape[-1]
    flat = values.reshape(-1, d)
    out = torch.empty(values.shape, device=values.device, dtype=values.dtype)
    rope_kernel[triton.cdiv(values.numel(), 256),](flat, cos, sin, out,
        values.numel(), d, cos.shape[-1], flat.stride(0), 256,
        num_warps=4, enable_fp_fusion=False)
    return out


def rope(q, k, cos, sin, unsqueeze_dim=1):
    if cos.numel() == cos.shape[-1] and cos.dtype == q.dtype and q.stride(-1) == 1 and k.stride(-1) == 1:
        return rotate(q, cos, sin), rotate(k, cos, sin)
    return original_rope(q, k, cos, sin, unsqueeze_dim)



@triton.jit
def norm_gate_query(X, Gamma, W, Bias, Gain, C, S, Q, Gate,
                   D: tl.constexpr, ROT: tl.constexpr, EPS: tl.constexpr):
    head = tl.program_id(0)
    column = tl.arange(0, D)
    raw = tl.load(X + head * D + column).to(tl.float32)
    gamma = tl.load(Gamma + column)
    normalized = (raw * tl.rsqrt(tl.sum(raw * raw, 0) / D + EPS) * gamma).to(X.dtype.element_ty)
    product = (normalized.to(tl.float32) * tl.load(W + head * D + column).to(tl.float32)).to(X.dtype.element_ty)
    score = tl.sum(product.to(tl.float32), 0).to(X.dtype.element_ty).to(tl.float32)
    score = (score / tl.sqrt(float(D))).to(X.dtype.element_ty).to(tl.float32)
    score = (score + tl.load(Bias + head).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
    sigmoid = tl.sigmoid(score).to(X.dtype.element_ty).to(tl.float32)
    gate = sigmoid * tl.load(Gain + head).to(tl.float32)
    tl.store(Gate + head, gate)
    paired = tl.where(column < ROT // 2, column + ROT // 2, column - ROT // 2)
    paired = tl.maximum(paired, 0) % D
    other = tl.gather(normalized, paired, 0).to(tl.float32)
    other = tl.where(column < ROT // 2, -other, other)
    cos = tl.load(C + column, column < ROT, 0).to(tl.float32)
    sin = tl.load(S + column, column < ROT, 0).to(tl.float32)
    first = (normalized.to(tl.float32) * cos).to(X.dtype.element_ty).to(tl.float32)
    second = (other * sin).to(X.dtype.element_ty).to(tl.float32)
    query = tl.where(column < ROT, first + second, normalized.to(tl.float32))
    tl.store(Q + head * D + column, query)


@triton.jit
def admission_kernel(X, W, Y, RANK: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    x = tl.load(X + column, column < RANK, 0).to(tl.float32)
    activation = (x * tl.sigmoid(x)).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + group * RANK + column, column < RANK, 0).to(tl.float32)
    tl.store(Y + group, tl.sum(activation * weight, 0))


def fused_query(m, u, position_embeddings):
    if u.shape[:2] != (1, 1):
        return original_query[m](u, position_embeddings)
    projected = m.q_proj(u).contiguous()
    query = torch.empty_like(projected).unflatten(-1, (m.heads, m.dim))
    gate = torch.empty((1, 1, m.heads), device=u.device, dtype=u.dtype)
    cos, sin = position_embeddings
    gain = m.read_scale.tanh() * 2
    norm_gate_query[m.heads,](projected, m.q_norm._qwen35_native_rms_gamma,
        m.read_gate_weight, m.read_gate_bias, gain, cos, sin, query, gate,
        m.dim, cos.shape[-1], float(m.q_norm.eps), num_warps=4, enable_fp_fusion=False)
    features = m.admission_down(u.detach()).contiguous()
    scores = torch.empty((1, 1, m.groups), device=u.device, dtype=torch.float32)
    admission_kernel[m.groups,](features, m.admission_up.weight, scores,
        features.shape[-1], m.groups, triton.next_power_of_2(features.shape[-1]),
        num_warps=4, enable_fp_fusion=False)
    return query, scores, gate



def install(model):
    projection_patch.install()
    kernels.triton_qwen35_rmsnorm_torch_tree = norm
    hf_model.apply_rotary_pos_emb = rope
    modules = [m for m in model.modules() if hasattr(m, '_query_scores') and hasattr(m, '_write_kv')]
    source_modules = {sys.modules[type(m).__module__] for m in modules}
    for module in source_modules:
        previous_memory_rope = module._memory_rope
        def memory_rope(values, cos, sin, original=previous_memory_rope):
            if cos.numel() == cos.shape[-1] and cos.dtype == values.dtype and values.stride(-1) == 1:
                return rotate(values, cos, sin)
            return original(values, cos, sin)
        module._memory_rope = memory_rope
        if hasattr(module, 'apply_rotary_pos_emb'):
            module.apply_rotary_pos_emb = rope
    for m in modules:
        if m not in original_query:
            original_query[m] = m._query_scores
        m._query_scores = MethodType(fused_query, m)
