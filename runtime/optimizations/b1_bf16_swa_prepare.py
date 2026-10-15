import inspect
import textwrap
from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit
def normalize_rotate_gate(QP, KP, QG, KG, C, S, Q, K, Gate,
                          NQ: tl.constexpr, D: tl.constexpr,
                          ROT: tl.constexpr, EPS: tl.constexpr):
    head = tl.program_id(0)
    column = tl.arange(0, D)
    if head < NQ:
        raw = tl.load(QP + head * 2 * D + column).to(tl.float32)
        gamma = tl.load(QG + column)
        gate = tl.load(QP + head * 2 * D + D + column)
        tl.store(Gate + head * D + column, gate)
    else:
        raw = tl.load(KP + (head - NQ) * D + column).to(tl.float32)
        gamma = tl.load(KG + column)
    normalized = (raw * tl.rsqrt(tl.sum(raw * raw, 0) / D + EPS)
                  * gamma).to(tl.bfloat16)
    paired = tl.where(column < ROT // 2, column + ROT // 2, column - ROT // 2)
    paired = tl.maximum(paired, 0) % D
    other = tl.gather(normalized, paired, 0).to(tl.float32)
    other = tl.where(column < ROT // 2, -other, other)
    cos = tl.load(C + column, column < ROT, 0).to(tl.float32)
    sin = tl.load(S + column, column < ROT, 0).to(tl.float32)
    first = (normalized.to(tl.float32) * cos).to(tl.bfloat16).to(tl.float32)
    second = (other * sin).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(column < ROT, first + second, normalized.to(tl.float32))
    if head < NQ:
        tl.store(Q + head * D + column, rotated)
    else:
        tl.store(K + (head - NQ) * D + column, rotated)


def prepare(module, q_projected, k_projected, position_embeddings):
    dim = module.head_dim
    nq, nk = q_projected.shape[-1] // (2 * dim), k_projected.shape[-1] // dim
    query = torch.empty((1, 1, nq, dim), device=q_projected.device, dtype=q_projected.dtype)
    key = torch.empty((1, 1, nk, dim), device=q_projected.device, dtype=q_projected.dtype)
    gate = torch.empty((1, 1, nq * dim), device=q_projected.device, dtype=q_projected.dtype)
    cos, sin = position_embeddings
    assert module.q_norm.eps == module.k_norm.eps
    normalize_rotate_gate[nq + nk,](q_projected, k_projected,
        module.q_norm._qwen35_native_rms_gamma, module.k_norm._qwen35_native_rms_gamma,
        cos, sin, query, key, gate, nq, dim, cos.shape[-1], float(module.q_norm.eps),
        num_warps=4, enable_fp_fusion=False)
    return query, key, gate


def install(model):
    modules = [m for m in model.modules()
        if getattr(m, '_qwen35_packed_swa_projection_enabled', False)]
    assert len(modules) == 8
    original = type(modules[0]).forward
    source = textwrap.dedent(inspect.getsource(original))
    start = source.index('    query, gate = torch.chunk(q_projected.view(')
    end = source.index('    window = int(', start)
    old = source[start:end]
    source = source[:start] + '''    if hidden_states.shape[:2] == (1, 1):
        query, key, gate = _prepare(self, q_projected, k_projected, position_embeddings)
        value = v_projected.view(hidden_shape)
    else:
''' + textwrap.indent(old, '    ') + source[end:]
    namespace = dict(original.__globals__, _prepare=lambda *args: prepare(*args))
    exec(compile(source, '<bf16-swa-normalize-rotate>', 'exec'), namespace)
    for module in modules:
        module.forward = MethodType(namespace['forward'], module)
