from types import MethodType
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

base_linear = F.linear
checks = []
checked = set()
writer_owners = {}


@triton.jit
def joint_matvec(X, Q, W, A, Y, NQ: tl.constexpr, NW: tl.constexpr,
                 NA: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    start = tl.program_id(0) * 2
    rows = start + tl.arange(0, 2)
    columns = tl.arange(0, BK)
    x = tl.load(X + columns, columns < K, 0).to(tl.float32)
    if start < NQ:
        weights = Q + rows[:, None] * K + columns[None, :]
    elif start < NQ + NW:
        weights = W + (rows[:, None] - NQ) * K + columns[None, :]
    else:
        weights = A + (rows[:, None] - NQ - NW) * K + columns[None, :]
    weight = tl.load(weights,
        (rows[:, None] < NQ + NW + NA) & (columns[None, :] < K), 0).to(tl.float32)
    result = tl.sum(weight * x[None, :], axis=1)
    tl.store(Y + rows, result, rows < NQ + NW + NA)


def writer_linear(values, weight, bias=None):
    owner = writer_owners.get(weight.data_ptr())
    if owner is not None and owner._b1_joint_active and values.numel() == values.shape[-1]:
        begin = owner.q_proj.out_features
        result = owner._b1_joint_output[..., begin:begin + weight.shape[0]].reshape(*values.shape[:-1], weight.shape[0])
        owner._b1_joint_active = False
        owner._b1_joint_output = None
        return result
    return base_linear(values, weight, bias)


def install(model):
    for m in model.modules():
        if not (hasattr(m, '_query_scores') and hasattr(m, '_inference_writer_gate_up')):
            continue
        m._b1_joint_output = None
        m._b1_joint_active = False
        writer_owners[m._inference_writer_gate_up.data_ptr()] = m
        previous_query, previous_admission = m.q_proj.forward, m.admission_down.forward
        assert all(w.is_contiguous() and w.shape[0] % 2 == 0 for w in
            (m.q_proj.weight, m._inference_writer_gate_up, m.admission_down.weight))

        def query(projection, values, *args, owner=m, previous=previous_query, **kwargs):
            owner._b1_joint_active = values.numel() == values.shape[-1]
            if not owner._b1_joint_active:
                return previous(values, *args, **kwargs)
            q, w, a = owner.q_proj.weight, owner._inference_writer_gate_up, owner.admission_down.weight
            n = q.shape[0] + w.shape[0] + a.shape[0]
            output = values.new_empty((*values.shape[:-1], n))
            joint_matvec[triton.cdiv(n, 2),](values, q, w, a, output,
                q.shape[0], w.shape[0], a.shape[0], q.shape[1],
                triton.next_power_of_2(q.shape[1]), num_warps=4, num_stages=1)
            if owner not in checked:
                expected = torch.cat([base_linear(values, weight) for weight in (q, w, a)], -1)
                checks.append(dict(layer=len(checks), shape=list(output.shape),
                    exact=bool(torch.equal(expected, output)), finite=bool(output.isfinite().all()),
                    max_abs=float((expected.float() - output.float()).abs().max())))
                checked.add(owner)
            owner._b1_joint_output = output
            return output[..., :q.shape[0]]

        def admission(projection, values, *args, owner=m, previous=previous_admission, **kwargs):
            if not owner._b1_joint_active:
                return previous(values, *args, **kwargs)
            return owner._b1_joint_output[..., -owner.admission_down.out_features:]

        m.q_proj.forward = MethodType(query, m.q_proj)
        m.admission_down.forward = MethodType(admission, m.admission_down)
    assert len(writer_owners) == 8
    F.linear = writer_linear
