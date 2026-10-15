import inspect
from types import MethodType

import torch
import triton
import triton.language as tl
from model import inference_kernels as kernels


@triton.jit
def head_scores(H, W, Y, SEEN, N: tl.constexpr, K: tl.constexpr,
                VOCAB: tl.constexpr, PENALTY: tl.constexpr):
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    accumulator = tl.zeros((8,), tl.float32)
    for start in range(0, K, 256):
        columns = start + tl.arange(0, 256)
        hidden = tl.load(H + columns, columns < K, 0.0)
        weight = tl.load(W + rows[:, None] * K + columns[None, :],
                         (rows[:, None] < N) & (columns[None, :] < K), 0.0)
        accumulator += tl.sum(weight.to(tl.float32) * hidden[None, :], axis=1)
    # Retain the output head's BF16 rounding before the original FP32 penalty.
    scores = accumulator.to(tl.bfloat16).to(tl.float32)
    seen = tl.load(SEEN + rows, rows < VOCAB, 0).to(tl.int1)
    scores = scores - tl.where(seen, PENALTY, 0.0)
    tl.store(Y + rows, scores, rows < N)


def install(model):
    original = model.lm_head.forward
    model.lm_head.__dict__['_edge_score_context'] = None

    def forward(self, hidden):
        context = self.__dict__['_edge_score_context']
        if context is None:
            return original(hidden)
        seen, penalty = context
        assert hidden.numel() == self.in_features and hidden.dtype == torch.bfloat16
        output = torch.empty((*hidden.shape[:-1], self.out_features), device=hidden.device, dtype=torch.float32)
        head_scores[triton.cdiv(self.out_features, 8),](hidden, self.weight, output, seen,
            self.out_features, self.in_features, seen.shape[-1], penalty,
            num_warps=4, num_stages=1)
        return output

    model.lm_head.forward = MethodType(forward, model.lm_head)


@triton.jit
def sample_update(VALUES, INDICES, ATTEMPTS, STEPS, TOKEN, STORAGE, SEEN,
                  LENGTHS, EOS, FINISHED, SLOT, SINK, CAPS, POSITIONS, IDS, EVENTS,
                  SEED: tl.constexpr, TOP_P: tl.constexpr, STRIDE: tl.constexpr,
                  MAXIMUM: tl.constexpr, PAD: tl.constexpr):
    columns = tl.arange(0, 32)
    mask = columns < 20
    values = tl.load(VALUES + columns, mask, -float('inf')).to(tl.float32)
    maximum = tl.max(values, axis=0)
    weights = tl.exp(values - maximum)
    total = tl.sum(weights, axis=0)
    probabilities = weights / total
    prefix_before = tl.cumsum(probabilities, axis=0) - probabilities
    truncated = tl.where(mask & (prefix_before < TOP_P), probabilities, 0.0)
    retained = tl.sum(truncated, axis=0)
    cumulative = tl.cumsum(truncated, axis=0)
    attempt_id = tl.load(ATTEMPTS).to(tl.uint32)
    step = tl.load(STEPS).to(tl.uint32)
    state = (attempt_id * 747796405 + step * 289133645 + SEED).to(tl.uint32)
    shift = (state >> 28) + 4
    word = ((state >> shift ^ state) * 277803737).to(tl.uint32)
    word = (word >> 22 ^ word).to(tl.uint32)
    uniform = ((word.to(tl.float64) + 0.5) / 4294967296.0).to(tl.float32)
    threshold = uniform * retained
    rank = tl.minimum(tl.sum((mask & (cumulative < threshold)).to(tl.int32), axis=0), 19)
    selected = tl.sum(tl.where(columns == rank, tl.load(INDICES + columns, mask, 0), 0), axis=0)
    tl.store(TOKEN, selected)
    finished = tl.load(FINISHED).to(tl.int1)
    slot = tl.load(SLOT).to(tl.int64)
    active = ~finished & (slot >= 0)
    sink = tl.load(SINK).to(tl.int64)
    current_step = tl.load(STEPS).to(tl.int64)
    write_row = tl.where(active, slot, sink)
    write_column = tl.where(active, current_step, MAXIMUM)
    stored = tl.where(active, selected, PAD)
    tl.store(STORAGE + write_row * STRIDE + write_column, stored)
    tl.store(SEEN + stored, 1)
    next_step = current_step + active.to(tl.int64)
    old_length = tl.load(LENGTHS + write_row)
    tl.store(LENGTHS + write_row, tl.where(active, next_step, old_length))
    old_eos = tl.load(EOS + write_row).to(tl.int1)
    tl.store(EOS + write_row, old_eos)
    cap = tl.load(CAPS).to(tl.int64)
    next_finished = finished | active & (next_step >= cap)
    tl.store(FINISHED, next_finished)
    tl.store(STEPS, next_step)
    position = tl.load(POSITIONS).to(tl.int64)
    tl.store(POSITIONS, position + active.to(tl.int64))
    tl.store(IDS, tl.where(next_finished, PAD, stored))
    events = tl.load(EVENTS).to(tl.int64)
    tl.store(EVENTS, events + active.to(tl.int64))


def sample_and_update(scores, attempts, storage, dense_seen, generated_lengths,
                      finished_by_eos, finished, slot_rows, sink_rows, slot_caps,
                      row_steps, row_positions, decode_ids, decode_useful_events,
                      *, seed, top_p, maximum_new_tokens, pad_token_id):
    values, indices = torch.topk(scores, 20, dim=-1)
    token = torch.empty((1, 1), device=scores.device, dtype=torch.long)
    sample_update[1,](values, indices, attempts, row_steps, token, storage, dense_seen,
        generated_lengths, finished_by_eos, finished, slot_rows, sink_rows, slot_caps,
        row_positions, decode_ids, decode_useful_events,
        (int(seed) & 4294967295) * 277803737 & 4294967295, float(top_p),
        storage.shape[1], int(maximum_new_tokens), int(pad_token_id), num_warps=1, num_stages=1)
    return token


def generation_function():
    source = inspect.getsource(kernels.cuda_graph_generate_top20_dynamic_refill)
    marker = '\n            fixed_graph = FixedMemoryDecodeGraph(model, cache, decode_ids, position=row_positions, scores_dtype=scores_dtype, consume_capture_steps=True)'
    assert source.count(marker) == 1
    source = source.replace(marker, "\n            model.lm_head.__dict__['_edge_score_context'] = (dense_seen, presence_penalty)" + marker)
    start = source.index('            _apply_dense_presence_penalty_(graph_scores, dense_seen, presence_penalty)')
    end = source.index('            replay_count += 1', start)
    source = source[:start] + '''            next_token = _edge_sample_update(graph_scores, slot_attempt_ids, storage, dense_seen,
                generated_lengths, finished_by_eos, finished, slot_rows, sink_rows, slot_caps,
                row_steps, row_positions, decode_ids, decode_useful_events, seed=counter_seed,
                top_p=top_p, maximum_new_tokens=max_new_tokens, pad_token_id=pad_token_id)
''' + source[end:]
    namespace = dict(kernels.__dict__, _edge_sample_update=sample_and_update)
    exec(compile(source, '<bf16-output-fusion>', 'exec'), namespace)
    generate = namespace[kernels.cuda_graph_generate_top20_dynamic_refill.__name__]

    def run(model, input_ids, attention_mask, **kwargs):
        assert input_ids.shape[0] == kwargs['slot_batch_size'] == 1
        assert kwargs['temperature'] == 1.0 and kwargs['top_k'] == 20
        assert not kwargs['eos_token_ids'] and kwargs['counter_sampling_backend'] == 'triton_top20'
        assert not kwargs.get('prefill_context') and not kwargs.get('stream')
        try:
            return generate(model, input_ids, attention_mask, **kwargs)
        finally:
            model.lm_head.__dict__['_edge_score_context'] = None
    return run
