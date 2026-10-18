# EdgeRock-4B inference runtime and results

The [technical report](report/EdgeRock_Technical_Report.pdf) records historical measurements of 233.7 decode tokens/s at batch size 1 with 128K input and 128K output, and 12,124 end-to-end output tokens/s at batch size 512 with 32K input and 32K output, on one Hopper 80GB GPU. Those measurements predate the all-batch BF16-state and attention changes in this release. The current Runtime qualification is reported separately below.

## Runtime precision and history contract

| Component | CUDA runtime |
| --- | --- |
| All model weights, including MLP and output head | BF16, unquantized |
| Local and anchor KV | BF16 |
| GDN recurrent-state storage | BF16 at every batch size |
| GDN state updates | FP32 |
| Local attention window | 1,024 positions |
| Committed anchor capacity | 1,024 records per KV group |
| Candidate block | 256 positions |
| Speculative decoding | Disabled |
| History offload | Disabled |

## Constant-size persistent history

EdgeRock-4B maintains 97.76 MiB of persistent history per sequence, including BF16 recurrent matrices and auxiliary tensors, independently of sequence length. The configuration-derived attention-KV capacity is 72 MiB. During decode, the local ring stores 1,023 preceding positions while the current token is supplied by the decoder, so the audited local-plus-anchor KV payload is 71.97 MiB. Full device allocation also includes weights, buffers, graphs, and workspace.

## Optimized execution

The `fast` CUDA profile uses BF16 recurrent-state storage and retains the report's FP32 state-update accumulation and cache capacities. Its selected implementation combines BF16 projection and normalization fusions, joint memory projection, cross-layer residual/RMSNorm fusion, local-attention Q/K preparation, and CUDA Graph replay. At batch size 512, original projection parameters share byte-identical storage with packed BF16 weights, removing 4.95 GiB of duplicate allocation.

Physical-ring decode attention uses PyTorch `EFFICIENT_ATTENTION` with grouped query heads and views of the existing KV storage within each request. Independent requests keep separate caches. This avoids the cuDNN SDPA forward nondeterminism observed in the validated environment, without duplicating persistent KV. The frozen report measurements below predate this attention change; new Runtime throughput must be measured separately.

The fixed-length batch-size-1 benchmark additionally uses the validated output-head/sampling-state fusion. That path is restricted to its tested sampling contract: temperature 1.0, top-k 20, top-p 0.95, EOS disabled, no streaming or resumed prefill. General generation retains EOS, configurable sampling, and request-stream handling through the native generator. Both use BF16 weights/KV and FP32 GDN updates. All batch sizes store recurrent state in BF16.

`reference` retains the reference local-attention execution path. It also uses unquantized weights, BF16 recurrent-state storage and FP32 state-update accumulation. It is a correctness-oriented execution profile, not the report's throughput configuration.

## Throughput definitions

- **Decode rate:** subsequent output tokens, after each request's first token, divided by decode time. Timing includes sampling, state updates, graph management, and runtime scheduling, and excludes prefill.
- **End-to-end output-token throughput:** total generated output tokens divided by the full generation-call duration, including prefill time. **Input tokens never enter the numerator.**
- **Per-sequence rate:** aggregate decode rate divided by the active batch size.

Loading, warmup, tokenization, and text decoding are excluded from both throughput measures. `1K = 1,024` tokens.

## Current Runtime qualification (2026-10-18)

On one H800, with PyTorch 2.11.0+cu130 and cuDNN 91900, the released source passed two independent reference/release process pairs at both B1 and B512. Generated tokens matched across implementations and within each implementation's repeats. Actual BF16 prefill/decode storage, independent per-request prefill, cache bytes, and the original peak-memory tolerance also passed. The release ran without an SDPA wrapper; the frozen reference used an external correction to the same grouped efficient-attention backend. This establishes equivalence under that common backend, not bitwise equivalence to historical cuDNN outputs.

The speed comparison below uses 4K input plus 2K output per request, BF16 weights/KV/recurrent storage, FP32 state updates, no quantization/speculation/offload, and no prefix caching or cross-request reuse. Each implementation completed one full-generation warmup followed by three timed generations. Output TPS includes independent prefill and excludes loading and warmup. The baseline is the original default frozen runtime; the release includes the attention correction.

| Batch | Original default output TPS, three trials | Released output TPS, three trials | Original median | Released median | Change | Peak allocated GiB, both |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 230.410 / 230.385 / 219.525 | 210.746 / 210.726 / 203.972 | 230.385 | 210.726 | −8.53% | 14.270 |
| 512 | 8270.370 / 8272.755 / 8233.773 | 8218.094 / 8217.190 / 8168.149 | 8270.370 | 8217.190 | −0.64% | 63.548 |

The released implementation produced identical token hashes in all three timed repetitions at each batch. Cache bytes and peak allocation did not increase. These measurements document the performance cost of the attention correction; they are Runtime qualification results and do not replace the frozen report tables or the separate BF16 paper rerun. They are output TPS, not decode-only rates. Acceptance run `edgerock-bf16-runtime-acceptance-20261018-v17`, artifact hashes, and exact measurements are recorded in [SOURCE.json](../runtime/optimizations/SOURCE.json).

## Historical report workloads

Single-request sustained decoding uses 128K input and 128K output. The native-context boundary check uses 254K input and 2K output; both total 262,144 tokens. The long-generation grid combines input and output lengths in {4K, 8K, 16K, 32K}, at batch sizes 32, 64, 128, 256, 320, 384, and 512. The short grid uses 4K input and 2K output at batch sizes 1, 2, 4, 8, 16, 32, 64, 128, 256, 320, 384, and 512.

Each EdgeRock configuration has three independent runs. Per-workload results use medians. Weighted batch results divide pooled subsequent output tokens by pooled decode time across the 16 length pairs and three repetitions. Memory is the maximum measured across the relevant runs.

The benchmark uses synthetic tokens, a separate 256-token warmup, EOS disabled, temperature 1.0, top-k 20, top-p 0.95, and presence penalty 1.5. Eight continuations share each prompt in the batch grid and then retain independent histories. Batch size denotes submitted requests; EdgeRock maintains target concurrency, while vLLM schedules within its cache budget.

## Historical report measurements

These are the published report measurements. The BF16-state rerun is in progress; these tables have not yet been replaced with its results.

The following tables report fixed-length performance on one Hopper 80GB GPU with BF16 weights and KV, FP32 GDN updates, and tensor-parallel size one. Recurrent-state storage is FP32 at batch size 1 and FP16 for larger batches.

### End-to-end output throughput, 32K output

Each cell is output tokens/s (best measured batch size).

| Model | 4K input | 8K input | 16K input | 32K input |
| --- | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | **12,479** (512) | **12,325** (512) | **12,330** (512) | **12,124** (512) |
| Qwen3.5-4B | 3,103 (64) | 2,853 (64) | 2,472 (64) | 1,889 (64) |
| Gemma 4 E4B | 3,940 (128) | 3,432 (128) | 2,880 (128) | 2,196 (64) |
| MiniCPM5-2B | 3,152 (64) | 2,938 (64) | 2,575 (32) | 2,117 (32) |
| LFM2.5-2.6B | 1,673 (64) | 1,771 (64) | 1,695 (32) | 1,727 (64) |

Mean workload-specific ratios to Qwen3.5-4B and LFM2.5-2.6B are 4.94× and 7.18×. At 32K input and 32K output on one Hopper 80GB GPU, EdgeRock-4B reaches 12,124 output tokens/s at batch size 512 versus 1,889 output tokens/s for Qwen3.5-4B at batch size 64: **6.42× higher end-to-end output-token throughput**. Each model uses its best measured batch size; elapsed time includes prefill and the numerator counts output tokens only.

### Batch scaling over all 16 length pairs

| Batch size | Weighted decode tokens/s | Per-sequence tokens/s | Peak allocated GiB | Peak reserved GiB |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 4,128 | 129.0 | 18.94 | 19.49 |
| 64 | 6,685 | 104.5 | 20.16 | 21.80 |
| 128 | 9,049 | 70.7 | 26.74 | 27.92 |
| 256 | 11,231 | 43.9 | 39.90 | 41.77 |
| 320 | 11,074 | 34.6 | 46.49 | 50.69 |
| 384 | 12,099 | 31.5 | 53.06 | 57.27 |
| 512 | 12,497 | 24.4 | 61.26 | 66.97 |

![Concurrency scaling](../assets/batch_scaling.png)

### Short-context grid: 4K input, 2K output

| Batch size | Decode tokens/s | Per-sequence tokens/s | Peak allocated GiB |
| ---: | ---: | ---: | ---: |
| 1 | 230 | 229.6 | 14.27 |
| 2 | 358 | 178.8 | 14.27 |
| 4 | 692 | 172.9 | 14.27 |
| 8 | 1,322 | 165.2 | 14.40 |
| 16 | 2,442 | 152.6 | 15.21 |
| 32 | 4,053 | 126.7 | 16.84 |
| 64 | 6,698 | 104.7 | 20.10 |
| 128 | 9,066 | 70.8 | 26.60 |
| 256 | 11,100 | 43.4 | 39.62 |
| 320 | 11,062 | 34.6 | 46.14 |
| 384 | 12,086 | 31.5 | 52.64 |
| 512 | 12,481 | 24.4 | 60.71 |

![GPU memory scaling](../assets/memory_capacity.png)

## Source identity

The report uses the model revision `e10f2b1818813b45ca53a7c98d757f388368cedb` and base runtime commit `49e4b0f8fdf0ff508f7570c6b3ef3eada3c8747a`, with the selected BF16 fusion stack and B512 shared-weight patch. The released optimizations carry source hashes in `runtime/optimizations/SOURCE.json`.
