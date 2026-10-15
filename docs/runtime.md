# EdgeRock-4B inference runtime and results

EdgeRock-4B sustains 233.7 decode tokens/s at batch size 1 with 128K input and 128K output, and reaches 10,669 end-to-end output tokens/s at batch size 512 with 32K input and 32K output. The [technical report](report/EdgeRock_Technical_Report.pdf) measures both on one Hopper 80GB GPU with tensor-parallel size one and ordinary autoregressive decoding.

## Precision and history contract

| Component | Report profile |
| --- | --- |
| All model weights, including MLP and output head | BF16, unquantized |
| Local and anchor KV | BF16 |
| GDN recurrent state, from the first prefill write onward | FP32 |
| Local attention window | 1,024 positions |
| Committed anchor capacity | 1,024 records per KV group |
| Candidate block | 256 positions |
| Speculative decoding | Disabled |
| History offload | Disabled |

## Constant-size persistent history

EdgeRock-4B maintains 121.76 MiB of persistent history per sequence, including FP32 recurrence and auxiliary tensors, independently of sequence length. The configuration-derived attention-KV capacity is 72 MiB. During decode, the local ring stores 1,023 preceding positions while the current token is supplied by the decoder, so the audited local-plus-anchor KV payload is 71.97 MiB. Full device allocation also includes weights, buffers, graphs, and workspace.

## Optimized execution

The `fast` CUDA profile retains the report's arithmetic and cache capacities. Its selected implementation combines BF16 projection and normalization fusions, joint memory projection, cross-layer residual/RMSNorm fusion, local-attention Q/K preparation, and CUDA Graph replay. At batch size 512, original projection parameters share byte-identical storage with packed BF16 weights, removing 4.95 GiB of duplicate allocation.

The fixed-length batch-size-1 benchmark additionally uses the validated output-head/sampling-state fusion. That path is restricted to its tested sampling contract: temperature 1.0, top-k 20, top-p 0.95, EOS disabled, no streaming or resumed prefill. General generation retains EOS, configurable sampling, and request-stream handling through the native generator. Both use BF16 weights/KV and FP32 recurrence.

`reference` retains the reference local-attention execution path. It also uses unquantized weights and FP32 recurrence. It is a correctness-oriented execution profile, not the report's throughput configuration.

## Throughput definitions

- **Decode rate:** subsequent output tokens, after each request's first token, divided by decode time. Timing includes sampling, state updates, graph management, and runtime scheduling, and excludes prefill.
- **End-to-end output-token throughput:** total generated output tokens divided by the full generation-call duration, including prefill time. **Input tokens never enter the numerator.**
- **Per-sequence rate:** aggregate decode rate divided by the active batch size.

Loading, warmup, tokenization, and text decoding are excluded from both throughput measures. `1K = 1,024` tokens.

## Workloads

Single-request sustained decoding uses 128K input and 128K output. The native-context boundary check uses 254K input and 2K output; both total 262,144 tokens. The long-generation grid combines input and output lengths in {4K, 8K, 16K, 32K}, at batch sizes 32, 64, 128, 256, 320, 384, and 512. The short grid uses 4K input and 2K output at batch sizes 1, 2, 4, 8, 16, 32, 64, 128, 256, 320, 384, and 512.

Each EdgeRock configuration has three independent runs. Per-workload results use medians. Weighted batch results divide pooled subsequent output tokens by pooled decode time across the 16 length pairs and three repetitions. Memory is the maximum measured across the relevant runs.

The benchmark uses synthetic tokens, a separate 256-token warmup, EOS disabled, temperature 1.0, top-k 20, top-p 0.95, and presence penalty 1.5. Eight continuations share each prompt in the batch grid and then retain independent histories. Batch size denotes submitted requests; EdgeRock maintains target concurrency, while vLLM schedules within its cache budget.

## Measured results

The following tables reproduce the latest report. They are measurements of the frozen implementation used in the report, not new measurements of a different checkpoint. Capability evaluation is separate from these fixed-length performance workloads.

### End-to-end output throughput, 32K output

Each cell is output tokens/s (best measured batch size).

| Model | 4K input | 8K input | 16K input | 32K input |
| --- | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | **11,032** (512) | **10,906** (512) | **10,874** (512) | **10,669** (512) |
| Qwen3.5-4B | 3,103 (64) | 2,853 (64) | 2,472 (64) | 1,889 (64) |
| Gemma 4 E4B | 3,940 (128) | 3,432 (128) | 2,880 (128) | 2,196 (64) |
| MiniCPM5-2B | 3,152 (64) | 2,938 (64) | 2,575 (32) | 2,117 (32) |
| LFM2.5-2.6B | 1,673 (64) | 1,771 (64) | 1,695 (32) | 1,727 (64) |

Mean workload-specific ratios to Qwen3.5-4B and LFM2.5-2.6B are 4.36× and 6.34×. At 32K input and 32K output on one Hopper 80GB GPU, EdgeRock-4B reaches 10,669 output tokens/s at batch size 512 versus 1,889 output tokens/s for Qwen3.5-4B at batch size 64: **5.65× higher end-to-end output-token throughput**. Each model uses its best measured batch size; elapsed time includes prefill and the numerator counts output tokens only.

### Batch scaling over all 16 length pairs

| Batch size | Weighted decode tokens/s | Per-sequence tokens/s | Peak allocated GiB | Peak reserved GiB |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 3,920 | 122.5 | 19.08 | 19.74 |
| 64 | 6,176 | 96.5 | 21.66 | 22.73 |
| 128 | 8,251 | 64.5 | 29.74 | 31.32 |
| 256 | 10,041 | 39.2 | 45.90 | 50.89 |
| 320 | 9,924 | 31.0 | 53.99 | 59.22 |
| 384 | 10,723 | 27.9 | 62.06 | 67.38 |
| 512 | 11,038 | 21.6 | 73.26 | 77.76 |

![Concurrency scaling](../assets/batch_scaling.png)

### Short-context grid: 4K input, 2K output

| Batch size | Decode tokens/s | Per-sequence tokens/s | Peak allocated GiB |
| ---: | ---: | ---: | ---: |
| 1 | 229 | 229.4 | 14.29 |
| 2 | 356 | 177.9 | 14.29 |
| 4 | 670 | 167.5 | 14.29 |
| 8 | 1,262 | 157.8 | 14.59 |
| 16 | 2,356 | 147.2 | 15.59 |
| 32 | 3,905 | 122.0 | 17.59 |
| 64 | 6,180 | 96.6 | 21.60 |
| 128 | 8,263 | 64.6 | 29.60 |
| 256 | 10,018 | 39.1 | 45.62 |
| 320 | 9,914 | 31.0 | 53.64 |
| 384 | 10,693 | 27.8 | 61.64 |
| 512 | 11,047 | 21.6 | 72.71 |

![GPU memory scaling](../assets/memory_capacity.png)

## Source identity

The report uses the model revision `e10f2b1818813b45ca53a7c98d757f388368cedb` and base runtime commit `49e4b0f8fdf0ff508f7570c6b3ef3eada3c8747a`, with the selected BF16 fusion stack and B512 shared-weight patch. The released optimizations carry source hashes in `runtime/optimizations/SOURCE.json`.
