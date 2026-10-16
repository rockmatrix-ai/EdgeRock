<div align="center">
  <img src="assets/rockmatrix.png" alt="Rock Matrix" width="64" />
  <h1>EdgeRock</h1>
  <h3>Strong Reasoning at Constant Memory</h3>
  <p>Rock Matrix</p>
  <p>
    <a href="docs/report/EdgeRock_Technical_Report.pdf"><b>Technical Report</b></a>
    &nbsp; · &nbsp;
    <a href="https://arxiv.org/abs/2610.10118"><b>arXiv:2610.10118</b></a>
    &nbsp; · &nbsp;
    <a href="https://github.com/rockmatrix-ai/EdgeRock"><b>GitHub</b></a>
    &nbsp; · &nbsp;
    <a href="https://huggingface.co/rockmatrix-ai/EdgeRock"><b>Hugging Face</b></a>
    &nbsp; · &nbsp;
    <a href="https://modelscope.cn/models/rockmatrix/EdgeRock"><b>ModelScope</b></a>
  </p>
</div>

**EdgeRock is a family of multimodal models that deliver strong reasoning at constant memory.** EdgeRock-4B is the first open-source release, with text and image reasoning; additional model sizes are in preparation.

With full-history attention, the key–value (KV) cache grows throughout long-context inference. Longer reasoning traces leave room for fewer concurrent requests; on edge devices, that memory also supports perception and control. EdgeRock keeps persistent history fixed through recurrent state, sliding-window attention, and **Rock Memory Anchors (RMA)**: learned key–value records that preserve direct access to selected distant information.

- **82.9% AIME pass@1 at constant memory.** EdgeRock-4B averages 82.9% across AIME 2024–2026, versus 18.8% for RWKV-7 G1j in the 14B class, the strongest evaluated bounded-state baseline on AIME.
- **O(1) attention-KV storage.** EdgeRock-4B keeps a fixed 72 MiB KV capacity per sequence, **99.1% less** than Qwen3.5-4B at 256K context with BF16 KV.
- **5.65× higher batch throughput.** At 32K input + 32K output on one Hopper 80GB GPU, EdgeRock-4B delivers 5.65× Qwen3.5-4B's end-to-end output throughput at each model's best measured batch size.
- **Text and image reasoning.** EdgeRock-4B reaches **84.9%** mean answer-level accuracy across six visual benchmarks, retaining **95.3%** of Qwen3.5-4B's mean accuracy.

![EdgeRock-4B: AIME reasoning, constant attention-KV storage, and Hopper 80GB inference throughput](assets/hero_capability.png)

Results below come from the [EdgeRock technical report](docs/report/EdgeRock_Technical_Report.pdf). Throughput counts **output tokens only**; end-to-end elapsed time includes prefill. Runtime comparisons use BF16 weights/KV and ordinary autoregressive decoding, with FP32 recurrence for EdgeRock-4B and Qwen3.5-4B.

[Quick start](#quick-start) · [Architecture](#how-rock-memory-anchors-work) · [Reasoning](#capability) · [Throughput and memory](#inference-efficiency) · [All benchmark tables](#full-text-benchmark-tables) · [FAQ](#frequently-asked-questions) · [Citation](#citation)

## Quick start

```bash
git clone https://github.com/rockmatrix-ai/EdgeRock.git
cd EdgeRock
python -m venv .venv
source .venv/bin/activate
python -m pip install ./wheels/transformers-5.7.0.dev0-py3-none-any.whl -e .
hf download rockmatrix-ai/EdgeRock --local-dir checkpoints/EdgeRock-4B \
  --include '*.safetensors' '*.json' '*.jinja' 'tokenizer*' 'vocab*' 'merges*' \
  --exclude 'evaluation_data/*'
cp config.json configuration_edgerock.py modeling_edgerock.py checkpoints/EdgeRock-4B/
python run.py demo --model checkpoints/EdgeRock-4B \
  --prompt 'Solve 17 × 23, explaining your reasoning.' --max-new-tokens 2048
```

EdgeRock-4B runs through the portable PyTorch backend on CPU or CUDA. For the optimized Hopper path, follow the [inference guide](USAGE.md); the **fast** profile uses fused BF16 operations and CUDA Graphs while preserving BF16 weights/KV and FP32 recurrent state.

## How Rock Memory Anchors work

In EdgeRock, a **memory anchor** is a learned key–value record that later queries can retrieve after its source information leaves the local attention window. Rock Memory Anchors keep these records in a fixed-capacity bank. The writer creates records, admission selects which to retain, and query-dependent retrieval brings stored information into the current computation.

![EdgeRock-4B architecture: recurrent state, sliding-window attention, and Rock Memory Anchors](assets/architecture_overview.png)

EdgeRock-4B converts the eight full-attention layers of its Qwen3.5-4B backbone while retaining 24 Gated DeltaNet layers and the feed-forward blocks:

```text
[GDN, GDN, GDN, sliding-window attention ∥ RMA] × 8
```

| History component | Role | Capacity |
| --- | --- | --- |
| Recurrent state | Summarize the evolving context | Fixed-dimensional matrices |
| Local attention | Resolve recent token interactions | 1,024 positions |
| RMA anchor bank | Retain individually retrievable distant records | 1,024 records per KV group |
| Pending candidates | Collect records for the next admission update | 256 positions |

Every 256 tokens, EdgeRock selects up to 1,024 records from the committed bank and new candidates in each of four KV groups. Queries read the **committed bank**; selections become visible in the next block. Each retained record stays individually addressable until replaced, and a query-dependent signed gate combines retrieved information with local attention.

EdgeRock fixes the capacity of every history component, giving **O(1) persistent history in sequence length** and **O(N) generation work** for N output tokens at fixed model and memory dimensions. Total GPU allocation includes this history, model weights, input/output buffers, CUDA Graphs, and workspace.

## Capability

### Strong reasoning within the bounded-state class

EdgeRock-4B reaches **82.9% mean pass@1 on AIME 2024–2026** with fixed persistent history; RWKV-7 G1j 13.3B, the strongest bounded-state baseline on AIME, reaches 18.8%. Across 29 text benchmarks, EdgeRock-4B leads on 24 and averages **77.2%**, compared with 53.4% for AHN-GDN Qwen2.5 14B.

![Bounded-state capability: all-benchmark and AIME means for 17 models](assets/bounded_comparison.png)

**Bounded-state comparison.** All scores are percentages. The 29-benchmark mean weights benchmarks equally; AIME weights the 2024–2026 editions equally. Bold marks column maxima, including ties. W2047 denotes Phi-3 Medium's configured 2,047-token attention window.

| Model | 29-benchmark mean | AIME mean | MATH-500 | MMLU-Pro | LiveCodeBench v6 | IFEval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | **77.2** | **82.9** | **97.5** | **79.8** | **61.6** | **88.0** |
| AHN-GDN Qwen2.5 14B | 53.4 | 11.4 | 80.0 | 63.5 | 32.9 | 77.4 |
| RWKV-7 G1j 13.3B | 52.3 | 18.8 | 77.2 | 47.6 | 30.7 | 77.8 |
| Phi-3 Medium 4K Instruct (W2047) | 45.4 | 2.2 | 49.0 | 54.0 | 19.1 | 61.4 |
| RWKV-7 G1j 7.2B | 43.5 | 8.1 | 64.2 | 46.2 | 21.3 | 66.5 |
| RWKV-7 G1i 7.2B | 42.9 | 9.4 | 68.6 | 44.9 | 18.8 | 53.4 |
| AHN-GDN Qwen2.5 3B | 40.9 | 4.2 | 66.2 | 42.8 | 14.6 | 58.6 |
| RWKV-7 G1j 2.9B | 37.0 | 2.2 | 48.2 | 37.9 | 12.1 | 56.4 |
| RWKV-7 G1i 2.9B | 35.3 | 1.8 | 43.8 | 37.8 | 11.6 | 43.1 |
| QRWKV7 7B Instruct | 32.6 | 0.0 | 41.0 | 23.0 | 12.4 | 59.0 |
| Falcon3-Mamba 7B Instruct | 31.6 | 1.2 | 41.8 | 26.1 | 9.3 | 64.7 |
| RecurrentGemma 9B IT | 29.9 | 0.4 | 22.8 | 13.4 | 7.9 | 39.7 |
| RWKV-7 G1j 1.5B | 28.4 | 1.1 | 31.4 | 16.6 | 6.3 | 46.2 |
| RWKV-7 G1i 1.5B | 27.9 | 0.3 | 28.2 | 13.3 | 7.2 | 37.0 |
| ARWKV-R1 7B | 18.1 | 0.1 | 33.2 | 8.8 | 2.6 | 10.4 |
| xLSTM 7B | 16.0 | 0.0 | 25.0 | 4.6 | 2.2 | 34.0 |
| ARWKV-R1 1.5B | 9.9 | 0.0 | 13.6 | 7.5 | 0.3 | 11.1 |

### EdgeRock vs Qwen, MiniCPM, and LFM

EdgeRock-4B brings constant-history reasoning into the range of compact models: **82.9%** mean AIME pass@1, compared with 86.1% for Qwen3.5-4B, 87.8% for MiniCPM5-2B, and 56.2% for LFM2.5-2.6B. EdgeRock-4B also leads this compact-model comparison on MMLU-Pro and SuperGPQA, and exceeds LFM2.5-2.6B on 28 of 29 text benchmarks.

EdgeRock-4B retains fixed history; the comparison models use growing history. Scores are percentages, AIME averages 2024–2026, and bold marks column maxima.

| Model | AIME mean | MMLU-Pro | SuperGPQA | GPQA-Diamond | LiveCodeBench v6 | IFEval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | 82.9 | **79.8** | **60.0** | 64.6 | 61.6 | 88.0 |
| Qwen3.5-4B | 86.1 | 77.7 | 52.8 | **77.3** | 75.7 | 89.6 |
| Gemma4-E4B-it | 40.1 | 69.0 | 41.1 | 55.6 | 69.1 | 84.3 |
| Nemotron-3-Nano-4B | 57.1 | 56.8 | 37.4 | 43.4 | 66.4 | 89.5 |
| Phi-4-mini-reasoning | 38.9 | 38.5 | 32.2 | 48.0 | 35.3 | 45.7 |
| MiniCPM5-2B | **87.8** | 65.3 | 40.9 | 67.7 | **83.3** | 86.3 |
| LFM2.5-2.6B | 56.2 | 59.9 | 27.1 | 58.2 | 59.7 | **93.3** |

### Reasoning retained after conversion

EdgeRock-4B retains **96.3%** of its Qwen3.5-4B backbone's AIME mean after replacing full attention with a fixed-capacity history. It reaches **97.5%** on MATH-500 and exceeds its backbone on eight text benchmarks, including MMLU-Pro, SuperGPQA, HumanEval, and MBPP.

![Capability after conversion: EdgeRock-4B and Qwen3.5-4B across all 29 text benchmarks](assets/backbone_delta.png)

The figure orders the 29 benchmarks by EdgeRock-4B minus Qwen3.5-4B in percentage points. Tables report absolute scores (%); bold marks the higher score. Retained is EdgeRock-4B's score divided by Qwen3.5-4B's, expressed as a percentage.

| Benchmark | EdgeRock-4B | Qwen3.5-4B | Retained (%) |
| --- | ---: | ---: | ---: |
| AIME 2024 | 86.7 | **89.6** | 96.8 |
| AIME 2025 | 77.4 | **80.8** | 95.8 |
| AIME 2026 | 84.6 | **87.9** | 96.2 |
| HMMT Feb 2025 | 71.8 | **74.6** | 96.2 |
| HMMT Nov 2025 | 69.8 | **78.3** | 89.1 |
| HMMT Feb 2026 | 63.6 | **65.9** | 96.5 |
| MATH-500 | 97.5 | **98.0** | 99.5 |
| AIME three-year mean | 82.9 | **86.1** | 96.3 |

#### EdgeRock-4B vs Qwen3.5-4B: all 29 benchmarks

| Benchmark | EdgeRock-4B | Qwen3.5-4B |
| --- | ---: | ---: |
| GSM8K | 94.5 | **94.8** |
| MATH-500 | 97.5 | **98.0** |
| AIME 2024 | 86.7 | **89.6** |
| AIME 2025 | 77.4 | **80.8** |
| AIME 2026 | 84.6 | **87.9** |
| HMMT Feb 2026 | 63.6 | **65.9** |
| MMLU | **88.2** | 85.2 |
| MMLU-Pro | **79.8** | 77.7 |
| MMLU-Redux | **92.0** | 89.8 |
| CMMLU | 73.3 | **83.9** |
| C-Eval | 74.6 | **84.0** |
| GPQA-Diamond | 64.6 | **77.3** |
| SuperGPQA | **60.0** | 52.8 |
| WinoGrande | 77.9 | **87.1** |
| HellaSwag | 64.9 | **80.6** |
| ARC-C | 95.5 | **96.8** |
| ARC-E | 98.0 | **98.4** |
| BBH | 85.4 | **86.2** |
| BBEH | 28.9 | **34.6** |
| HumanEval | **96.7** | 86.0 |
| MBPP | **87.8** | 71.0 |
| LiveCodeBench v6 | 61.6 | **75.7** |
| IFEval | 88.0 | **89.6** |
| IFBench | **65.3** | 60.3 |
| DROP | **77.2** | 76.0 |
| MuSR | 60.4 | **64.9** |
| LogiQA2 | 72.2 | **82.2** |
| HMMT Feb 2025 | 71.8 | **74.6** |
| HMMT Nov 2025 | 69.8 | **78.3** |

### Visual reasoning

EdgeRock-4B supports text and image reasoning and reaches **84.9%** equal-weight mean visual accuracy, retaining **95.3%** of Qwen3.5-4B's mean accuracy.

Answer-level accuracy (%) on six visual benchmarks. Retained is EdgeRock-4B's score divided by Qwen3.5-4B's score. Bold marks the higher accuracy in each row.

| Benchmark | EdgeRock-4B | Qwen3.5-4B | Retained (%) |
| --- | ---: | ---: | ---: |
| MMBench EN | 88.3 | **90.1** | 98.0 |
| MMBench CN | 89.0 | **91.4** | 97.4 |
| POPE | 85.4 | **88.8** | 96.2 |
| MME | 83.0 | **86.8** | 95.6 |
| ChartQA | 83.9 | **90.3** | 92.9 |
| AI2D | 80.0 | **87.0** | 92.0 |
| Equal-weight mean | 84.9 | **89.1** | 95.3 |

**Standard MME scores (benchmark points):**

| MME metric | EdgeRock-4B | Qwen3.5-4B |
| --- | ---: | ---: |
| Perception (points) | 1,600.4 | **1,738.3** |
| Cognition (points) | 728.2 | **751.8** |
| Total (points) | 2,328.7 | **2,490.1** |

### Distant memory that reasoning can use

EdgeRock-4B answers **94.1%** of distant-memory queries across 256 contexts of 16K–130,000 input tokens, versus **0.4%** with all eight RMA modules disabled. It answers all four queries correctly in **80.1%** of contexts. The tasks test retrieval, state updates, cross-document combination, and variable binding.

![Distant-memory reasoning across adaptation stages and with RMA enabled or disabled](assets/memory_diagnostic.png)

**Memory across adaptation stages.** EdgeRock learns distant retrieval during RMA adaptation and retains it through joint adaptation, supervised fine-tuning, and reinforcement learning with verifiable rewards (RLVR). Each stage uses the same 256 contexts and 1,024 queries; complete-context success requires four correct answers.

| Adaptation stage | Query accuracy (%) | All four correct (%) |
| --- | ---: | ---: |
| RMA adaptation (M0) | 79.4 | 54.7 |
| Joint adaptation (R) | **96.1** | **87.9** |
| Supervised fine-tuning (S) | 93.8 | 82.4 |
| Reinforcement learning (RLVR) | 94.1 | 80.1 |

**RMA contribution in the final model.** Enabling the eight RMA modules raises correct distant-memory answers from 4 to 964 out of 1,024. The recurrent backbone and local attention remain active in both settings.

| Final-model setting | Correct queries / total | Query accuracy (%) | Complete contexts / total | All four correct (%) |
| --- | ---: | ---: | ---: | ---: |
| RMA enabled | **964 / 1,024** | **94.1** | **205 / 256** | **80.1** |
| RMA disabled | 4 / 1,024 | 0.4 | 0 / 256 | 0.0 |

**Results by task.** Each task contains 64 contexts and requires four correct answers per context.

| Task | Complete contexts / total | All four correct (%) |
| --- | ---: | ---: |
| Exact retrieval | 58 / 64 | 90.6 |
| State-update interpretation | 56 / 64 | 87.5 |
| Cross-document combination | 57 / 64 | 89.1 |
| Variable binding | 34 / 64 | 53.1 |
| All tasks | 205 / 256 | 80.1 |

**Paired evidence changes.** Success requires both answers in a pair to be correct.

| Change to distant evidence | Correct pairs / total | Pair accuracy (%) |
| --- | ---: | ---: |
| Answer should change | 230 / 256 | 89.8 |
| Answer should stay the same | 247 / 256 | 96.5 |

## Inference efficiency

EdgeRock-4B turns fixed history into serving capacity: **512 concurrent long-generation sequences on one Hopper 80GB GPU**. The comparisons below use BF16 weights and KV, ordinary autoregressive decoding, and GPU-resident state; EdgeRock-4B and Qwen3.5-4B use FP32 GDN recurrence. CUDA Graphs, fused BF16 operations, and shared projection storage reduce execution overhead while preserving precision and history capacity.

### Single-request speed and best-batch throughput

![Long-context inference: single-request decoding and best-batch output-token throughput](assets/runtime_scaling.png)

#### Single-request decoding

EdgeRock-4B sustains **233.7 decode tokens/s**, versus 151.3 for Qwen3.5-4B (**1.54×**), with **128K input + 128K output at batch size 1 on one Hopper 80GB GPU**. Decode rate counts output tokens after the first token and includes sampling and state updates; timing excludes prefill.

| Model | Decode tokens/s |
| --- | ---: |
| **EdgeRock-4B** | **233.7** |
| Qwen3.5-4B | 151.3 |
| Gemma 4 E4B | 133.8 |
| MiniCPM5-2B | 164.5 |
| LFM2.5-2.6B | 229.6 |

At the same 256K total length with **254K input + 2K output**, EdgeRock-4B reaches **226.6** decode tokens/s versus Qwen3.5-4B's 133.9 (**1.69×**).

#### Best-batch output throughput

EdgeRock-4B reaches **10,669 output tokens/s at batch size 512**, versus Qwen3.5-4B's 1,889 at batch size 64, for **5.65× higher throughput** at **32K input + 32K output on one Hopper 80GB GPU**. Each model uses its best measured batch.

All columns use 32K output. Cells give output tokens/s with batch size in parentheses. End-to-end throughput divides **output tokens only** by elapsed generation time, **including prefill**.

| Model | 4K input | 8K input | 16K input | 32K input |
| --- | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | **11,032** (512) | **10,906** (512) | **10,874** (512) | **10,669** (512) |
| Qwen3.5-4B | 3,103 (64) | 2,853 (64) | 2,472 (64) | 1,889 (64) |
| Gemma 4 E4B | 3,940 (128) | 3,432 (128) | 2,880 (128) | 2,196 (64) |
| MiniCPM5-2B | 3,152 (64) | 2,938 (64) | 2,575 (32) | 2,117 (32) |
| LFM2.5-2.6B | 1,673 (64) | 1,771 (64) | 1,695 (32) | 1,727 (64) |

Across 4K, 8K, 16K, and 32K inputs with 32K output, EdgeRock-4B averages **4.36×** Qwen3.5-4B's and **6.34×** LFM2.5-2.6B's output throughput at each model's best measured batch size.

### Concurrency scaling

![Concurrency scaling: aggregate and per-sequence decode rates at batch sizes 32–512](assets/batch_scaling.png)

EdgeRock-4B reaches **11,038 weighted decode tokens/s** across all 16 input/output pairs formed from 4K, 8K, 16K, and 32K lengths at batch size 512 on one Hopper 80GB GPU. This gives **21.6 tokens/s per active sequence**. Weighted throughput divides pooled subsequent output tokens by pooled decode time; memory columns report the peak across these workloads.

| Batch size | Weighted decode tokens/s | Per-sequence tokens/s | Peak allocated GiB | Peak reserved GiB |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 3,920 | 122.5 | 19.08 | 19.74 |
| 64 | 6,176 | 96.5 | 21.66 | 22.73 |
| 128 | 8,251 | 64.5 | 29.74 | 31.32 |
| 256 | 10,041 | 39.2 | 45.90 | 50.89 |
| 320 | 9,924 | 31.0 | 53.99 | 59.22 |
| 384 | 10,723 | 27.9 | 62.06 | 67.38 |
| 512 | **11,038** | 21.6 | 73.26 | 77.76 |

### Short-context decoding

For **4K input + 2K output on one Hopper 80GB GPU**, EdgeRock-4B decodes at **229.4 tokens/s at batch size 1** and **11,047 tokens/s at batch size 512**. Rates exclude prefill; all batches use BF16 weights/KV and FP32 recurrence.

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
| 512 | **11,047** | 21.6 | 72.71 |

### Constant history and GPU memory

![GPU memory scaling: concurrency and input length at fixed batch sizes](assets/memory_capacity.png)

EdgeRock-4B keeps attention-KV capacity at **72 MiB per sequence**, independent of sequence length. At 256K context, Qwen3.5-4B requires 8 GiB of attention KV at the same BF16 precision: EdgeRock uses **99.1% less**. These are configuration-derived KV capacities; the complete persistent-history budget follows.

| Model | 32K context (MiB) | 64K context (MiB) | 128K context (MiB) | 256K context (MiB) | History growth |
| --- | ---: | ---: | ---: | ---: | --- |
| **EdgeRock-4B** | **72** | **72** | **72** | **72** | **O(1)** |
| Qwen3.5-4B | 1,024 | 2,048 | 4,096 | 8,192 | O(N) |

**Complete persistent history: 121.76 MiB per sequence.** This includes attention KV, FP32 recurrence, and metadata. During decode, the local ring stores 1,023 preceding positions and the decoder supplies the current token.

| Component | MiB per sequence |
| --- | ---: |
| Committed RMA anchors | 32.00 |
| Pending RMA candidates | 8.00 |
| Local-window KV | 31.97 |
| FP32 recurrent matrices | 48.00 |
| Convolution state | 1.50 |
| Metadata | 0.29 |
| **Total persistent history** | **121.76** |

**Measured GPU allocation.** On one Hopper 80GB GPU, EdgeRock-4B peaks at **16.02 GiB** for 128K input + 128K output at batch size 1 and **73.26 GiB** across the 16 long-generation workloads at batch size 512. These allocations include weights, history, input/output buffers, and workspace. The figure shows how allocation varies with concurrency and input length.

**Runtime protocol.** 1K = 1,024 tokens. EdgeRock configurations use three independent runs; per-workload values are medians, and weighted batch rates pool the 16 length pairs and three repetitions. Timing excludes loading, warmup, tokenization, and text decoding. Workloads use synthetic tokens, a separate 256-token warmup, EOS disabled, temperature 1.0, top-k 20, top-p 0.95, and presence penalty 1.5. In the batch grid, eight continuations share each prompt and retain independent histories. Batch size denotes submitted requests; EdgeRock maintains target concurrency, while vLLM schedules within its cache budget.

## Full text benchmark tables

EdgeRock-4B is compared with **16 bounded-state baselines** and **11 growing-history models** across all 29 text benchmarks. Scores use each benchmark's primary metric on a 0–100 scale; **bold marks column maxima, including ties**. AIME reports mean pass@1, and DROP reports F1.

### Bounded-state comparison: all 29 benchmarks

The comparison contains EdgeRock-4B and 16 bounded-state baselines. Phi-3 Medium uses a configured 2,047-token attention window.

#### Mathematics: bounded-state models

| Model | GSM8K | MATH-500 | AIME 2024 | AIME 2025 | AIME 2026 | HMMT Feb 2025 | HMMT Nov 2025 | HMMT Feb 2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | 94.5 | **97.5** | **86.7** | **77.4** | **84.6** | **71.8** | **69.8** | **63.6** |
| ARWKV-R1 1.5B | 24.8 | 13.6 | 0.0 | 0.0 | 0.0 | 0.4 | 0.8 | 1.1 |
| RWKV-7 G1i 1.5B | 60.3 | 28.2 | 0.0 | 0.4 | 0.4 | 1.7 | 5.0 | 0.0 |
| RWKV-7 G1j 1.5B | 57.2 | 31.4 | 1.2 | 2.1 | 0.0 | 1.7 | 4.2 | 0.0 |
| RWKV-7 G1i 2.9B | 71.7 | 43.8 | 1.7 | 1.7 | 2.1 | 1.7 | 5.0 | 0.8 |
| RWKV-7 G1j 2.9B | 74.1 | 48.2 | 0.4 | 4.6 | 1.7 | 2.1 | 5.0 | 0.4 |
| AHN-GDN Qwen2.5 3B | 83.8 | 66.2 | 7.1 | 2.1 | 3.3 | 2.9 | 6.7 | 4.9 |
| ARWKV-R1 7B | 51.2 | 33.2 | 0.0 | 0.0 | 0.4 | 0.8 | 1.7 | 1.1 |
| Falcon3-Mamba 7B Instruct | 61.4 | 41.8 | 1.7 | 0.8 | 1.2 | 0.8 | 5.4 | 2.7 |
| QRWKV7 7B Instruct | 65.5 | 41.0 | 0.0 | 0.0 | 0.0 | 0.0 | 6.7 | 0.0 |
| xLSTM 7B | 43.4 | 25.0 | 0.0 | 0.0 | 0.0 | 1.7 | 5.0 | 1.5 |
| RWKV-7 G1i 7.2B | 85.1 | 68.6 | 8.8 | 11.7 | 7.9 | 2.1 | 5.0 | 3.4 |
| RWKV-7 G1j 7.2B | 81.8 | 64.2 | 8.3 | 11.2 | 4.6 | 4.2 | 4.2 | 0.8 |
| RecurrentGemma 9B IT | 51.7 | 22.8 | 0.4 | 0.8 | 0.0 | 1.2 | 5.4 | 1.9 |
| RWKV-7 G1j 13.3B | **94.6** | 77.2 | 17.5 | 27.5 | 11.2 | 3.3 | 5.8 | 12.1 |
| AHN-GDN Qwen2.5 14B | 93.6 | 80.0 | 13.8 | 10.0 | 10.4 | 7.5 | 10.0 | 4.9 |
| Phi-3 Medium 4K Instruct | 85.3 | 49.0 | 3.3 | 3.3 | 0.0 | 6.7 | 2.5 | 3.0 |

#### Knowledge and science: bounded-state models

| Model | MMLU | MMLU-Pro | MMLU-Redux | CMMLU | C-Eval | GPQA-Diamond | SuperGPQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | **88.2** | **79.8** | **92.0** | 73.3 | **74.6** | **64.6** | **60.0** |
| ARWKV-R1 1.5B | 17.4 | 7.5 | 17.9 | 15.6 | 15.2 | 13.1 | 5.8 |
| RWKV-7 G1i 1.5B | 49.7 | 13.3 | 55.4 | 33.6 | 33.0 | 26.8 | 17.7 |
| RWKV-7 G1j 1.5B | 49.3 | 16.6 | 53.8 | 35.3 | 33.3 | 26.3 | 16.1 |
| RWKV-7 G1i 2.9B | 63.6 | 37.8 | 63.4 | 36.1 | 32.5 | 26.8 | 21.6 |
| RWKV-7 G1j 2.9B | 64.7 | 37.9 | 58.9 | 37.9 | 36.5 | 31.8 | 21.8 |
| AHN-GDN Qwen2.5 3B | 62.7 | 42.8 | 65.7 | 62.4 | 62.9 | 31.3 | 23.0 |
| ARWKV-R1 7B | 32.3 | 8.8 | 35.2 | 29.0 | 29.0 | 13.1 | 8.8 |
| Falcon3-Mamba 7B Instruct | 57.7 | 26.1 | 64.8 | 33.4 | 35.2 | 20.7 | 12.2 |
| QRWKV7 7B Instruct | 64.6 | 23.0 | 34.1 | 57.7 | 56.0 | 8.6 | 15.4 |
| xLSTM 7B | 19.6 | 4.6 | 37.9 | 16.9 | 19.5 | 12.1 | 5.3 |
| RWKV-7 G1i 7.2B | 69.4 | 44.9 | 73.5 | 49.8 | 51.6 | 30.3 | 25.7 |
| RWKV-7 G1j 7.2B | 67.5 | 46.2 | 72.7 | 50.8 | 48.9 | 36.9 | 25.3 |
| RecurrentGemma 9B IT | 55.5 | 13.4 | 55.0 | 43.4 | 41.7 | 24.7 | 19.5 |
| RWKV-7 G1j 13.3B | 76.7 | 47.6 | 81.8 | 67.4 | 66.3 | 42.4 | 31.8 |
| AHN-GDN Qwen2.5 14B | 77.2 | 63.5 | 83.5 | **75.0** | 74.2 | 40.9 | 35.5 |
| Phi-3 Medium 4K Instruct | 76.3 | 54.0 | 81.4 | 53.3 | 51.3 | 26.8 | 25.6 |

#### Commonsense and reasoning: bounded-state models

| Model | WinoGrande | HellaSwag | ARC-C | ARC-E | BBH | BBEH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | 77.9 | 64.9 | **95.5** | **98.0** | **85.4** | **28.9** |
| ARWKV-R1 1.5B | 31.4 | 7.4 | 19.6 | 22.3 | 9.8 | 1.6 |
| RWKV-7 G1i 1.5B | 46.6 | 33.4 | 68.6 | 83.0 | 27.8 | 4.6 |
| RWKV-7 G1j 1.5B | 49.3 | 24.6 | 66.2 | 80.8 | 31.4 | 5.0 |
| RWKV-7 G1i 2.9B | 57.4 | 62.7 | 81.9 | 92.2 | 30.3 | 4.5 |
| RWKV-7 G1j 2.9B | 58.1 | 65.9 | 83.3 | 91.6 | 39.5 | 6.5 |
| AHN-GDN Qwen2.5 3B | 61.7 | 52.3 | 76.6 | 86.1 | 39.5 | 4.6 |
| ARWKV-R1 7B | 34.4 | 10.0 | 47.4 | 56.8 | 16.6 | 2.4 |
| Falcon3-Mamba 7B Instruct | 57.5 | 38.6 | 74.8 | 87.6 | 32.8 | 3.4 |
| QRWKV7 7B Instruct | 61.8 | 48.0 | 86.3 | 95.4 | 37.4 | 1.9 |
| xLSTM 7B | 28.3 | 7.9 | 25.1 | 28.2 | 12.8 | 3.3 |
| RWKV-7 G1i 7.2B | 60.4 | 70.2 | 84.5 | 94.5 | 38.8 | 7.0 |
| RWKV-7 G1j 7.2B | 59.4 | 71.7 | 83.7 | 94.5 | 41.4 | 7.7 |
| RecurrentGemma 9B IT | 56.4 | 45.8 | 73.3 | 87.2 | 35.5 | 3.2 |
| RWKV-7 G1j 13.3B | 72.1 | **77.4** | 91.8 | 96.3 | 48.7 | 7.7 |
| AHN-GDN Qwen2.5 14B | 74.0 | 67.9 | 91.5 | 94.0 | 52.0 | 5.3 |
| Phi-3 Medium 4K Instruct | **79.6** | 74.0 | 89.8 | 95.3 | 53.3 | 3.1 |

#### Code and instruction following: bounded-state models

| Model | HumanEval | MBPP | LiveCodeBench v6 | IFEval | IFBench |
| --- | ---: | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | **96.7** | **87.8** | **61.6** | **88.0** | **65.3** |
| ARWKV-R1 1.5B | 4.3 | 5.4 | 0.3 | 11.1 | 8.3 |
| RWKV-7 G1i 1.5B | 42.1 | 36.6 | 7.2 | 37.0 | 14.7 |
| RWKV-7 G1j 1.5B | 41.5 | 38.6 | 6.3 | 46.2 | 16.0 |
| RWKV-7 G1i 2.9B | 58.5 | 49.2 | 11.6 | 43.1 | 17.0 |
| RWKV-7 G1j 2.9B | 58.5 | 48.2 | 12.1 | 56.4 | 17.3 |
| AHN-GDN Qwen2.5 3B | 67.7 | 49.0 | 14.6 | 58.6 | 26.3 |
| ARWKV-R1 7B | 16.5 | 21.2 | 2.6 | 10.4 | 8.3 |
| Falcon3-Mamba 7B Instruct | 34.1 | 39.2 | 9.3 | 64.7 | 18.7 |
| QRWKV7 7B Instruct | 62.2 | 30.2 | 12.4 | 59.0 | 21.3 |
| xLSTM 7B | 36.6 | 9.2 | 2.2 | 34.0 | 14.7 |
| RWKV-7 G1i 7.2B | 65.2 | 59.0 | 18.8 | 53.4 | 17.7 |
| RWKV-7 G1j 7.2B | 72.6 | 58.0 | 21.3 | 66.5 | 18.3 |
| RecurrentGemma 9B IT | 37.2 | 19.0 | 7.9 | 39.7 | 12.7 |
| RWKV-7 G1j 13.3B | 86.0 | 67.4 | 30.7 | 77.8 | 27.3 |
| AHN-GDN Qwen2.5 14B | 75.6 | 68.6 | 32.9 | 77.4 | 33.3 |
| Phi-3 Medium 4K Instruct | 73.8 | 59.4 | 19.1 | 61.4 | 22.0 |

#### Reading and multi-step reasoning: bounded-state models

| Model | DROP F1 | MuSR | LogiQA2 |
| --- | ---: | ---: | ---: |
| **EdgeRock-4B** | **77.2** | **60.4** | 72.2 |
| ARWKV-R1 1.5B | 5.6 | 8.6 | 17.5 |
| RWKV-7 G1i 1.5B | 23.0 | 19.3 | 38.7 |
| RWKV-7 G1j 1.5B | 24.9 | 29.1 | 36.1 |
| RWKV-7 G1i 2.9B | 24.6 | 31.9 | 47.6 |
| RWKV-7 G1j 2.9B | 30.7 | 34.5 | 45.4 |
| AHN-GDN Qwen2.5 3B | 32.8 | 37.8 | 50.1 |
| ARWKV-R1 7B | 11.1 | 15.4 | 26.8 |
| Falcon3-Mamba 7B Instruct | 19.9 | 29.1 | 41.4 |
| QRWKV7 7B Instruct | 21.2 | 2.6 | 33.2 |
| xLSTM 7B | 29.0 | 19.3 | 20.0 |
| RWKV-7 G1i 7.2B | 33.3 | 49.6 | 53.1 |
| RWKV-7 G1j 7.2B | 34.2 | 48.6 | 56.4 |
| RecurrentGemma 9B IT | 37.1 | 35.2 | 40.2 |
| RWKV-7 G1j 13.3B | 51.5 | 51.4 | 66.6 |
| AHN-GDN Qwen2.5 14B | 66.5 | 56.4 | **72.3** |
| Phi-3 Medium 4K Instruct | 51.5 | 51.1 | 62.7 |

### Comparison with growing-history models: all 29 benchmarks

EdgeRock-4B retains constant persistent history; the eleven comparison models retain history that grows with sequence length.

#### Mathematics: EdgeRock vs growing-history models

| Model | GSM8K | MATH-500 | AIME 2024 | AIME 2025 | AIME 2026 | HMMT Feb 2025 | HMMT Nov 2025 | HMMT Feb 2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | 94.5 | 97.5 | 86.7 | 77.4 | 84.6 | 71.8 | 69.8 | 63.6 |
| Qwen3.5-4B | **94.8** | **98.0** | **89.6** | 80.8 | 87.9 | 74.6 | 78.3 | 65.9 |
| Gemma4-E2B-it | 90.1 | 78.8 | 38.3 | 33.8 | 33.8 | 21.7 | 17.5 | 20.1 |
| MiniCPM5-2B | 93.6 | 97.0 | 87.5 | 85.4 | 90.4 | 76.7 | 74.6 | 63.3 |
| LFM2.5-2.6B | 91.0 | 92.7 | 63.4 | 56.3 | 48.9 | 34.7 | 42.9 | 32.9 |
| Qwen3.5-2B | 87.6 | 86.8 | 41.2 | 34.2 | 35.4 | 26.2 | 23.3 | 25.4 |
| Granite4.2-3B | 93.8 | 97.8 | 85.0 | 78.8 | 82.9 | 75.8 | 72.5 | 59.5 |
| Phi-4-mini-reasoning | 93.6 | 91.6 | 45.4 | 32.5 | 38.8 | 25.0 | 28.7 | 24.6 |
| Gemma4-E4B-it | 91.3 | 88.2 | 42.1 | 40.0 | 38.3 | 25.4 | 32.9 | 28.8 |
| Nemotron-3-Nano-4B | 92.4 | 94.8 | 60.8 | 56.7 | 53.8 | 40.0 | 47.1 | 36.4 |
| Granite4.2-8B | 94.4 | **98.0** | 88.3 | **87.9** | **91.7** | **81.7** | **86.7** | **73.9** |
| MiniCPM4.1-8B | 93.6 | 96.4 | 77.1 | 70.0 | 70.8 | 58.3 | 59.6 | 49.2 |

#### Knowledge and science: EdgeRock vs growing-history models

| Model | MMLU | MMLU-Pro | MMLU-Redux | CMMLU | C-Eval | GPQA-Diamond | SuperGPQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | **88.2** | **79.8** | **92.0** | 73.3 | 74.6 | 64.6 | **60.0** |
| Qwen3.5-4B | 85.2 | 77.7 | 89.8 | **83.9** | **84.0** | **77.3** | 52.8 |
| Gemma4-E2B-it | 72.2 | 47.0 | 78.2 | 60.2 | 61.2 | 46.5 | 34.1 |
| MiniCPM5-2B | 79.6 | 65.3 | 85.3 | 79.2 | 78.3 | 67.7 | 40.9 |
| LFM2.5-2.6B | 75.3 | 59.9 | 81.7 | 57.2 | 59.5 | 58.2 | 27.1 |
| Qwen3.5-2B | 72.5 | 62.8 | 79.5 | 68.2 | 68.7 | 54.5 | 39.3 |
| Granite4.2-3B | 77.2 | 66.4 | 81.6 | 59.1 | 60.8 | 58.6 | 39.5 |
| Phi-4-mini-reasoning | 77.5 | 38.5 | 78.3 | 45.3 | 42.4 | 48.0 | 32.2 |
| Gemma4-E4B-it | 81.2 | 69.0 | 86.4 | 68.7 | 68.6 | 55.6 | 41.1 |
| Nemotron-3-Nano-4B | 75.8 | 56.8 | 81.5 | 54.6 | 56.4 | 43.4 | 37.4 |
| Granite4.2-8B | 78.6 | 72.0 | 86.3 | 55.1 | 58.4 | 61.6 | 45.7 |
| MiniCPM4.1-8B | 82.7 | 67.0 | 87.8 | 80.2 | 79.0 | 51.5 | 40.6 |

#### Commonsense and reasoning: EdgeRock vs growing-history models

| Model | WinoGrande | HellaSwag | ARC-C | ARC-E | BBH | BBEH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | 77.9 | 64.9 | 95.5 | 98.0 | 85.4 | 28.9 |
| Qwen3.5-4B | **87.1** | **80.6** | **96.8** | **98.4** | 86.2 | **34.6** |
| Gemma4-E2B-it | 63.2 | 50.9 | 89.2 | 94.8 | 80.3 | 11.6 |
| MiniCPM5-2B | 69.8 | 52.2 | 91.9 | 97.0 | 82.1 | 25.4 |
| LFM2.5-2.6B | 67.9 | 40.7 | 91.6 | 96.5 | 76.7 | 13.1 |
| Qwen3.5-2B | 71.5 | 59.3 | 91.4 | 95.5 | 79.1 | 14.9 |
| Granite4.2-3B | 74.0 | 67.4 | 92.4 | 97.1 | 85.5 | 24.5 |
| Phi-4-mini-reasoning | 75.8 | 62.6 | 90.2 | 94.4 | 72.5 | 12.9 |
| Gemma4-E4B-it | 76.5 | 53.5 | 94.3 | 97.7 | 84.1 | 17.4 |
| Nemotron-3-Nano-4B | 65.6 | 46.3 | 91.9 | 96.7 | 79.3 | 10.8 |
| Granite4.2-8B | 76.2 | 63.1 | 90.9 | 94.6 | **88.9** | 31.5 |
| MiniCPM4.1-8B | 78.2 | 49.8 | 93.4 | 97.4 | 86.6 | 14.0 |

#### Code and instruction following: EdgeRock vs growing-history models

| Model | HumanEval | MBPP | LiveCodeBench v6 | IFEval | IFBench |
| --- | ---: | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | 96.7 | 87.8 | 61.6 | 88.0 | 65.3 |
| Qwen3.5-4B | 86.0 | 71.0 | 75.7 | 89.6 | 60.3 |
| Gemma4-E2B-it | 90.9 | 79.6 | 59.6 | 82.4 | 29.0 |
| MiniCPM5-2B | 97.6 | 94.8 | 83.3 | 86.3 | 64.0 |
| LFM2.5-2.6B | 92.9 | 73.6 | 59.7 | 93.3 | 56.2 |
| Qwen3.5-2B | 54.9 | 35.4 | 25.0 | 81.5 | 43.7 |
| Granite4.2-3B | 97.0 | 92.6 | 75.9 | 92.2 | 74.3 |
| Phi-4-mini-reasoning | 79.9 | 64.0 | 35.3 | 45.7 | 14.3 |
| Gemma4-E4B-it | 93.9 | 79.2 | 69.1 | 84.3 | 34.7 |
| Nemotron-3-Nano-4B | 90.9 | 72.2 | 66.4 | 89.5 | 57.7 |
| Granite4.2-8B | 96.3 | **95.4** | **85.6** | **94.6** | **77.0** |
| MiniCPM4.1-8B | **98.2** | 92.2 | 70.4 | 73.4 | 23.7 |

#### Reading and multi-step reasoning: EdgeRock vs growing-history models

| Model | DROP F1 | MuSR | LogiQA2 |
| --- | ---: | ---: | ---: |
| **EdgeRock-4B** | 77.2 | 60.4 | 72.2 |
| Qwen3.5-4B | 76.0 | 64.9 | **82.2** |
| Gemma4-E2B-it | 72.9 | 58.9 | 59.4 |
| MiniCPM5-2B | 76.0 | **65.3** | 73.7 |
| LFM2.5-2.6B | 72.3 | 60.4 | 63.2 |
| Qwen3.5-2B | 73.7 | 61.3 | 66.3 |
| Granite4.2-3B | **83.7** | 65.1 | 69.5 |
| Phi-4-mini-reasoning | 50.6 | 56.7 | 58.5 |
| Gemma4-E4B-it | 72.9 | 62.2 | 72.6 |
| Nemotron-3-Nano-4B | 72.6 | 58.2 | 67.9 |
| Granite4.2-8B | 76.9 | 62.4 | 73.8 |
| MiniCPM4.1-8B | 78.3 | 64.5 | 75.7 |

### Capability evaluation protocol

Most tasks report accuracy; DROP uses F1 in the 29-benchmark mean. The aggregate weights benchmarks equally after each benchmark’s required subject or subtask aggregation. AIME pools retained responses within each edition and weights the three editions equally. Repetition counts vary by model and benchmark.

EdgeRock text evaluation uses temperature 0.6, top-p 0.95, top-k 20, presence penalty 0, seed 20260923, and a 131,072-token response cap. It uses 64 independent responses per item for benchmarks with fewer than 100 questions, 16 for 100–10,000 questions, and one above 10,000. Visual and memory evaluations use the reference arithmetic path. Appendix A of the [technical report](docs/report/EdgeRock_Technical_Report.pdf) gives the scoring and verification procedures.

## Frequently asked questions

### What does O(1) memory mean for EdgeRock?

EdgeRock-4B uses a fixed amount of **persistent history per sequence**, independent of sequence length at fixed cache capacities. That history includes recurrence, local KV, RMA records, pending candidates, and metadata. The report configuration uses 121.76 MiB per sequence. Total GPU allocation also includes model weights and execution buffers; it scales with concurrency.

### Why does fixed history matter for agents and robots?

Agents can generate long reasoning traces, while robots and other edge devices share memory with perception and control. EdgeRock fixes the history budget before generation starts, making that part of memory demand predictable. In server inference, the same design leaves capacity for more concurrent requests, demonstrated by EdgeRock-4B's 512-sequence long-generation results above.

### Where can I download and run EdgeRock-4B?

Rock Matrix publishes EdgeRock-4B weights on [Hugging Face](https://huggingface.co/rockmatrix-ai/EdgeRock) and [ModelScope](https://modelscope.cn/models/rockmatrix/EdgeRock). This repository contains the model implementation and inference code; the [quick start](#quick-start) runs a text-generation example. Runtime code uses the [Apache-2.0 license](LICENSE), and weights use the [EdgeRock model license](MODEL_LICENSE).

## Repository guide

- [USAGE.md](USAGE.md): installation, generation, serving, and benchmark commands.
- [docs/runtime.md](docs/runtime.md): precision, optimized execution, and reproduction conditions.
- [docs/results.md](docs/results.md): bounded-state, compact-model, source-model, and visual results.
- [docs/report/EdgeRock_Technical_Report.pdf](docs/report/EdgeRock_Technical_Report.pdf): current English technical report.
- [docs/report/latex/main.tex](docs/report/latex/main.tex): report source with the original figures.
- `runtime/`: inference, model implementation, optimized kernels, evaluation, and scoring.
- [llms.txt](llms.txt): concise navigation to model definitions, results, and inference documentation.

## Citation

```bibtex
@techreport{edgerock2026,
  title = {EdgeRock: Strong Reasoning at Constant Memory},
  author = {Ji, Huishan and Zhang, Weiming and Ye, Qirui and Xu, Jinchao and Guo, Yike and Xu, Hua},
  institution = {Rock Matrix},
  year = {2026},
  eprint = {2610.10118},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2610.10118}
}
```

## License

Runtime code is distributed under [Apache-2.0](LICENSE). Model weights are governed by the [model license](MODEL_LICENSE). See [NOTICE](NOTICE) for attribution and third-party terms.
