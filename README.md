<div align="center">
  <img src="assets/rockmatrix.png" alt="Rock Matrix" width="64" />
  <h1>EdgeRock</h1>
  <h3>Strong Reasoning at Constant Memory</h3>
  <p>Rock Matrix</p>
  <p>
    <a href="docs/report/EdgeRock_Technical_Report.pdf"><b>Technical Report</b></a>
    &nbsp; · &nbsp;
    <a href="https://huggingface.co/rockmatrix-ai/EdgeRock"><b>Model Weights</b></a>
    &nbsp; · &nbsp;
    <a href="USAGE.md"><b>Inference Guide</b></a>
    &nbsp; · &nbsp;
    <a href="docs/results.md"><b>Evaluation Results</b></a>
  </p>
</div>

**EdgeRock is a family of multimodal models that deliver strong reasoning at constant memory.** EdgeRock-4B is the first open-source release, with text and image reasoning; additional model sizes are in preparation.

Long reasoning grows the KV cache, consumes accelerator memory, and limits serving concurrency. On edge devices, reasoning must also share memory with perception and control. EdgeRock fixes the history budget: recurrent state summarizes context, local attention covers recent tokens, and **Rock Memory Anchors (RMA)** keep selected distant details directly retrievable.

- **Strong reasoning:** 82.9% mean pass@1 on AIME 2024–2026, versus at most 18.8% among the evaluated bounded-state models, including 14B-class models.
- **Constant history:** 72 MiB attention KV per sequence; 121.76 MiB including FP32 recurrence and auxiliary history. At 256K context, attention KV is 99.1% smaller than the source backbone's cache.
- **Sustained inference:** 233.7 output tokens/s for a single 128K-input/128K-output request. At the best measured batch, up to 5.65× the source backbone's end-to-end output throughput on one Hopper 80GB GPU.
- **Multimodal capability:** 84.9% mean answer-level accuracy across six visual benchmarks, retaining 95.3% of the source model's mean.

![Reasoning, KV memory, and inference speed](assets/hero_capability.png)

[Full results](docs/results.md) cover bounded-state capability, source-model retention, and compact-model comparisons. Throughput counts **output tokens only**; end-to-end elapsed time includes prefill. All reported runtime comparisons use BF16 weights/KV and ordinary autoregressive decoding, with FP32 recurrence for EdgeRock and its backbone.

## Quick start

```bash
git clone https://github.com/rockmatrix-ai/EdgeRock.git
cd EdgeRock
python -m venv .venv
source .venv/bin/activate
python -m pip install ./wheels/transformers-5.7.0.dev0-py3-none-any.whl -e .
hf download rockmatrix-ai/EdgeRock --local-dir checkpoints/EdgeRock-4B \
  --include '*.safetensors' '*.json' '*.py' '*.jinja' 'tokenizer*' 'vocab*' 'merges*' \
  --exclude 'evaluation_data/*'
python run.py demo --model checkpoints/EdgeRock-4B \
  --prompt 'Solve 17 × 23, explaining your reasoning.' --max-new-tokens 2048
```

The portable backend provides PyTorch execution on CPU and CUDA. The optimized Hopper CUDA path requires the additional dependencies in the [inference guide](USAGE.md). The **fast** profile preserves BF16 weights/KV and FP32 recurrent state; it changes execution, not precision.

## How RMA works

![EdgeRock architecture](assets/architecture_overview.png)

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

A learned writer produces dedicated key–value records. Every 256 tokens, admission retains the highest-priority records separately in each of four KV groups. Queries read the **committed** bank; newly selected records become visible in the next block. Retained records remain separate and unchanged. A query-dependent read and signed gate integrate retrieved information alongside local attention.

Every component of persistent history has fixed capacity. History storage is **O(1) in sequence length**, and bounded history access gives **O(N) generation work** for N output tokens at fixed model and memory dimensions. Total GPU memory also includes model weights, input/output buffers, CUDA Graphs, and workspace.

## Capability

### Strong reasoning within the bounded-state class

EdgeRock-4B leads the 17-model bounded-state comparison on 24 of 29 text benchmarks. Its 77.2% equal-weight mean is 23.8 percentage points above the strongest aggregate baseline, AHN-GDN Qwen2.5 14B. On AIME 2024–2026, EdgeRock-4B reaches 82.9%; the strongest bounded-state baseline reaches 18.8%.

![Bounded-state capability](assets/bounded_comparison.png)

### Reasoning retained after conversion

EdgeRock-4B retains **96.3%** of its Qwen3.5-4B source model's AIME mean and reaches **97.5%** on MATH-500. It exceeds the source model on eight text benchmarks, including MMLU-Pro, SuperGPQA, HumanEval, and MBPP. The report provides the complete paired comparison.

### Competitive compact-model capability

EdgeRock-4B's 82.9% AIME mean lies between Granite4.2-3B (82.2%) and Qwen3.5-4B (86.1%), with MiniCPM5-2B at 87.8%. It exceeds LFM2.5-2.6B on 28 of 29 text benchmarks while keeping persistent history constant. See [compact-model results](docs/results.md#compact-model-capability).

### Distant memory that reasoning can use

Across 256 contexts of 16K–130,000 input tokens, EdgeRock-4B answers **94.1%** of queries and gets all four answers correct in **80.1%** of contexts. The tasks cover retrieval, state updates, cross-document combination, and variable binding. Disabling RMA at inference reduces query accuracy to 0.4%.

![RMA training and contribution](assets/memory_diagnostic.png)

## Inference efficiency

The reported execution profile uses **BF16 weights, MLP, output head, and KV; FP32 GDN recurrence; full-vocabulary autoregressive sampling; and GPU-resident state**. CUDA Graphs and fused BF16 operations reduce data movement and launch overhead. Shared storage for original and packed BF16 projections removes 4.95 GiB of duplicate weights and enables batch size 512 without reducing history capacity.

| Workload | EdgeRock-4B result |
| --- | ---: |
| Single request, 128K input + 128K output | 233.7 decode tokens/s |
| Single request, 254K input + 2K output | 226.6 decode tokens/s |
| 32K input + 32K output, best batch size 512 | 10,669 end-to-end output tokens/s |
| Batch size 512, 16 input/output pairs from 4K–32K | 11,038 weighted decode tokens/s |
| Same batch-size-512 sweep | 21.6 decode tokens/s per active sequence |
| Peak allocated memory across that sweep | 73.26 GiB |

![Sustained decoding and best-batch throughput](assets/runtime_scaling.png)

At 32K input and 32K output, EdgeRock-4B achieves **5.65×** Qwen3.5-4B's throughput at each runtime's best measured batch. Across the four 32K-output workloads, the mean ratio is **4.36×**. Each model uses its best measured batch size for this serving-throughput comparison. Single-request decode rates exclude prefill; end-to-end output-token throughput includes prefill time and excludes input tokens from its numerator. [Full measurements and protocol](docs/runtime.md).

## Repository guide

- [USAGE.md](USAGE.md): installation, generation, serving, and benchmark commands.
- [docs/runtime.md](docs/runtime.md): precision, optimized execution, and reproduction conditions.
- [docs/results.md](docs/results.md): bounded-state, compact-model, source-model, and visual results.
- [docs/report/EdgeRock_Technical_Report.pdf](docs/report/EdgeRock_Technical_Report.pdf): current English technical report.
- [docs/report/latex/main.tex](docs/report/latex/main.tex): report source with the original figures.
- `runtime/`: inference, model implementation, optimized kernels, evaluation, and scoring.

## Citation

```bibtex
@techreport{edgerock2026,
  title = {EdgeRock: Strong Reasoning at Constant Memory},
  author = {Ji, Huishan and Zhang, Weiming and Ye, Qirui and Xu, Jinchao and Guo, Yike and Xu, Hua},
  institution = {Rock Matrix},
  year = {2026},
  url = {https://github.com/rockmatrix-ai/EdgeRock}
}
```

## License

Runtime code is distributed under [Apache-2.0](LICENSE). Model weights are governed by the [model license](MODEL_LICENSE). See [NOTICE](NOTICE) for attribution and third-party terms.
