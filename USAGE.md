# Running EdgeRock-4B

[Model overview](README.md) · [Runtime and measurement protocol](docs/runtime.md) · [Full results](docs/results.md)

This repository provides inference and evaluation code. Download the matching checkpoint from [Hugging Face](https://huggingface.co/rockmatrix-ai/EdgeRock). Run the commands below from this repository's root and pass the checkpoint directory with `--model`.

## Installation

Install PyTorch for your platform, then install the bundled Transformers dependency and runtime:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ./wheels/transformers-5.7.0.dev0-py3-none-any.whl -e .
hf download rockmatrix-ai/EdgeRock --local-dir checkpoints/EdgeRock-4B \
  --include '*.safetensors' '*.json' '*.jinja' 'tokenizer*' 'vocab*' 'merges*' \
  --exclude 'evaluation_data/*'
cp config.json configuration_edgerock.py modeling_edgerock.py checkpoints/EdgeRock-4B/
```

The runtime and checkpoint use `EdgeRockConfig`, `EdgeRockForCausalLM`, and the `edgerock` model type. The installation command copies the matching configuration and model modules from this release into the checkpoint directory. The runtime supplies `runtime/chat_template.jinja` for the text checkpoint; it opens the assistant reasoning segment with `<think>`.

### Portable PyTorch

```bash
python -I run.py demo --model checkpoints/EdgeRock-4B \
  --backend torch --device cpu --prompt 'Compute 17 × 23.' --max-new-tokens 512
```

Use `--backend torch --device cuda` for CUDA without custom extensions. CPU defaults to FP32; supported CUDA devices default to BF16. `--backend auto` selects the optimized path on Hopper when its dependencies are installed and otherwise uses PyTorch. The report's performance results use the optimized Hopper CUDA runtime.

For standard Transformers generation:

```python
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "checkpoints/EdgeRock-4B"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
tokenizer.chat_template = Path("runtime/chat_template.jinja").read_text()
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, dtype=dtype, backend="torch"
).to(device).eval()
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Compute 17 × 23."}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(text, return_tensors="pt").to(device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### Optimized Hopper environment

The supplied Dockerfile installs the pinned CUDA dependencies against its NVIDIA PyTorch base image. Build it on a Linux GPU host:

```bash
docker build -t edgerock-runtime .
docker run --rm --gpus all --ipc=host \
  -v "$PWD/checkpoints/EdgeRock-4B:/checkpoint:ro" \
  -v "$PWD/outputs:/outputs" edgerock-runtime \
  demo --model /checkpoint --output /outputs/demo --max-new-tokens 512
```

The CUDA path uses FLA, TileLang, causal-conv1d, FlashAttention, and Triton. See `requirements.txt` and the Dockerfile for the pinned environment. First use of a new shape can trigger compilation.

## Generate and serve

```bash
python -I run.py demo --model checkpoints/EdgeRock-4B \
  --backend cuda --profile fast --batch 1 \
  --prompt 'Compute 17 × 23.' --max-new-tokens 2048

python -I run.py generate --model checkpoints/EdgeRock-4B \
  --backend cuda --profile fast --batch 32 \
  --input prompts.jsonl --output outputs/generation
```

Each line of `prompts.jsonl` is a JSON object, for example:

```json
{"prompt":"Compute 17 × 23.","max_tokens":512}
```

`messages` and preformatted `prompt_text` inputs are also accepted. Output directories contain `responses.jsonl` and `metrics.json`.

Start the lightweight text API:

```bash
python -I run.py serve --model checkpoints/EdgeRock-4B \
  --backend cuda --profile fast --batch 1 --port 8000
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"EdgeRock-4B","messages":[{"role":"user","content":"Compute 17 × 23."}],"temperature":0,"max_tokens":512,"stream":true,"stream_options":{"include_usage":true}}'
```

The server supports `GET /v1/models`, text chat completions, SSE streaming, and token usage. It serializes generation calls; native frameworks provide scheduling for concurrent independent API requests. The batch endpoint at `/` accepts a `messages` array or a `prompts` array.

## Precision profiles

| Profile | Weights / MLP / output head | Local and anchor KV | GDN recurrence |
| --- | --- | --- | --- |
| `fast`, every batch size | BF16, unquantized | BF16 | FP32 |
| `reference`, CUDA BF16 | BF16, unquantized | BF16 | FP32 |

`fast` enables the selected BF16 optimizations and, at batch size 512, shared packed-weight storage. `reference` retains reference local-attention arithmetic. Both profiles retain BF16 MLP computation and FP32 recurrent state. The report uses the fast profile; all history remains on the GPU, and decoding is not speculative.

## Reproduce fixed-length performance workloads

```bash
# Single request, sustained generation
python -I run.py benchmark --model checkpoints/EdgeRock-4B \
  --profile fast --batch 1 --prefill 131072 --max-new-tokens 131072 \
  --output outputs/b1-128k-128k.json

# Batch size 512, 32K input and 32K output
python -I run.py benchmark --model checkpoints/EdgeRock-4B \
  --profile fast --batch 512 --prefill 32768 --max-new-tokens 32768 \
  --output outputs/b512-32k-32k.json

# Short workload
python -I run.py benchmark --model checkpoints/EdgeRock-4B \
  --profile fast --batch 1 --prefill 4096 --max-new-tokens 2048 \
  --output outputs/b1-4k-2k.json
```

Each invocation includes a separate 256-token warmup. Use three fresh processes per configuration. The benchmark uses synthetic tokens, temperature 1, top-k 20, top-p 0.95, presence penalty 1.5, EOS disabled, and fixed output length. Batch size 1 uses one prompt; larger batches use up to eight continuations per prompt (`--k 8` from batch size 8 onward). `--k 1` requests unique prompts.

The fixed-length single-request benchmark uses the selected output-head/sampling-state fusion. General generation uses the native generator to retain EOS, configurable sampling, streaming, and refill behavior. Both keep the same model precision and history capacity.

`decode_tokens_per_second` excludes prefill and the first token of each response. `end_to_end_output_tokens_per_second` counts all generated output tokens and includes prefill **time**, but never input tokens. Loading, warmup, tokenization, and text decoding are outside these timings. [Complete workload grid and results](docs/runtime.md).

## Native serving frameworks

Use separate environments for vLLM and SGLang and retain each framework's dependencies. The adapters implement fixed-memory state in request slots, on one GPU with tensor-parallel size one. Prefix caching and speculative decoding remain disabled. These are separate serving implementations from the report's specialized CUDA runtime.

### vLLM

The adapter has been exercised with vLLM `0.28.1rc1.dev13+g76c0c6530`, PyTorch `2.13.0+cu130`, and Transformers `5.16.1`:

```bash
python -m pip install --no-deps --no-build-isolation -e .
VLLM_PLUGINS=edgerock vllm serve "$PWD/checkpoints/EdgeRock-4B" \
  --trust-remote-code --served-model-name EdgeRock-4B \
  --chat-template "$PWD/runtime/chat_template.jinja" \
  --dtype bfloat16 --max-model-len 16384 \
  --max-num-batched-tokens 4096 --max-num-seqs 4 \
  --gpu-memory-utilization 0.6 --no-enable-prefix-caching \
  --mamba-cache-mode none --mamba-ssm-cache-dtype float32 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
  --additional-config '{"gdn_prefill_backend":"triton"}'
```

The `edgerock` plugin uses BF16 MLP execution and FP32 recurrent state.

### SGLang

The adapter has been exercised with SGLang `0.5.20`, PyTorch `2.13.0+cu130`, and Transformers `5.12.1`:

```bash
python -m pip install --no-deps --no-build-isolation -e .
SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_models python -m sglang.launch_server \
  --model-path "$PWD/checkpoints/EdgeRock-4B" --trust-remote-code --served-model-name EdgeRock-4B \
  --chat-template "$PWD/runtime/chat_template.jinja" \
  --dtype bfloat16 --context-length 16384 --chunked-prefill-size 4096 \
  --max-running-requests 4 --max-mamba-cache-size 4 \
  --max-total-tokens 65536 --mem-fraction-static 0.6 \
  --disable-radix-cache --disable-prefill-cuda-graph \
  --cuda-graph-bs-decode 1 2 4 \
  --attention-backend fa3 --page-size 1 \
  --linear-attn-prefill-backend triton --linear-attn-decode-backend triton \
  --mamba-ssm-dtype float32
```

Set request and Mamba-slot capacities together: anchor-state allocation follows Mamba-slot capacity. These framework paths have been exercised with complete EOS-terminated answers, concurrent requests, and slot reuse. The report's long-generation measurements come from the specialized runtime above.

## Evaluation

Download evaluation inputs and the visual frontend into the checkpoint directory, then install the evaluation dependencies:

```bash
hf download rockmatrix-ai/EdgeRock --local-dir checkpoints/EdgeRock-4B \
  --include 'evaluation_data/*' 'vision/*'
python -m pip install -e '.[evaluation]'

python -I run.py evaluate --model checkpoints/EdgeRock-4B \
  --suite text --gpus 0,1,2,3,4,5,6,7 --batch 320 --output outputs/text
python -I run.py evaluate --model checkpoints/EdgeRock-4B \
  --suite vision --gpus 0,1,2,3,4,5,6,7 --output outputs/vision
python -I run.py evaluate --model checkpoints/EdgeRock-4B \
  --suite memory --gpus 0,1,2,3,4,5,6,7 --output outputs/memory
python -I run.py status --output outputs/text
python -I run.py score --model checkpoints/EdgeRock-4B --output outputs/text
```

Use a fresh output directory for each evaluation. `--generate-only` stops after response generation; `score` runs CPU verification on existing fragments. `--sources` selects benchmarks, and `--k` overrides the automatic repetition count. The report's text settings are temperature 0.6, top-p 0.95, top-k 20, presence penalty 0, seed 20260923, and a 131,072-token response cap. Automatic repetition is 64 responses for sources below 100 questions, 16 for 100–10,000, and one above 10,000. Visual and memory diagnostics use reference arithmetic.

Final-answer extraction is independent of the reference answer. Code is checked by executable tests; IFEval and IFBench use strict instruction verification. DROP contributes F1 to the aggregate, and subject/subtask macro-averages are preserved where prescribed. [Results and metric definitions](docs/results.md) and Appendix A of the report give the complete protocol.

`STATUS.json` tracks attempts; `GENERATION_COMPLETE.json` records persisted generation counts; `RESULTS.json` contains scores, denominators, and sampling settings after verification. A partial run is marked `INCOMPLETE`.
