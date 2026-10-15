# EdgeRock-4B evaluation results

EdgeRock-4B combines strong mathematical reasoning with constant-size persistent history. It reaches 82.9% mean pass@1 on AIME 2024–2026 and retains 96.3% of its Qwen3.5-4B backbone's AIME mean. The tables below reproduce the [English technical report](report/EdgeRock_Technical_Report.pdf), with separate comparisons for bounded-state capability, source-model retention, and compact-model capability.

[Bounded-state models](#bounded-state-models) · [Source-model retention](#source-model-retention) · [Compact models](#compact-model-capability) · [Visual reasoning](#visual-reasoning) · [Runtime results](runtime.md)

## Bounded-state models

EdgeRock-4B reaches 82.9% mean pass@1 on AIME 2024–2026, compared with 18.8% for RWKV-7 G1j 13.3B, the strongest evaluated bounded-state baseline on AIME. The fixed-capacity comparison covers EdgeRock-4B and 16 bounded-state models. Scores use each benchmark’s primary metric on a 0–100 scale. Bold marks column maxima, including ties.

### Mathematics

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

### Knowledge and science

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

### Commonsense and reasoning

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

### Code and instruction following

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

### Reading and multi-step reasoning

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

## Compact-model capability

EdgeRock-4B keeps constant persistent history. The other models in this table have growing history. AIME is the equal-weight 2024–2026 mean.

| Model | AIME | MMLU-Pro | SuperGPQA | GPQA-Diamond | LiveCodeBench v6 | IFEval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **EdgeRock-4B** | 82.9 | **79.8** | **60.0** | 64.6 | 61.6 | 88.0 |
| Qwen3.5-4B | 86.1 | 77.7 | 52.8 | **77.3** | 75.7 | 89.6 |
| Gemma4-E4B-it | 40.1 | 69.0 | 41.1 | 55.6 | 69.1 | 84.3 |
| Nemotron-3-Nano-4B | 57.1 | 56.8 | 37.4 | 43.4 | 66.4 | 89.5 |
| Phi-4-mini-reasoning | 38.9 | 38.5 | 32.2 | 48.0 | 35.3 | 45.7 |
| MiniCPM5-2B | **87.8** | 65.3 | 40.9 | 67.7 | **83.3** | 86.3 |
| LFM2.5-2.6B | 56.2 | 59.9 | 27.1 | 58.2 | 59.7 | **93.3** |

## Source-model retention

EdgeRock-4B retains 96.3% of Qwen3.5-4B’s AIME 2024–2026 mean after conversion to constant-size persistent history. Retained is EdgeRock-4B’s score divided by Qwen3.5-4B’s score, expressed as a percentage.

| Benchmark | EdgeRock-4B | Qwen3.5-4B | Retained (%) |
| --- | ---: | ---: | ---: |
| AIME 2024 | 86.7 | 89.6 | 96.8 |
| AIME 2025 | 77.4 | 80.8 | 95.8 |
| AIME 2026 | 84.6 | 87.9 | 96.2 |
| HMMT Feb 2025 | 71.8 | 74.6 | 96.2 |
| HMMT Nov 2025 | 69.8 | 78.3 | 89.1 |
| HMMT Feb 2026 | 63.6 | 65.9 | 96.5 |
| MATH-500 | 97.5 | 98.0 | 99.5 |
| AIME three-year mean | 82.9 | 86.1 | 96.3 |

## Growing-history models

These eleven models retain history that grows with sequence length. Their capability results place EdgeRock among compact releases; the source-model comparison separately measures retention.

### Mathematics

| Model | GSM8K | MATH-500 | AIME 2024 | AIME 2025 | AIME 2026 | HMMT Feb 2025 | HMMT Nov 2025 | HMMT Feb 2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
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

### Knowledge and science

| Model | MMLU | MMLU-Pro | MMLU-Redux | CMMLU | C-Eval | GPQA-Diamond | SuperGPQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | **85.2** | **77.7** | **89.8** | **83.9** | **84.0** | **77.3** | **52.8** |
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

### Commonsense and reasoning

| Model | WinoGrande | HellaSwag | ARC-C | ARC-E | BBH | BBEH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
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

### Code and instruction following

| Model | HumanEval | MBPP | LiveCodeBench v6 | IFEval | IFBench |
| --- | ---: | ---: | ---: | ---: | ---: |
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

### Reading and multi-step reasoning

| Model | DROP F1 | MuSR | LogiQA2 |
| --- | ---: | ---: | ---: |
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

## Visual reasoning

| Benchmark | EdgeRock-4B answer accuracy (%) |
| --- | ---: |
| MMBench EN | 88.3 |
| MMBench CN | 89.0 |
| POPE | 85.4 |
| MME | 83.0 |
| ChartQA | 83.9 |
| AI2D | 80.0 |
| Equal-weight mean | 84.9 |

Across MMBench EN/CN, POPE, MME, ChartQA, and AI2D, EdgeRock-4B reaches 84.9% equal-weight mean answer-level accuracy, versus 89.1% for Qwen3.5-4B (95.3% retained). MME answer accuracy is distinct from standard MME points; the report lists both.

## Evaluation protocol

Most tasks report accuracy; DROP uses F1 in the 29-benchmark mean. The aggregate weights benchmarks equally after each benchmark’s required subject or subtask aggregation. AIME pools retained responses within each edition and weights the three editions equally. Repetition counts vary by model and benchmark.

EdgeRock text evaluation uses temperature 0.6, top-p 0.95, top-k 20, presence penalty 0, seed 20260923, and a 131,072-token response cap. It uses 64 independent responses per item for benchmarks with fewer than 100 questions, 16 for 100–10,000 questions, and one above 10,000. Visual and memory evaluations use the reference arithmetic path. See Appendix A of the report for verification and scoring details.
