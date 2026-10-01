# Ported SDAR-1.7B-Chat-b64 results

These are the completed exp05 reference points that exp06_vgb will match with
SDAR-8B-Chat-b64. Each arm contains 100 scored tasks.

| Benchmark | fixed k | Primary score | Tokens/s | Seconds/task |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 1 | 0.80 | 23.93 | 6.08 |
| GSM8K | 2 | 0.71 | 45.52 | 3.24 |
| GSM8K | 4 | 0.61 | 88.29 | 1.57 |
| GSM8K | 8 | 0.29 | 170.77 | 0.98 |
| HumanEval+ | 1 | 0.35 | 20.95 | 5.76 |
| HumanEval+ | 2 | 0.27 | 42.94 | 2.87 |
| HumanEval+ | 4 | 0.09 | 83.37 | 1.56 |
| HumanEval+ | 8 | 0.06 | 176.32 | 1.20 |
| MBPP+ | 1 | 0.27 | 12.87 | 2.53 |
| MBPP+ | 2 | 0.21 | 25.87 | 1.27 |
| MBPP+ | 4 | 0.11 | 52.92 | 0.65 |
| MBPP+ | 8 | 0.04 | 93.35 | 0.35 |

The authoritative values and all additional metrics are in
`reference/sdar_1p7b_chat_b64/comparison/summary.csv`.

## Ordered task-ID fingerprints

The ordered `example_id` sequence is identical across all four fixed-k arms of
each benchmark. SHA-256 fingerprints of the newline-delimited IDs are:

| Benchmark | SHA-256 |
| --- | --- |
| GSM8K | `987fd1e556a9a4fce4ec097d90de2a11f3ab5acd0787c3fba2190351b6566294` |
| HumanEval+ | `d1ebc039377e8e813e79586f681222c15b7a8b1aeb83f42776e96e5475c16aa8` |
| MBPP+ | `dac76d8097602a13c559b198366790efc74062fbb23263c24539329e2fec996e` |

These fingerprints should be checked on the 8B outputs before comparing
scores or latency.

