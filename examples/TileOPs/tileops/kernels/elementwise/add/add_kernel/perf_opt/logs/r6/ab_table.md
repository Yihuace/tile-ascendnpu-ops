| workload | dtype | base µs | final µs | Δ | ratio_h base | ratio_h final | ratio_m base | ratio_m final |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cnn-feat-broadcast | bfloat16 | 36.600 | 36.260 | -0.9% | 80.2 | 81.0 | 81.3 | 82.1 |
| cnn-feat-broadcast | float16 | 36.640 | 35.900 | -2.0% | 80.1 | 81.8 | 81.2 | 82.9 |
| cnn-feat-broadcast | float32 | 70.120 | 41.340 | -41.0% | 83.7 | 71.0 | 84.9 | 96.0 |
| hidden-state-prefill | bfloat16 | 26.600 | 25.160 | -5.4% | 72.1 | 76.2 | 73.1 | 77.2 |
| hidden-state-prefill | float16 | 24.000 | 24.000 | +0.0% | 79.9 | 79.9 | 81.0 | 81.0 |
| hidden-state-prefill | float32 | 46.620 | 47.500 | +1.9% | 82.3 | 80.7 | 83.4 | 81.8 |
| linear-bias | bfloat16 | 13.820 | 9.680 | -30.0% | 69.4 | 49.6 | 70.3 | 67.0 |
| linear-bias | float16 | 13.340 | 8.560 | -35.8% | 71.9 | 56.1 | 72.8 | 75.7 |
| linear-bias | float32 | 23.860 | 13.560 | -43.2% | 80.4 | 70.8 | 81.4 | 95.6 |
| llama-7b-ffn | bfloat16 | 32.800 | 32.740 | -0.2% | 78.6 | 78.7 | 79.6 | 79.8 |
| llama-7b-ffn | float16 | 31.960 | 32.780 | +2.6% | 80.6 | 78.6 | 81.7 | 79.7 |
| llama-7b-ffn | float32 | 62.800 | 62.900 | +0.2% | 82.1 | 81.9 | 83.2 | 83.0 |

total: base 419.160 us, final 370.380 us, Δ -11.64%
