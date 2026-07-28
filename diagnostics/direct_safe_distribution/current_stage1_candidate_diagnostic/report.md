# Direct-safe evaluation: legacy_global / validation

> 诊断限定：该旧 checkpoint 训练过 dataset0 全部地图，本结果不是 strict OOD。

| K | Safe@K | Valid Rate | Best-Cost@K | Safe count | Modes | Dedup valid |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.0% | 40.0% | 1.999 | 0.40 | 0.40 | 0.40 |
| 4 | 46.7% | 45.0% | 0.744 | 1.80 | 0.93 | 1.47 |
| 8 | 46.7% | 45.0% | 0.744 | 3.60 | 1.27 | 2.80 |
| 16 | 53.3% | 45.0% | 0.433 | 7.20 | 1.33 | 5.00 |
| 32 | 53.3% | 44.2% | 0.433 | 14.13 | 1.33 | 8.13 |

Best-Cost@K 是 privileged ESDF 离线 Oracle 上限，不属于在线方法。
