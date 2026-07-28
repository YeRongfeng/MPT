# Teacher safety / student imitation 分层分析

- Held-out：10 个条件，40 个候选。
- Teacher-safe / unsafe 条件：2/8。
- 严格安全 teacher 候选：6。
- Strict-safe：100 个 yaw-ESDF 采样点全部非负，且路径长度比不超过 1.17。

## Teacher-safe 条件上的学生复现

| Student | safe@4 | Recovery 中位数 | Teacher error 中位数 | 最大控制点误差中位数 (m) | 模式保持 |
|---|---:|---:|---:|---:|---:|
| B_cost_only | 0.0% | 0.155 | 1.277 | 1.473 | 50.0% |
| D_induced_coupling | 0.0% | 0.332 | 0.960 | 0.848 | 62.5% |

## Cost 与 strict-safe

| Trajectory | 安全候选数 | Cost AUROC | 安全 cost 中位数 | 不安全 cost 中位数 | 不安全最大违规中位数 (m) |
|---|---:|---:|---:|---:|---:|
| stage1 | 0 | nan | nan | 4.930 | 0.787 |
| teacher | 6 | 0.941 | 0.286 | 1.178 | 0.529 |
| B_cost_only | 0 | nan | nan | 3.709 | 0.811 |
| D_induced_coupling | 0 | nan | nan | 4.375 | 0.611 |

Teacher cost 下降量与危险点减少量：Pearson `0.536`，Spearman `0.514`。

逐候选重新变得不安全的轨迹点索引、最大违规和控制点偏差见 `per_candidate.csv`。
