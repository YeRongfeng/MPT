# Stage-2 失败来源分解

## 1. 原 teacher：上限与模仿失败同时存在

固定 N=60 的 10 个 held-out 条件、40 个候选中：

- teacher-safe 条件为 2/10；
- 严格安全 teacher 候选为 6/40；
- 在这两个 teacher-safe 条件上，B 和 D 的学生 `safe@4` 均为 0。

| Student | Teacher-safe subset Recovery | Teacher error | 最大控制点误差中位数 | safe@4 |
|---|---:|---:|---:|---:|
| B cost-only | 0.155 | 1.277 | 1.473 m | 0% |
| D coupling | 0.332 | 0.960 | 0.848 m | 0% |

D 对 6 条严格安全 teacher 均未复现安全性。学生轨迹中位新增 5.5 个负 ESDF 点；这些点集中出现在轨迹索引 11–15、66–68 或 94–99，最小 ESDF 约为 `-0.20` 至 `-0.68 m`。这不是“控制点误差很小但 strict-safe 极敏感”的情形，而是 teacher 映射尚未被可靠学习。

## 2. 连续 cost 与 strict-safe

原 teacher 的 safe cost 对 strict-safe 的候选级 AUROC 为 0.941：

- 安全 teacher cost 中位数：0.286；
- 不安全 teacher cost 中位数：1.178；
- cost 下降量与危险点减少量的 Pearson/Spearman 相关为 0.536/0.514。

因此 cost 并非整体失效。但 34 条不安全 teacher 中有 3 条 cost 低于最差安全 teacher；其中一条 cost 仅 0.256，却仍有 `-0.212 m` 的最大违规。这验证了 tail mean 可能允许少量残余危险点。

## 3. 安全优先 teacher

实现了：

- 平滑最大 ESDF 违规作为第一优先目标；
- 质量、偏移、长度、曲率、端点姿态和 jerk 作为第二优先级；
- 1/2/3 m 的自适应位移信赖域；
- 1.10/1.17/1.25 的自适应长度信赖域；
- 安全候选停止扩大信赖域；
- 每个 proposal 两个局部起点；
- 跨阶段保留最佳严格安全解；
- 不允许跨越原 proposal 的拓扑侧。

结果：

| Teacher | Train safe@4 | Held-out safe@4 | Held-out 安全候选率 |
|---|---:|---:|---:|
| 原 tail-risk teacher | 20%（N=60 train 汇总） | 20% | 15% |
| 安全优先 teacher | 40% | 60% | 47.5% |

安全优先 teacher 的 held-out 模式类别保持率为 82.5%，左右侧实际跨越率为 0；最大控制点位移中位数为 2.243 m，路径长度比中位数为 1.152。它明显提高了可用于判断学生模仿能力的安全上限，但仍有 4/10 条件在 K=4 下无严格安全解。

## 4. 强 teacher 下的 Stage-2 接口对照

使用同一安全优先 teacher、相同 60 张训练地图、20 epochs、batch 16、paired targets 和 pretrained DiT/map encoder：

| Interface | Held-out Recovery | Teacher error | Mode keep | safe@4 | Teacher-safe subset safe@4 |
|---|---:|---:|---:|---:|---:|
| `(M,z) → R*` | 0.001 | 1.022 | 87.5% | 0% | 0% |
| `(M,R0) → R0+ΔR` | 0.049 | 1.014 | 62.5% | 10% | 16.7% |

显式 proposal correction 在一个 held-out 条件上产生了严格安全候选，而当前 D 没有；但它只复现 6 个 teacher-safe 条件中的 1 个，teacher error 仍约为 1，并损失更多模式身份。这个结果是弱正信号，不足以证明复合映射是主要瓶颈，也不足以把 Refiner 作为最终方法。

## 5. 当前判定

可以确认：

1. 原 teacher 上限不足；
2. 学生模仿和跨地图泛化是独立问题；
3. 原 cost 排序总体有效，但 tail 聚合会漏掉少数严重或临界违规；
4. 安全优先 teacher 能把 held-out `safe@4` 从 20% 提高到 60%；
5. 强 teacher 本身不会让当前 D 自动成功；
6. 显式 `R0→ΔR` 仅带来有限改善，尚未建立方法结论。

因此两条论文分支目前都没有闭合。下一轮最小实验应围绕学生目标进行：在同一强 teacher 和同一显式 correction 接口上，对比普通 residual Huber 与安全优先的 privileged imitation loss，判断学生是否需要直接惩罚最大 ESDF 违规，而不是继续修改地图分辨率或扩大网络。
