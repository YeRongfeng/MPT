# 实验（简短版）

实验依次评价 Path MeanFlow 在未见地形上的总体性能、结构化路径生成器的关键设计、PMTA 的物理适配作用，以及真实系统中的规划与执行表现。

## A. 实验设置

训练集、development set 和 unseen test set 在环境层面互斥。公平比较的 partial-information methods 在部署时仅使用部分地形观测、planning-support mask 和起终点位姿；完整地形只用于 PMTA 的训练期稳定性监督、离线评价及 privileged reference。Feasible@1 衡量一次独立规划调用的联合可行率，Path MeanFlow 每次仅采样一个 source 并执行一次前向，不进行 best-of-$K$ 选择。核心设置包括窗口与栅格、轨迹表示、网络、mask/context 构造、物理阈值、训练预算和模型选择，低层级数值超参数见附录。

## B. 未见地形上的总体性能

在环境互斥的 unseen test set 上，将 Path MeanFlow 与搜索式、优化式、生成式学习基线及 full-information privileged reference 比较。主要指标为 Feasible@1，并同时报告各物理分量 pass rate、可行路径长度、p95 延迟和 NFE 或迭代次数。

**表 [表号]（双栏）. 未见地形上的总体规划性能。Priv. 表示部署时读取完整地形的 privileged reference；Length 仅在可行路径上统计。**

| Method | Input | Feasible@1 $\uparrow$ | F / S / $\kappa$ / yaw pass $\uparrow$ | Length (m) $\downarrow$ | p95 time (ms) $\downarrow$ | NFE / Iter. $\downarrow$ |
|---|---|---:|---:|---:|---:|---:|
| [搜索式基线] | Partial | [数值] | [数值] | [数值] | [数值] | [数值] |
| [优化式基线] | Partial | [数值] | [数值] | [数值] | [数值] | [数值] |
| [生成式学习基线] | Partial | [数值] | [数值] | [数值] | [数值] | [数值] |
| ERPL | Partial | [数值] | [数值] | [数值] | [数值] | 1 |
| **Path MeanFlow** | **Partial** | **[数值]** | **[数值]** | **[数值]** | **[数值]** | **1** |
| [Full-information reference] | Priv. | [数值] | [数值] | [数值] | [数值] | [数值] |

## C. 结构化路径生成器分析

### 1. Boundary-aligned trajectory representation

检验解析解码器的边界残差，并在相同解码器、网络容量和训练预算下比较 unwhitened 与 whitened boundary-aligned coordinates，评价坐标整形对路线误差和优化过程的影响。

### 2. Planning-support conditioning

比较 constant-mask 与 explicit-mask，检验显式 mask channel 是否帮助生成路径响应不规则规划支撑及支撑缺口；两组网络保持相同结构、参数量和 masked-normal fill。

### 3. Expert route-prior learning

比较 $\lambda_{\mathrm{end}}=0$ 与 $0.25$ 的 ERPL，检验部署 endpoint 监督对路线拟合、单次可行性和路线分布的影响。

**表 [表号]. 结构化路径生成器消融。$\Delta$ 始终按第二个变体减第一个变体计算。**

| Component | Compared variants (first / second) | Feasible@1 first / second / $\Delta$ | Route ADE first / second / $\Delta$ (m) | Diversity first / second |
|---|---|---:|---:|---:|
| Coordinates | Unwhitened / Whitened boundary-aligned | [数值] | [数值] | [数值] |
| Support condition | Constant / Explicit mask | [数值] | [数值] | [数值] |
| ERPL endpoint | $\lambda_{\mathrm{end}}=0$ / $0.25$ | [数值] | [数值] | [数值] |

## D. Privileged multi-objective terrain adaptation

### 1. Physical feasibility and route-distribution trade-off

以相同 source 下的 ERPL 输出为参照，比较 PMTA 前后的联合可行性、分量可行性、不可行与可行状态转移，以及路线分布变化。

**表 [表号]. PMTA 的物理可行性与路线分布权衡。转移率均以 ERPL 的对应状态为分母。**

| Variant | Feasible@1 $\uparrow$ | F / S / $\kappa$ pass $\uparrow$ | Invalid $\rightarrow$ feasible $\uparrow$ | Feasible $\rightarrow$ invalid $\downarrow$ | Route drift (m) $\downarrow$ |
|---|---:|---:|---:|---:|---:|
| ERPL | [数值] | [数值] | N/A | N/A | N/A |
| ERPL$\rightarrow$PMTA | [数值] | [数值] | [数值] | [数值] | [数值] |

### 2. Staged learning

在匹配 ERPL updates、privileged updates、训练样本和 source 预算的条件下，比较 sequential ERPL$\rightarrow$PMTA 与 interleaved ERPL/PMTA。另将 PMTA from random initialization 作为专家路线初始化的附加对照。

**表 [表号]. ERPL 与 PMTA 的学习顺序。$\Delta$ 按 sequential 减 interleaved 计算。**

| Metric | Sequential ERPL$\rightarrow$PMTA | Interleaved ERPL/PMTA | $\Delta$ | PMTA random init. |
|---|---:|---:|---:|---:|
| Feasible@1 $\uparrow$ | [数值] | [数值] | [带符号数值] | [数值] |
| Route ADE / Fréchet (m) $\downarrow$ | [数值] | [数值] | [带符号数值] | [数值] |
| Pairwise diversity / Eff. rank | [数值] | [数值] | [带符号数值] | [数值/N/A] |

### 3. Multi-objective gradient coordination

从相同 ERPL checkpoint 出发，比较 tuned fixed scalarization 与 MGDA-coordinated PMTA，并在 update-matched 和 backward-或 wall-clock-matched 预算下报告可行性、实际 backward 次数与训练时间；梯度统计仅作为训练点上的诊断。

**表 [表号]. 固定标量化与多目标梯度协调。**

| Budget view | Variant | Feasible@1 $\uparrow$ | F / S / $\kappa$ pass $\uparrow$ | Actual backward $\downarrow$ | Training time $\downarrow$ |
|---|---|---:|---:|---:|---:|
| Update matched | Tuned fixed scalarization + SGD | [数值] | [数值] | [数值] | [数值] |
| Update matched | MGDA-coordinated PMTA + SGD | [数值] | [数值] | [数值] | [数值] |
| [Backward/wall-clock] matched | Tuned fixed scalarization + SGD | [数值] | [数值] | [数值] | [数值] |
| [Backward/wall-clock] matched | MGDA-coordinated PMTA + SGD | [数值] | [数值] | [数值] | [数值] |

## E. 实物部署展示

按照会议建议，方法间的公平比较和消融集中在可重复的仿真环境中；实物部分只部署最终的 Path MeanFlow，在若干具有代表性的崎岖地形上展示从当前观测生成参考路径并由车辆执行的完整过程。该实验用于验证方法在所测试场景中的系统集成与任务执行能力，不承担相对基线的优越性结论。除视频与典型轨迹可视化外，按场景报告预先定义的任务次数、成功率、人工干预、车辆最大横滚/俯仰角和规划延迟。

**表 [表号]. Path MeanFlow 的实物部署结果。每一行对应预先确定的一类地形场景。**

| Terrain scenario | Trials | Task success $\uparrow$ | Interventions $\downarrow$ | Max. roll / pitch (deg) $\downarrow$ | p95 planning time (ms) $\downarrow$ |
|---|---:|---:|---:|---:|---:|
| [场景 A] | [数值] | [数值] | [数值] | [数值] | [数值] |
| [场景 B] | [数值] | [数值] | [数值] | [数值] | [数值] |
| [场景 C] | [数值] | [数值] | [数值] | [数值] | [数值] |
| Overall | [数值] | [数值] | [数值] | [数值] | [数值] |
