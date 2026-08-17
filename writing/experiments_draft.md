# 实验

实验围绕 Path MeanFlow 的整体规划性能、结构化生成机制与特权物理适配展开。我们首先在环境互斥的未见地形上比较不同规划器的单次生成可行性与计算效率，随后分别分析 boundary-aligned trajectory representation、planning-support conditioning 和 expert route-prior learning（ERPL）的作用。在此基础上，我们检验 privileged multi-objective terrain adaptation（PMTA）引起的物理可行性与路线分布变化，并通过实物系统实验评价其闭环部署表现。

## A. 实验设置

**数据与任务。** 我们在由非结构化起伏地形构成的数据集上训练和评价所有学习方法。训练集、development set 与 unseen test set 分别包含 [训练环境数]、[开发环境数] 和 [测试环境数] 个环境，三个集合在环境层面互不重叠。每个环境采样 [任务数] 组起终点位姿，覆盖 [起终点距离范围]、[高程变化范围] 以及不同的可靠规划支撑形态。专家路线由 [专家规划器] 生成，仅用于 ERPL 的示范监督和路线评价。

部署时，Path MeanFlow 与所有公平比较方法仅接收部分地形观测 $X_{\mathrm{obs}}$、planning-support mask $m$ 及起终点位姿 $(S,G)$。完整地形 $X_{\mathrm{full}}$ 只用于 PMTA 的训练期稳定性监督和离线评价，不进入部署输入。依赖 $X_{\mathrm{full}}$ 的方法作为 full-information privileged reference 单独列示，不参与 partial-information methods 的直接排名。

**比较方法与公平性。** 实验包括 [搜索式基线]、[优化式基线]、[生成式学习基线]、ERPL route generator 以及完成 PMTA 的 Path MeanFlow。[搜索式基线] 与 [优化式基线] 使用各自原生的搜索或迭代求解过程，并在统一的 [超时阈值] 内运行；[生成式学习基线] 使用相同的训练环境、专家示范和部署观测。预先指定的主要 partial-information baseline 为 [主要基线名称]，其依据 [development-set 指标/预先规定的代表性规则] 在查看 unseen-test 结果之前确定。所有方法由同一 hard evaluator 评价，timeout、no solution 和 numerical failure 均计为不可行。各方法在 development set 上获得可比较的 [调参试验次数/计算预算]，所有学习方法均报告按 [固定训练终点/checkpoint 规则] 确定且未使用 test data 选择的 checkpoint。除端到端延迟外，我们还报告原生推理时间、迭代次数或 neural function evaluations（NFE）；一次前向推理是 Path MeanFlow 的部署特性，而非对外部方法推理过程的限制。

**实现细节。** Path MeanFlow 使用 [物理窗口宽度] m $\times$ [物理窗口高度] m 的局部物理窗口，栅格大小为 [栅格高度] $\times$ [栅格宽度]，分辨率为 [栅格分辨率] m/cell。输入由 $[\tilde n_x,\tilde n_y,\tilde n_z,m]$ 四个通道构成。Cubic B-spline 包含 26 个控制点，其中 4 个边界控制点解析固定，生成器预测其余自由控制点对应的 44 维坐标。网络采用 [层数] 层、宽度为 [通道数] 的 DiT-style 路径主干和 [编码器配置] 地形编码器，并使用 [attention heads 数] 个注意力头。训练路径以 [解码采样点数] 个点解码；hard evaluator 使用 [审计采样点数] 个点，并将支撑线段采样间距限制为不超过 $\rho/2$。

Planning-support mask 在车辆配置空间中构造：候选支撑按 [车辆半径] m 腐蚀，并检查起点、终点及其连通性。ERPL context 仅保留示范路径仍受支撑的 mask，PMTA context 则以 [中段阻塞采样概率] 允许示范路径中段被支撑缺口阻断。稳定性评价采用 $w_\psi=$ [数值]、[yaw bin 数] 个 yaw bins 和 hard threshold [稳定性阈值]；曲率上限设为 $\kappa_{\max}=$ [数值] $\mathrm{m}^{-1}$。每个 context 使用 [评价 source 数] 个固定评价 sources。

**训练协议。** ERPL 与 PMTA 分别训练 [ERPL 更新数] 和 [PMTA 更新数] 次 updates，PMTA 每个条件采样 $K=$ [每条件 source 数] 个 sources。ERPL 采用 $\lambda_{\mathrm{end}}=0.25$，以对应后文的 endpoint-supervision 消融；三项 PMTA objective scale 与 gradient-normalization 规则在查看 unseen-test 结果前冻结。模型选择遵循 [固定训练终点/checkpoint 规则]，各基线共享 [调参试验次数/计算预算]。优化器、学习率、gradient clipping、MeanFlow 时间采样、PMTA 几何参数、目标 reduction 和尺度定义等低层级数值设置见附录 [附录编号]。

**评价指标。** Feasible@1 表示一次独立规划调用生成可行路径的概率。令 $Y_{seqr}=I_FI_SI_\kappa I_{\mathrm{yaw}}I_{\mathrm{finite}}$ 表示训练种子 $s$、环境 $e$、context $q$ 和 source $r$ 对应路径的联合 hard feasibility，其中 $I_F$ 同时检查 planning support 与地图边界。采用每个 context 固定的 $R=$ [评价 source 数] 个 source 后，论文报告

$$
\widehat{\operatorname{Feasible@1}}
=\frac{1}{S}\sum_{s=1}^{S}\frac{1}{E}\sum_{e=1}^{E}
\frac{1}{Q_e}\sum_{q=1}^{Q_e}\frac{1}{R}\sum_{r=1}^{R}Y_{seqr}.
$$

因此，指标依次在 source、context、environment 和 training seed 上平均，并给予每个环境相同权重。每个 source 均对应一次独立 one-call trial；Path MeanFlow 在一次调用中采样一个 source 并执行一次生成器前向，不进行 best-of-$K$ selection。路线 ADE、离散 Fréchet distance、pairwise route distance 和 effective rank 使用相同的固定 source 数及当前每个 context 的单条 expert reference。路径长度仅在 feasible subset 上统计，并给出分母；Feasible@$K$ 仅作为补充诊断。

所有学习方法使用 [训练随机种子数] 个独立训练随机种子，主文 CI 同时反映模型训练随机性与测试环境、context 和 source 的采样变化。每次 paired crossed-hierarchical bootstrap 独立重采 training seeds 与 environments，并在被抽中的环境内依次重采 contexts 和 sources；相同的 environment/context/source 抽样在方法间保持配对，因此 seeds 与 environments 被视为交叉层级而非嵌套层级。确定性外部基线在每个 environment/context 上仅评价一次；计算其与学习方法的差值时，同一基线结果可与每个被抽中的训练 seed 配对，但不作为新的独立 seed 观测重复计权。若仅有一个训练 seed，相应区间明确记为 conditional-on-trained-model CI，而不用于概括训练随机性。本文采用 [Bootstrap 次数] 次重采样并报告 $95\%$ CI。延迟在 [硬件型号] 上以 batch size 1 测量，经过 [预热次数] 次预热后统计 [调用次数] 次调用的 p50 与 p95 时间。

## B. 未见地形上的总体性能

表 [表号] 汇总了未见地形上的总体结果。Path MeanFlow 与预先指定的主要 partial-information baseline [主要基线名称] 的 Feasible@1 分别为 [数值] 和 [数值]，带符号差值为 [带符号数值] pp（$95\%$ CI [区间]），对应 [改善/无显著变化/回归及其解释]。Path MeanFlow 的 planning-support、stability、curvature 和 yaw pass rate 分别为 [数值]、[数值]、[数值] 和 [数值]；相对 [主要基线名称] 的差值依次为 [带符号数值]、[带符号数值]、[带符号数值] 和 [带符号数值] pp，对应 CI 依次为 [区间]、[区间]、[区间] 和 [区间]，表明 [分量变化及其解释]。Full-information privileged reference 的 Feasible@1 为 [数值]，其与 Path MeanFlow 的差值为 [带符号数值] pp（CI [区间]），对应 [额外完整地形信息下的差异解释]。

在可行路径子集上，Path MeanFlow 与 [主要基线名称] 的平均路径长度分别为 [数值] 和 [数值] m，带符号差值为 [带符号数值] m（CI [区间]），对应 [改善/无显著变化/回归及其解释]。Path MeanFlow 的端到端 p50/p95 延迟为 [数值]/[数值] ms；采用原生迭代推理的 [主要基线名称] 为 [数值]/[数值] ms，并使用 [NFE/迭代次数]。图 [图号] 展示了不同地形与 planning-support 形态下的代表性路径，其中 [定性差异]；失败案例主要呈现 [失败类型及其物理指标表现]。

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

### 1) Boundary-aligned trajectory representation

我们首先验证解析边界构造的数值正确性。在 [样本数] 个随机任务和每任务 [评价 source 数] 个 source 上，解码路径的起点位置、终点位置及一阶方向残差分别为 [数值]、[数值] 和 [数值]，对应 [数值实现误差解释]。该检查仅确认固定边界解码器的数值残差。

随后，在使用同一固定边界解码器、网络容量与训练预算的条件下，比较 unwhitened 与 whitened boundary-aligned coordinates。该可逆仿射变换不改变自由轨迹的表示容量或解析边界满足，因而实验只考察均值/协方差整形与 whitening 对优化条件的影响。本文将 endpoint route ADE 定义为部署生成端 $(t,r)=(1,0)$ 输出与 expert reference 的路线误差，而非几何端点误差。固定 [更新数] 次更新后，两者的 endpoint route ADE 分别为 [数值] 和 [数值] m，差值为 [带符号数值] m（CI [区间]）；达到 development loss [阈值] 所需更新数分别为 [数值] 和 [数值]，差值为 [带符号数值] updates（CI [区间]），未达到该阈值的运行记为 $>$[训练预算] updates。两者的梯度/条件数诊断分别为 [数值] 和 [数值]，其统计差异为 [带符号数值]（CI [区间]）。图 [图号] 给出完整优化曲线，结果对应 [改善/无显著变化/回归及其解释]。

### 2) Planning-support conditioning

为分离显式 mask channel 的作用，我们采用结构和参数量完全一致的两组网络。Constant-mask 变体保留相同的四通道输入与 masked-normal fill，仅将第四通道设为常量；explicit-mask 变体接收实际 planning-support mask。按 [支撑完整/局部缺失/参考路线受阻等实际支撑分层] 汇总时，两者的 Feasible@1 分别为 [数值] 和 [数值]，差值为 [带符号数值] pp（CI [区间]）；planning-support pass rate 分别为 [数值] 和 [数值]，差值为 [带符号数值] pp（CI [区间]）。Minimum support clearance 分别为 [数值] 和 [数值] m，差值为 [带符号数值] m（CI [区间]），整体结果对应 [改善/无显著变化/回归及其解释]。图 [图号] 展示参考路线被支撑缺口阻断时两种变体的路线响应，其中 [定性差异及其解释]。

### 3) Expert route-prior learning

ERPL 消融比较 $\lambda_{\mathrm{end}}=0$ 与 $\lambda_{\mathrm{end}}=0.25$。两者保留相同的 endpoint atom、MeanFlow 目标、时间采样和训练预算，因此 $\lambda_{\mathrm{end}}=0$ 仅移除部署 endpoint 上的显式 MSE，而不移除 endpoint 时间对。两种设置的 endpoint route ADE 分别为 [数值] 和 [数值] m，差值为 [带符号数值] m（CI [区间]）；Fréchet distance 分别为 [数值] 和 [数值] m，差值为 [带符号数值] m（CI [区间]）；Feasible@1 分别为 [数值] 和 [数值]，差值为 [带符号数值] pp（CI [区间]）。Pairwise route distance/effective rank 分别为 [数值]/[数值] 和 [数值]/[数值]，带符号差值为 [带符号数值]/[带符号数值]（CI [区间]/[区间]），对应 [改善/无显著变化/回归及其解释]。

**表 [表号]. 结构化路径生成器消融。$\Delta$ 始终按第二个变体减第一个变体计算。**

| Component | Compared variants (first / second) | Feasible@1 first / second / $\Delta$ | Route ADE first / second / $\Delta$ (m) | Diversity first / second |
|---|---|---:|---:|---:|
| Coordinates | Unwhitened / Whitened boundary-aligned | [数值] | [数值] | [数值] |
| Support condition | Constant / Explicit mask | [数值] | [数值] | [数值] |
| ERPL endpoint | $\lambda_{\mathrm{end}}=0$ / $0.25$ | [数值] | [数值] | [数值] |

## D. Privileged multi-objective terrain adaptation

### 1) Physical feasibility and route-distribution trade-off

我们比较 ERPL route generator 与经 PMTA 适配后的 Path MeanFlow。两者的 Feasible@1 分别为 [数值] 和 [数值]，差值为 [带符号数值] pp（CI [区间]），对应 [改善/无显著变化/回归及其解释]。Planning-support、stability 和 curvature pass rate 的 ERPL/PMTA 数值分别为 [数值]/[数值]、[数值]/[数值] 和 [数值]/[数值]，带符号差值为 [带符号数值]、[带符号数值] 和 [带符号数值] pp（CI [区间]、[区间] 和 [区间]）。单路径代价 $C_F$、$C_S$ 和 $C_\kappa$ 的 ERPL/PMTA 数值分别为 [数值]/[数值]、[数值]/[数值] 和 [数值]/[数值]，差值为 [带符号数值]、[带符号数值] 和 [带符号数值]（CI [区间]、[区间] 和 [区间]），作为 hard feasibility 的次级解释。

所有 PMTA 转移均以相同 source 下的 ERPL 输出为参照。Invalid-to-feasible rate 的分母为 ERPL 不可行的配对调用数，其中 PMTA 可行的比例为 [数值]（CI [区间]）；feasible-to-invalid rate 的分母为 ERPL 可行的配对调用数，其中 PMTA 不可行的比例为 [数值]（CI [区间]）。这两个 eligible denominator 在每次 crossed-hierarchical bootstrap 重采样后依据该次抽样中的 ERPL 状态重新计算。适配后相对 ERPL 的 source-paired route drift 为 [数值] m（CI [区间]），expert-reference Fréchet distance、pairwise route distance 和 effective rank 的带符号变化分别为 [带符号数值] m、[带符号数值] m 和 [带符号数值]（CI [区间]、[区间] 和 [区间]）。这些结果共同对应 [物理可行性与路线分布的改善/无显著变化/回归及权衡解释]，不预设 PMTA 保持路线分布。

**表 [表号]. PMTA 的物理可行性与路线分布权衡。转移率均以 ERPL 的对应状态为分母。**

| Variant | Feasible@1 $\uparrow$ | F / S / $\kappa$ pass $\uparrow$ | Invalid $\rightarrow$ feasible $\uparrow$ | Feasible $\rightarrow$ invalid $\downarrow$ | Route drift (m) $\downarrow$ |
|---|---:|---:|---:|---:|---:|
| ERPL | [数值] | [数值] | N/A | N/A | N/A |
| ERPL$\rightarrow$PMTA | [数值] | [数值] | [数值] | [数值] | [数值] |

### 2) Staged learning

为检验先学习路线结构、再吸收特权物理监督的顺序设计，我们比较 sequential ERPL$\rightarrow$PMTA 与 interleaved ERPL/PMTA。两种方案采用相同数量的 ERPL updates、privileged updates、训练样本和 source。其 Feasible@1 分别为 [数值] 和 [数值]，sequential 减 interleaved 的差值为 [带符号数值] pp（CI [区间]）；route ADE 分别为 [数值] 和 [数值] m，差值为 [带符号数值] m（CI [区间]）；Fréchet distance 分别为 [数值] 和 [数值] m，差值为 [带符号数值] m（CI [区间]）；pairwise diversity/effective rank 分别为 [数值]/[数值] 和 [数值]/[数值]，差值为 [带符号数值]/[带符号数值]（CI [区间]/[区间]）。整体结果对应 [改善/无显著变化/回归及其关于监督顺序的解释]。PMTA from random initialization 的 Feasible@1 和 route ADE 分别为 [数值] 和 [数值] m，作为专家路线初始化的附加对照。

**表 [表号]. ERPL 与 PMTA 的学习顺序。$\Delta$ 按 sequential 减 interleaved 计算。**

| Metric | Sequential ERPL$\rightarrow$PMTA | Interleaved ERPL/PMTA | $\Delta$ | PMTA random init. |
|---|---:|---:|---:|---:|
| Feasible@1 $\uparrow$ | [数值] | [数值] | [带符号数值] | [数值] |
| Route ADE / Fréchet (m) $\downarrow$ | [数值] | [数值] | [带符号数值] | [数值] |
| Pairwise diversity / Eff. rank | [数值] | [数值] | [带符号数值] | [数值/N/A] |

### 3) Multi-objective gradient coordination

该实验比较 tuned fixed scalarization 与 PMTA 中采用的 MGDA gradient coordination。两者从相同 ERPL checkpoint 初始化，均使用 plain SGD、相同学习率、global gradient clipping、batch、source 顺序和参数更新次数，仅改变三项目标梯度的组合方式。Fixed scalarization 的权重通过 development set 上的 [权重搜索空间与预算] 确定为 $(w_F,w_S,w_\kappa)=$ [数值]。

在 update-matched 设置下，fixed scalarization 与 MGDA coordination 的 Feasible@1 分别为 [数值] 和 [数值]，差值为 [带符号数值] pp（CI [区间]），对应 [改善/无显著变化/回归及其解释]；三项 hard pass rate 的带符号差值为 [带符号数值]、[带符号数值] 和 [带符号数值] pp（CI [区间]、[区间] 和 [区间]）。两者实际 backward 次数分别为 [数值] 和 [数值]，训练时间分别为 [数值] 和 [数值]。在 [backward-matched/wall-clock-matched] 设置下，两者使用 [共同预算定义]，Feasible@1 分别为 [数值] 和 [数值]，差值为 [带符号数值] pp（CI [区间]）。因此，本文的计算公平性结论限定为 [相同更新次数下的结果及 MGDA 的额外训练代价/在共同训练计算预算下仍观察到的差异]。

梯度分析在训练过程中预先抽取的 [minibatch 数] 个 minibatches 上进行。$g_F$ 与 $g_S$、$g_F$ 与 $g_\kappa$、$g_S$ 与 $g_\kappa$ 的负 cosine 比例分别为 [数值]、[数值] 和 [数值]；$g_a^\top g^\star\geq0$ 对 $a\in\{F,S,\kappa\}$ 的比例分别为 [数值]、[数值] 和 [数值]；MGDA active-set frequency 及 $\omega_F^\star/\omega_S^\star/\omega_\kappa^\star$ 的分布为 [数值]；一次有限步更新后的 $(\Delta C_F,\Delta C_S,\Delta C_\kappa)$ 为 [数值]。这些统计仅描述所观察训练点与 minibatches 上的梯度几何和有限步变化，不构成对普遍共同下降、Pareto optimality 或收敛性的主张。

**表 [表号]. 固定标量化与多目标梯度协调。**

| Budget view | Variant | Feasible@1 $\uparrow$ | F / S / $\kappa$ pass $\uparrow$ | Actual backward $\downarrow$ | Training time $\downarrow$ |
|---|---|---:|---:|---:|---:|
| Update matched | Tuned fixed scalarization + SGD | [数值] | [数值] | [数值] | [数值] |
| Update matched | MGDA-coordinated PMTA + SGD | [数值] | [数值] | [数值] | [数值] |
| [Backward/wall-clock] matched | Tuned fixed scalarization + SGD | [数值] | [数值] | [数值] | [数值] |
| [Backward/wall-clock] matched | MGDA-coordinated PMTA + SGD | [数值] | [数值] | [数值] | [数值] |

## E. 实物系统实验

我们在 [机器人平台] 上部署 Path MeanFlow。系统由 [上游传感器/建图模块]、[计算平台与 GPU]、全局规划器及固定的 [局部规划器/跟踪器] 构成，地图以 [频率] 更新。实验包含 [场景数] 类起伏地形和每类 [任务数] 个预先确定的 start-goal tasks，覆盖 [普通起伏地形]、[稳定性关键地形] 和 [规划支撑变化场景]。实物实验的主要比较基线 [实物主要基线名称] 依据 [development 场景/预先规定的代表性规则] 在查看实物测试结果前确定。

实物评价包含两套相互区分的数据。Planned-path one-call evaluation 在 [冻结快照数] 个 matched frozen observation snapshots 上运行；每个 snapshot、task 和 method 只计首次 one-call 输出，不进行候选选择，因而实物 Feasible@1 的分母为全部预先确定的 frozen snapshots。若另行统计在线 replanning invocations，其 Feasible@1 以 [独立 invocation 分母] 单列，不与 frozen-snapshot 指标合并。Executed-trajectory evaluation 则由每种方法独立在线运行，各方法因自身执行与 replanning 过程产生各自的观测序列；它们共享机器人平台、局部控制栈、start-goal tasks、终止条件和安全协议，而不假定共享相同在线观测。

在 frozen-snapshot planned-path evaluation 中，Path MeanFlow 与 [实物主要基线名称] 的 Feasible@1 分别为 [数值] 和 [数值]，差值为 [带符号数值] pp（CI [区间]）；minimum support clearance、minimum stability margin、maximum curvature 和 feasible-path length 的 Path MeanFlow/[实物主要基线名称] 数值分别为 [数值]/[数值] m、[数值]/[数值]、[数值]/[数值] $\mathrm{m}^{-1}$ 和 [数值]/[数值] m，其带符号差值与 CI 分别为 [带符号数值及区间]，对应 [改善/无显著变化/回归及其解释]。

在 executed-trajectory evaluation 中，Path MeanFlow 与 [实物主要基线名称] 的任务成功率分别为 [数值] 和 [数值]，方法间差值为 [带符号数值] pp（CI [区间]）。仅当 planned-path 与 executed-trajectory evaluation 使用一一对应的同一任务单位时，才报告同一方法的 planned Feasible@1 与 task success 之差 [带符号数值] pp（CI [区间]）；否则两项比例分别报告，不计算跨数据集差值。Intervention 采用 [每任务干预次数/发生过干预的任务比例] 定义，其余 tracking RMSE、最大 roll/pitch、replanning 次数和 p95 规划延迟的两方法数值、带符号差值及 CI 分别为 [数值及区间]，整体对应 [改善/无显著变化/回归及其系统解释]。

图 [图号] 展示机器人在 [代表性场景] 中的观测支撑、生成路线和实际执行轨迹，其中 [定性差异]。失败分层显示 planned-path feasibility 与 executed success 的差异同 [跟踪误差/地图更新延迟/局部控制行为] 相关；除非另行进行 [因果分析名称]，本文不将该关联解释为因果来源。该分析将全局规划器输出与下游执行行为分开评价，并将结论限定于当前平台、地形范围和安全协议。

**表 [表号]（双栏）. 实物系统评价。(a) Frozen-snapshot planned-path performance；(b) independent online executed-trajectory performance。**

**(a) Frozen-snapshot planned-path performance**

| Method | Feasible@1 $\uparrow$ | Min. support clearance $\uparrow$ | Min. stability margin $\uparrow$ | Max. curvature $\downarrow$ | Path length (m) $\downarrow$ |
|---|---:|---:|---:|---:|---:|
| [实物主要基线名称] | [数值] | [数值] | [数值] | [数值] | [数值] |
| ERPL | [数值] | [数值] | [数值] | [数值] | [数值] |
| **Path MeanFlow** | **[数值]** | **[数值]** | **[数值]** | **[数值]** | **[数值]** |

**(b) Independent online executed-trajectory performance**

| Method | Task success $\uparrow$ | Intervention [定义] $\downarrow$ | Tracking RMSE (m) $\downarrow$ | Max roll / pitch (deg) $\downarrow$ | Replans $\downarrow$ | p95 time (ms) $\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|
| [实物主要基线名称] | [数值] | [数值] | [数值] | [数值] | [数值] | [数值] |
| ERPL | [数值] | [数值] | [数值] | [数值] | [数值] | [数值] |
| **Path MeanFlow** | **[数值]** | **[数值]** | **[数值]** | **[数值]** | **[数值]** | **[数值]** |
