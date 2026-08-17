# Stage 2：约束后验传输调研与验证路线

状态：**研究备忘录，不替代当前正式研究协议**  
日期：**2026-08-02**  
范围：Stage 2 训练机制、分布保持、特权信息使用方式及其对网络结构的影响。

## 1. 当前结论

目前最有根据的 Stage 2 问题表述不是：

\[
\text{最小化生成器输出的平均 privileged cost},
\]

也不是：

\[
\text{模仿一个确定性数值优化器的更新过程}.
\]

更合适的待验证表述是：

\[
\boxed{
\text{以 Stage 1 条件路径分布为先验，学习其受联合安全约束倾斜后的可部署后验分布。}
}
\]

这把 Stage 2 定义为 **amortized constrained posterior transport**。它同时要求：

1. 提高单次采样的 strict-safe 概率；
2. 保留 Stage 1 已经学到的走廊和模式结构；
3. 在当前支持不足时允许同模式的局部支持扩充；
4. 训练期可使用完整地形，推理期只依赖部署观测；
5. 推理仍是固定且很少的网络前向，不保留在线数值优化器。

这里仍然只是研究假设。文献提供的是机制依据，不是本任务上的有效性证明。

## 2. 为什么直接 cost 反传确实存在结构性风险

直接反传优化：

\[
\min_\theta
\mathbb E_{z\sim p(z)}
\left[V_\xi(G_\theta(o,z))\right].
\]

如果暂时忽略参数化，把它写成对输出分布的优化：

\[
\min_q \mathbb E_{y\sim q}[V_\xi(y)],
\]

该目标对 \(q\) 是线性的。没有熵、KL、Wasserstein 或模式保持约束时，把越来越多
概率质量集中到少数最低 cost 输出完全符合目标。因此：

\[
\boxed{
\text{噪声到输出的敏感度下降不是必然发生，}
\text{但它是 direct-cost 目标明确允许甚至可能鼓励的解。}
}
\]

[DRaFT](https://deepmind.google/research/publications/51081/) 证明了可微奖励可以通过完整
或截断采样链直接反传；它说明该路线可行，但不提供我们需要的条件模式保持机制。
[ORW-CFM-W2](https://arxiv.org/abs/2502.06061) 更直接地分析了 flow matching 的奖励
微调：无分布正则的在线指数重加权会趋向最大回报的退化分布，并用可计算的
Wasserstein-2 上界控制参考模型与微调模型的距离。

本项目已有结果与这个风险一致，但还没有直接证明“噪声被忽略”：

- [10-step direct-cost](../../tests/audits/direct_privileged_cost_no_anchor_step10_20260729/REPORT.md)
  在小漂移下提高 held-out Safe@1；
- [100-step direct-cost](../../tests/audits/direct_privileged_cost_no_anchor_20260729/REPORT.md)
  不再提高 held-out Safe@1，且 held-out cost 与路径漂移变差；
- [balanced direct-cost](../../tests/audits/a0_balanced_direct_cost/REPORT.md) 的正增益没有通过
  环境级准入；
- 这些旧实验只报告了样本距离、模式数或有效秩，没有直接测量
  \(\partial G(o,z)/\partial z\) 的物理谱，所以不能事后把它们解释成已经确认的
  noise collapse；它们也早于当前 corrected `dataset1` 和重训 Stage 1，不能作为当前
  checkpoint 的正式前后对照。

因此，噪声敏感度需要成为下一轮独立审计量，而不是继续用平均 pair distance 代替。

## 3. 相关工作中的四种主要解法

| 路线 | 代表工作 | 核心机制 | 对本项目的适用性 |
| --- | --- | --- | --- |
| 可微奖励反传 | [DRaFT](https://deepmind.google/research/publications/51081/) | 通过采样器反传 reward，可截断反传步数 | 适合作为 direct-cost 控制组；自身不解决模式保持 |
| 分布约束的生成器微调 | [ORW-CFM-W2](https://arxiv.org/abs/2502.06061)、[DPOK](https://arxiv.org/abs/2305.16381) | 奖励重加权或策略梯度，同时用 W2/KL 约束参考分布 | 最接近“微调整个 Path MeanFlow”的正式对照 |
| 约束后验采样 | [Motion Planning Diffusion](https://arxiv.org/abs/2412.19948)、[SafeDiffuser](https://openreview.net/pdf?id=ig2wk7kK9J)、[Constrained Diffusers](https://openreview.net/forum?id=tahkGZjjWA) | 在生成迭代中融合 cost gradient、投影、primal-dual 或 barrier/CBF | 能制造高质量联合约束 teacher；部署时需要 cost 和多步采样，不宜直接作为后端 |
| 后验摊销与噪声空间控制 | [Amortized Posterior Sampling](https://arxiv.org/abs/2407.17907)、[Outsourced Diffusion Sampling](https://openreview.net/forum?id=94c9hu6Fsv) | 冻结先验，用条件 flow 一步采后验，或在生成器噪声空间训练独立 sampler | 最符合“冻结 Stage 1、保持随机性、少步部署”的目标 |

[DiffusionSeeder](https://proceedings.mlr.press/v270/huang25f.html) 则采用“多模态生成 seed +
少量数值优化”的系统分工。它支持保留前端生成分布和后端可行化的层级结构，但其优化器
仍在推理时运行；对本项目更适合作为 teacher 生成思路，而不是最终部署形式。

### 3.1 不能直接照搬 DDPO/DPOK

[DDPO](https://proceedings.iclr.cc/paper_files/paper/2024/hash/14f75513f0f1ca01de1e826b52e6b840-Abstract-Conference.html)
把多步随机去噪写成 MDP，并利用每步可计算的策略概率做 policy gradient。当前
Path MeanFlow 是从 44D source 到路径的一次确定性映射，没有现成的多步随机转移概率。
为了套用 DDPO 而重构生成过程，会同时改变 Stage 1、推理预算和 Stage 2，实验归因过差。

### 3.2 不能把约束 diffusion 直接当部署方法

Motion Planning Diffusion 在 B 样条控制点空间内，把先验和 cost gradient 融合进多步
denoising；论文显示这种 posterior sampling 可比“先采样、再单独优化”得到更多样的可行
轨迹。这与本项目的几何表示很接近。但它要求推理时知道 cost。当前 stability 的完整地形
信息只在训练期可用，所以这类方法只能作为离线 oracle/teacher。

### 3.3 仅增加 noise-distance loss 也不够

[Mode-Seeking GAN](https://openaccess.thecvf.com/content_CVPR_2019/html/Mao_Mode_Seeking_Generative_Adversarial_Networks_for_Diverse_Image_Synthesis_CVPR_2019_paper.html)
通过增大输出距离与 latent 距离之比，防止条件生成器忽略噪声。这能提供有用的诊断或
辅助正则，但不能成为 Stage 2 主目标：它只要求输出不同，并不保证差异对应可行走廊，
甚至可能主动维持不安全的离散度。

本项目应保护的是 **有用的条件分布结构**，而不是要求每一个 noise 方向都产生大位移。

## 4. 建议的概率模型

令冻结 Stage 1 诱导：

\[
z\sim\mathcal N(0,I),\qquad
y_a=G_1(o,z),\qquad
y_a\sim q_1(y\mid o).
\]

若完整特权环境为 \(\xi\)，联合违反量为 \(V_\xi(y)\)，理想的受约束后验可以写成：

\[
q_\beta^\star(y\mid o,\xi)
\propto
q_1(y\mid o)\exp[-\beta V_\xi(y)].
\]

该式只会重加权 Stage 1 已有支持。为允许同模式局部扩充，可把 Stage 2 写成一个以
proposal 为条件的随机转移核：

\[
u\sim\mathcal N(0,I),\qquad
y_f=y_a+R_\phi(o,y_a,u),
\]

\[
q_\phi(y_f\mid o)
=
\int q_1(y_a\mid o)
K_\phi(y_f\mid o,y_a)\,dy_a.
\]

这里 \(K_\phi\) 学的是 **可行修正的条件分布**，而不是一个 optimizer update。其离线
teacher 核可由如下能量定义指导：

\[
K^\star(y_f\mid \xi,y_a)
\propto
\exp\left[
-\beta V_\xi(y_f)
-\lambda d_G^2(y_f,y_a)
-\omega C_{\rm mode}(y_f,y_a)
\right].
\]

这三个项分别表达联合可行性、有限修正距离和走廊/模式保持。具体采样算法不是监督语义；
多起点优化、Langevin、augmented Lagrangian 或数据中已有可行路径都只用于近似该目标集合。

## 5. 部分观测下必须先补的一层定义

teacher 看到 \(\xi\)，student 只看到 \(o\)。因此真正可部署的目标不能简单写成
\(q_\beta^\star(y\mid o,\xi)\)。至少有两个候选：

\[
\bar q_\beta(y\mid o)
=
\int q_\beta^\star(y\mid o,\xi)p(\xi\mid o)d\xi,
\]

或对一组与观测一致的隐藏补全 \(\Xi(o)\) 定义保守目标：

\[
V_{\rm robust}(y,o)
=
\max_{\xi\in\Xi(o)}V_\xi(y).
\]

前者保留隐藏不确定性下的多模态，后者追求共同可行但可能过于保守。二者不能凭偏好选择，
必须先做 bit-identical 可观测性审计：固定 \((o,y_a)\)，改变隐藏补全，比较 **可行目标集合**
而不是只比较某个优化器返回的一条 residual。

如果不同补全的近邻可行集合显著重叠，可训练保守修正器；如果集合不同但不冲突，应训练
随机后验；如果集合互斥，单帧 observation-only 网络不可能同时保证真实隐藏环境安全。
此时需要输出风险/不确定性、增加观测历史，或规定未知区域的保守行为，而不是加大网络。

[Learning by Cheating](https://arxiv.org/abs/1912.12294) 证明了完整状态 teacher 到视觉
student 的两阶段路线可以有效，但并不消除本任务中特权目标是否由 student 输入可辨识的
要求。

## 6. 对网络结构的直接影响

### 6.1 首选候选：冻结 Stage 1 的随机 proposal-conditioned refiner

\[
\boxed{
z\rightarrow G_1(o,z)=y_a,
\qquad
(o,y_a,u)\rightarrow R_\phi\rightarrow y_f.
}
\]

建议后端本身是一个小型 conditional flow/MeanFlow，而不是确定性 44D 回归头：

- 原 source \(z\) 保留 Stage 1 的全局模式；
- 新 source \(u\) 表达同一 proposal 可能存在的多个可行修正；
- 若 teacher 条件分布近似单峰，网络可以自然弱化 \(u\)，不应人为强迫所有样本高敏感；
- Stage 1 冻结使前端模式不会被后端 loss 直接改写；
- 输出仍在当前 44D 一阶边界约束 B 样条空间中。

后端输入需要显式 proposal-map 对齐。最低结构应包含：

1. 将 \(y_a\) 解码为稠密路径几何；
2. 沿路径从多尺度地图特征中采样局部窗口；
3. 让全路径 token 交互，以表达长路径段 stability/forbidden 协调；
4. 输出稠密位移场再固定提升到 44D，或直接输出结构化 44D residual；
5. 对已安全 proposal 保留 identity target，而不是统一产生非零修正。

[direct coarse-to-fine 审计](../../tests/audits/DIRECT_COARSE_TO_FINE_CONCLUSION_20260731.md)
已经证明 path-aligned 位移场在理想同模式输入上有强信号，但该实验主要是 curvature 修复，
且完整 gate 为 4/5；它支持这种表示进入比较，不支持提前宣布为最终结构。

### 6.2 对照一：冻结 Stage 1 的 latent transport

受 Outsourced Diffusion Sampling 启发，可测试：

\[
z'=H_\phi(o,z,u),\qquad y_f=G_1(o,z').
\]

优点是输出始终位于冻结生成器可达流形，并且 source 的角色清楚。局限也很明确：它只能
重排或探索 Stage 1 的可达支持，无法可靠创造生成器本来不能表达的同模式路径。当前
[D0-confirm](../../tests/audits/d0_confirm_kl02_20260730/REPORT.md) 中 65% 条件在有限池内
observed-zero-safe，而 [D1 支持分层诊断](../../tests/audits/d1_uniform_weighted_20260730/SUPPORT_STRATIFIED.md)
中 D1-W 相对 D1-U 的增益只发生在已有安全支持条件上，因此 latent transport 应作为支持
保持对照，不宜预设为主方法。

### 6.3 对照二：全生成器 reward-weighted Path MeanFlow + reference regularization

这是 ORW-CFM-W2 在本项目中的最近似版本：

- 从当前模型在线刷新 proposals；
- 只用 privileged cost 前向计算 condition-wise 权重，不把 cost gradient 传入生成器；
- 对重加权目标继续做 Path MeanFlow；
- 同时约束当前与冻结 Stage 1 在相同中间状态上的平均速度/输出差异。

必须注意：ORW-CFM-W2 的 W2 理论针对其连续 flow 设定。当前 Path MeanFlow 的
平均速度和单步 endpoint 参数化不同，任何参考场损失在本项目里首先只是 surrogate，不能
直接继承论文的 W2 保证。它需要与同 source endpoint coupling 分别审计。

## 7. 建议的验证顺序

### P0：当前 Stage 1 的噪声响应标定，不训练新网络

先在当前冻结 Stage 1 上固定 condition、mask 和 source，定义并标定：

\[
S_\epsilon
=
\frac{d_G(G(o,z+\epsilon),G(o,z))}{\lVert\epsilon\rVert_2}.
\]

同时用 JVP/randomized SVD 估计物理度量下 \(J_z=\partial G/\partial z\) 的：

- singular-value q10/q25/median；
- effective rank 与 participation ratio；
- 近零奇异值比例；
- 局部有限扰动 gain；
- condition 内 covariance、pair distance、走廊模式质量与熵；
- Safe@1、Safe@K 和 valid/invalid 转移。

P0 只给出未做 Stage 2 时的自然波动范围，不重跑旧 direct-cost。P2 中所有训练 arm，包括
direct-cost 机制对照，都在完全相同的 manifest 上逐 checkpoint 记录这些量。只有这种配对
结果才能回答某种 Stage 2 训练是否真的压低 source influence。

### P1：联合约束后验 teacher 审计，不训练 student

固定真正 environment-disjoint 的研究划分。当前 corrected `dataset1` 的 train/val 地图逐一
相同，只能用于代码 smoke 和同地图路径 holdout，不能用于 Stage 2 架构准入。

每个 condition 固定 Stage 1 proposals，并比较三种 teacher 支持：

1. 只在现有 proposals 内重加权；
2. 冻结 \(G_1\) 后在 latent \(z\) 中搜索；
3. 在 44D 路径空间中做同模式联合约束采样/修正。

至少报告：

- ideal weighted strict-safe mass；
- 原 observed-zero-safe 条件中新增 witness 的比例；
- identity 保留和 passed-constraint regression；
- 模式覆盖、模式质量、ESS/KL 与 \(d_G\) 修正半径；
- 每个 proposal 的可行 target 数量与 target-set 多模态；
- hidden-completion target-set overlap；
- 收益的环境分布。

P1 的核心决策不是选择优化器，而是判断：

\[
\boxed{
\text{Stage 2 主要需要支持重排、latent transport，还是输出空间支持扩充。}
}
\]

### P2：冻结 target archive 后比较三种摊销机制

所有 arm 使用同一组 condition、proposal、teacher target set、训练预算和环境划分：

| Arm | 可训练部分 | 作用空间 | 目的 |
| --- | --- | --- | --- |
| A | 整个 Path MeanFlow | 输出分布 | reward-weighted PathMF + reference regularization 文献基线 |
| B | latent transport \(H_\phi\) | Stage 1 source | 测试仅重排冻结先验是否足够 |
| C | 随机 path-aligned refiner \(R_\phi\) | proposal 邻域路径 | 测试支持扩充与少步后验摊销，当前首选 |

direct-cost 保留为机制对照，不用它选择正则或结构。

主准入量应为：

1. environment-bootstrap Safe@1 相对 Stage 1 的下界大于 0；
2. student 至少保留 P1 ideal posterior Safe@1 增益的 50%；
3. mode coverage retention 不低于 90%；
4. identity proposal 的 valid-to-invalid 不超过预注册上限；
5. source sensitivity、有效秩和条件 covariance 不出现显著收缩；
6. 原 observed-zero-safe 条件中出现显著非零的 invalid-to-strict；
7. 改善不是单个环境或单个约束子类贡献；
8. final test 始终关闭。

其中第 5 项在 P0 得到自然波动范围后再冻结数值阈值，不能现在凭经验指定。

## 8. 当前推荐的决策

目前不应直接实现最终 Stage 2 网络。最小且信息量最大的下一步是：

\[
\boxed{
\text{P0 噪声响应标定}
\;\rightarrow\;
\text{P1 联合约束后验 teacher/可观测性审计}
\;\rightarrow\;
\text{P2 三种摊销机制比较}.
}
\]

如果 P1 表明大部分新增安全质量只能由 44D 输出空间修正获得，优先进入随机
proposal-conditioned refiner；如果 latent search 已覆盖绝大部分收益，则优先选择更简单的
latent transport；只有全生成器 reward-weighted PathMF 明显更强且通过分布保护，才考虑
解冻 Stage 1。

因此当前最有根据、同时最容易被实验推翻的 Stage 2 候选是：

\[
\boxed{
\text{冻结的多模态 Path MeanFlow prior}
+
\text{proposal-conditioned stochastic posterior refiner}
+
\text{训练期联合约束 teacher}.
}
\]

它不是 optimizer 蒸馏，也不是简单 residual 回归；它把后端定义成一个保留随机性的条件
传输核。网络结构是否采用位移场、控制点输出或共享地图编码器，仍由 P1/P2 决定。

## 9. 2026-08-02 已执行验证

### 9.1 cost 契约复核

生产链路现统一为
`gauge44_analytic_yaw_curvature_no_yaw_cost_dense200_physical_bounds_v9`。
本次修正了 signed-mask 的半像素采样偏移、地图上界少一个 cell、uniform stability
场的伪距离以及物理地图长度少一个 cell，并让 hard acceptance 一律使用 1001 点解析
曲率审计。A0-GRAD 的 forbidden/stability/curvature hard mismatch 均为 0，有限差分通过；
训练入口与独立入口连续三步的 loss、输出、梯度、参数和 optimizer state 完全一致。

这只证明实现与当前契约一致，并不把 soft task cost 等同于 strict validity。可行域内部的
stability/forbidden softplus 仍有非零梯度，而 curvature 在通过阈值后没有同等保护；因此
direct-cost 仍可能移动已可行路径并压缩 source influence。它继续只作为 P2 机制对照。

### 9.2 P0：冻结 Stage 1 的 source 响应

冻结 epoch-17 checkpoint，在 16 个不同环境、每条件 32 个 source 上得到：

- strict-valid proposal rate 为 22.85%，Safe@1/4/8/16/32 为
  25.00%/43.75%/43.75%/56.25%/62.50%；
- 路径两两距离中位数 0.390 m，chord-relative 三模式覆盖中位数为 3/3；
- 物理路径 covariance effective rank 中位数仅 2.24；
- 4 个条件的 12 维随机正交 source 子空间中，局部 effective rank 中位数约 2.63；
- source-L2 扰动 0.01 与 0.05 的路径响应 gain 中位数分别为
  0.0533 与 0.0534 m，两个尺度几乎一致。

因此当前 Stage 1 **没有完全忽略 source**，且局部映射在这两个尺度上近似线性；但变异
主要集中于少数方向。该结果是训练前基线，不是 Stage 2 准入门槛。16 个条件中有 12 个
为 complete observation，不能从该样本比较 complete/partial 的 source sensitivity。

### 9.3 P1-screen：三种 teacher 支持机制

复用 P0 的同一 manifest。每个条件保留 32-source pool，并对前 4 个 source 分别执行固定
30 步、source-L2 半径 2.0 的 latent 搜索，以及固定 80 步的 44D path-space 联合修正。
student 不训练，final test 不打开，初始 proposal 始终参与 hard candidate retention。

| 机制 | 条件级 Safe@K/support | pool observed-zero-safe 中新增 witness |
| --- | ---: | ---: |
| 现有 32-source 池内重排上界 | 62.50% | 0/6 |
| 冻结 Stage 1 的 latent 搜索（4 anchors） | 81.25% | 4/6 |
| 44D path-space 联合修正（4 anchors） | 100.00% | 6/6 |

anchor 级 latent/path invalid-to-strict 为 20/49，二者 valid-to-invalid 都为 0。heldout
8 个不同环境中，两者都在 8/8 条件找到至少一个 witness。不过样本仍小：condition
bootstrap 中 latent 相对 pool 的 +18.75 pp 区间为 [-6.25,+43.75] pp，path 相对 latent
的 +18.75 pp 区间为 [0,+37.5] pp，所以这只是机制筛查，不是总体 admission 结论。

两种机制的性质不同：latent 搜索的条件内平均路径移动中位数为 0.266 m、mode
preservation 平均 82.8%，且 source 位移中位数 1.87，已经接近冻结半径；path-space
修正的对应路径移动中位数为 0.467 m、mode preservation 平均 73.4%。因此 path-space
结果证明了输出支持扩充能力，却不支持“统一小型局部 residual”假设。

### 9.4 当前更新后的决定

现有证据排除了两个过强预设：仅在已有 source pool 内重排并不足够；Stage 2 也不能预设
为毫米级同模式微调。latent transport 能补出一部分新支持，值得保留为低侵入对照；
path-space 支持扩充更强，继续支持“冻结 Stage 1 + proposal-conditioned stochastic
refiner”为首选候选，但后端必须容纳较长路径段协调和可能的模式变化。

在 P2 训练前仍需补两个确认：

1. 用 balanced complete/partial manifest 复核 P0/P1，避免当前 observation 构成偏置；
2. 对固定 `(o,z,y_a)` 的多 hidden completion 做 target-set overlap 审计，判断 path-space
   teacher 是否能由部署输入辨识。

在这两项完成前，不开始 Arm A/B/C student，也不把 P1 的 optimizer endpoint 直接当作
唯一确定性回归标签。
