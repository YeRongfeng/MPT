# T-RL 投稿定位与写作指南

> 调研日期：2026-08-02  
> 用途：本文件不直接进入论文正文，用于指导后续的论文定位、Introduction、
> Method 与 Experiments 写作。T-RL 尚处于创刊初期，正式投稿前应重新检查其
> 最新卷期、专题和作者指南。

## 1. 期刊定位

IEEE Transactions on Robot Learning (T-RL) 于 2026 年正式启动，目前尚未
形成足够长的发表历史，不能把少量早期文章概括为固定的 T-RL 文风。现阶段的
写作判断应以期刊官方范围和审稿要求为硬约束，并以 T-RO、RA-L、CoRL、RSS
等相邻渠道的成熟论文作为叙事参考。

T-RL 关注的不是一般意义上的“将神经网络用于机器人”，而是针对机器人和
自动化系统特有挑战的 AI 与机器学习方法。这些挑战包括物理系统约束、有限
数据、跨平台和跨环境迁移、泛化与鲁棒性、训练和部署效率、安全可靠控制、
可解释性以及真实系统落地。期刊同时接收具有理论意义的基础算法和具有实际
意义的应用工作，并明确期待真实硬件实验作为仿真评估的补充。

与本文直接相关的 T-RL 主题包括：

- 融合机器人几何或物理结构的学习算法与结构化输出空间；
- 从示范中学习机器人轨迹先验与随机路径生成；
- 部分观测和非结构化环境中的泛化与鲁棒部署；
- 数据收集、策略训练和分阶段学习方法；
- 学习规划器在真实机器人系统中的效率、可靠性与可复现评价。

Regular Paper 要求结果具有原创性、影响力和实质性的 robot learning 推进，
正文最多 12 页 IEEE Transactions 双栏格式，摘要不超过 200 词，并采用
双匿名评审。官方审稿指导重点询问：论文贡献是什么、重要性是否得到解释、
Introduction 是否明确陈述研究目的、全文是否清楚组织，以及方法是否技术
可靠。因此，正文需要围绕一个可验证的科学命题组织，而不是平铺系统模块。

官方资料：

- [T-RL Purpose and Mission](https://www.ieee-ras.org/publications/t-rl/)
- [T-RL Information for Authors](https://www.ieee-ras.org/publications/t-rl/information-for-authors/)
- [T-RL Information for Reviewers](https://www.ieee-ras.org/publications/t-rl/information-for-reviewers/)
- [T-RL Launch Announcement](https://www.ieee-ras.org/new-journal-launch-ieee-transactions-on-robot-learning-t-rl/)

## 2. 近邻论文与可借鉴内容

以下论文不是 T-RL 已形成的固定模板，而是从方法、任务和 Transactions 写作
三个维度选择的近邻参考。

| 论文 | 与本文的关系 | 主要叙事结构 | 可借鉴与不可借鉴之处 |
| --- | --- | --- | --- |
| [BridgeFlow](https://arxiv.org/html/2607.14725) | Flow Matching、端点对齐的 source、轨迹生成 | 先提出几何泛化与实时性的矛盾，再由 prior、OT、环境引导和规范化分别解决子问题；实验按 RQ 组织 | 可借鉴“生成 source 和表示应预先包含任务结构”的思想；不能迁移其 Brownian bridge、OT 成本降低、流场直化或严格 SE(2) 等变结论 |
| [DARE](https://www.marmotlab.org/publications/75-ICRA2025-DARE.pdf) | 部分 belief map、从示范学习、显式生成长程路径 | 先定义部分地图下的探索问题，再介绍 belief 编码、生成策略、特权示范来源和闭环部署 | 适合参考部分观测和显式长程路径的叙述；其目标是主动探索，本文目标是由无人机观测引导的地面全局规划 |
| [Motion Planning Diffusion](https://arxiv.org/html/2308.01557v2) | 多模态轨迹先验、生成结果交由底层控制器执行 | 运动规划问题、planning as inference、生成先验、引导与规划代价依次展开 | 适合参考 Method 中从问题形式自然推导生成模型的逻辑；本文的解析边界空间可比端点 inpainting 更深入地展开 |
| [DiPPeR](https://arxiv.org/html/2310.07842v3) | 地图图像条件、二维全局路径生成、真实机器人部署 | 数据生成、图像条件扩散、推理管线、仿真和实机 | 是直接近邻和潜在基线，但写法偏短篇系统报告，不应成为本文整体文风模板 |
| [SPARTA](https://proceedings.mlr.press/v305/dong25a.html) | 起伏地形、方向相关可通行性、解析结构注入网络输出 | 从车辆状态相关的地形作用出发，将几何结构写成解决数据量、泛化和规划效率问题的原则 | 适合参考如何把解析表示提升为 robot learning 贡献，而不是把它写成预处理步骤 |
| [NeuPAN](https://arxiv.org/abs/2403.06828) | 物理约束、解析模型与学习结合、真实系统验证 | 真实物理困难、核心 crux、系统和数学模型、网络、理论性质、仿真与多平台实机 | 是 Transactions 层次叙事的重要参考：先给出统一问题，再让各模块承担清楚且必要的角色 |
| [Potential-Based Diffusion Motion Planning](https://proceedings.mlr.press/v235/luo24h.html) | 学习生成先验与传统规划结构结合 | 保留传统方法的可组合优势，同时用学习消除全局优化瓶颈 | 适合参考“保留机器人问题中的有价值结构，再用学习处理剩余困难”的叙述方式 |
| [YOPO](https://tju-air-lab.github.io/projects/YOPO/YOPO.pdf) | 训练期完整环境信息、部署期有限观测、可微路径 cost | 先提出模仿与强化学习的监督缺口，再以环境 cost 的数值梯度直接指导网络，并单独说明 privileged information split | 本文借鉴训练/部署的信息边界；当前 Stage 2 将完整地形上的可微 task cost 直接反向传播到部署端点，但不照搬其 numerical-gradient 结构 |

NoMaD 等工作也使用 mask，但其 mask 用于切换是否提供目标条件，与本文表示地图
支撑域的 mask 含义不同，不能因为术语相同而作为最直接的方法对照。

## 3. 本文的推荐定位

不建议将论文定位为：

> 无人机完成建图，神经网络生成路径，局部规划器负责跟踪。

该表述只描述了系统组件，没有给出可推广的 robot learning 问题。推荐定位为：

> 面向固定规划窗口内可变地形支撑的边界对齐全局路径学习。

对应的物理矛盾是：无人机的先行观测延展了地面车辆的规划视距，但其渐进式建图
可能只覆盖神经规划器固定空间窗口的一部分。网络输入尺寸保持不变，真正变化的是
窗口内当前具有地形证据且允许规划的空间支撑。更进一步，任务边界已经准确给定，
而完整窗口地形上的物理可行性又只在训练阶段可用。

本文可以由下面的核心 insight 统领：

> Global path generation operates on a fixed terrain window with variable
> planning support: current support conditions deployment, endpoint geometry
> is embedded analytically, and complete-window feasibility is available only
> as training supervision.

对应的中文表述为：

> 全局路径生成面对的是固定地形窗口和可变规划支撑：当前支撑决定部署时的路线
> 条件，端点几何被解析写入，完整窗口地形可行性则只作为训练监督。

这条主线同时容纳轨迹表示、条件网络和分阶段训练，不再让边界表示独自承担全文
故事；它也不需要借助类人感知比喻或未经验证的性质。

## 4. 三部分方法的叙事职责

完整方法始终由三个部分组成，三者应共同服务于上述核心问题。

### 4.1 边界对齐的轨迹表示

轨迹表示回答“模型生成什么”。规范几何相位、任务坐标系、一阶边界 B 样条和
白化自由坐标共同定义边界一致的生成空间。其主要贡献不是压缩路径维度，而是将
任务已经确定的几何从学习问题中解析消除，使生成过程只作用于内部路径自由度。

推荐用语包括：

- boundary-aligned path space；
- boundary-consistent generative coordinates；
- learning only the unresolved interior geometry；
- task geometry as part of the generative space。

在没有额外证明和实验前，不使用以下表述：

- Brownian bridge prior；
- 一般性的 optimal transport cost reduction；
- straightened flow field；
- 完整规划器的严格 SE(2) equivariance。

### 4.2 规划支撑条件网络

网络回答“固定窗口中的当前规划支撑如何决定路线”。其科学职责不是简单编码四通道
图像，而是建立规划支撑内的地形证据、起终点任务、source 和自由路径状态之间的条件交互，从而
形成与当前地形证据一致的内部路线。

网络结构尚未冻结，因此当前只保留这一职责和统一输入输出接口。最终结构确定后，
应从这一职责反推编码器、路径状态组织方式和条件融合机制，不能先选网络模块再
倒推叙事。

### 4.3 分阶段训练

训练策略回答“路线知识如何获得，物理监督如何进入部署映射”。Stage 1 条件
Path MeanFlow 从示范中建立 source-conditioned 路线先验，并支持单步全局路径
采样。Stage 2 在部署端点上直接反向传播由完整窗口地形计算的可微地形可行性目标，
将规划支撑、倾覆稳定性和路径曲率监督吸收到同一个生成器中。该目标没有显式模式保持项，因此
安全塑形可能伴随路线覆盖收缩；最终方法定位为随机单路径规划器，而不是多模态
生成器。

最终 Introduction 的三个贡献点仍应与轨迹表示、网络和训练三部分对应。生成器
网络和 Stage 2 训练协议冻结前，相关贡献只陈述机制，不提前写入未经独立确认的
优越性。

## 5. 推荐正文架构

### I. Introduction

1. 从起伏山地全局路径规划的两个基本需求开始：满足端点位姿，并在部分已知地形
   上保持可行；
2. 引出渐进式建图（无人机前视测绘是其实验实现之一）在固定规划窗口内形成的
   可变地形支撑；
3. 将问题定义为 fixed-window global path generation under variable terrain support；
4. 说明三类信息为何需要不同机制，而不是由同一无结构预测问题处理；
5. 给出“让每类规划知识进入与其作用相匹配的环节”这一核心原则；
6. 概述轨迹表示、网络和训练三部分；
7. 用可由实验逐项验证的贡献列表收束。

### II. Related Work

- Learning-Based Global Planning in Partial and Off-Road Environments；
- Generative Models for Robot Trajectories；
- Structured Trajectory Representations and Constraint-Aware Learning。

空地协作在本文中主要界定观测来源。若无人机建图和空地协同算法本身不是贡献，
不需要扩展成与生成规划同等篇幅的相关工作主线。

### III. Problem Formulation and System Overview

- UAV 前沿探索、地形建图与 UGV 全局规划之间的系统关系；
- 固定规划窗口、地形法向、planning-support mask、起终点位姿和目标路径的定义；
- mask=1 与 mask=0 的当前规划语义；
- 全局路径输出与下游局部规划器的接口；
- 条件路径分布的学习目标。

### IV. Method

- Method Overview；
- Boundary-Aligned Geometric Path Representation；
- Mask-Conditioned Terrain-Path Interaction Network；
- Demonstration Route-Prior Learning and Privileged Safety Adaptation；
- Inference。

“一致的数据与评价合同”不作为 Method 独立章节。示范路径的拟合、平滑、缓存和
失败记账也不作为主要方法贡献，除非后续的数据或教师生成机制本身形成新方法。

### V. Experiments

实验应按科学问题组织，而不是按照代码模块组织。建议至少回答：

- RQ1：边界对齐表示是否持续、精确地满足起终点位置和方向，并改善基础生成
  学习？
- RQ2：模型能否随 planning-support coverage 和障碍配置变化，利用当前地形支撑生成有效的
  全局路线？
- RQ3：条件网络的关键交互机制是否优于较简单的图像条件或拼接基线？
- RQ4：Stage 2 在改善物理有效性时造成了多少 source diversity 和路线覆盖损失？
- RQ5：完整规划器能否以足够低的延迟在未见地形和真实 UGV 闭环中工作？

评价应覆盖表示性质、生成质量、source diversity 与路线覆盖、planning-support
coverage、
环境外泛化、规划时延、
下游执行成功率和真实硬件表现。具体指标及阈值放在 Experimental Setup，而非
Method。

## 6. 叙述风格准则

较强的 robot learning 论文通常遵循以下逻辑：

1. 真实物理矛盾；
2. 可推广的学习问题；
3. 一个统领全文的核心原则；
4. 与该原则一一对应的方法组成；
5. 与贡献一一对应的研究问题和证据。

后续写作应遵守：

- 先写问题和方法为何必要，再写网络和训练公式；
- 用正面定义建立论文范围，不在正文中插入防御性自我否定；
- 可以使用一个有物理依据的高层概念，但随后必须立即给出数学定义；
- 方法名称和术语应总结真实机制，不能替代证据；
- 不使用“completely”“strictly”“naturally”等绝对词，除非已有解析证明；
- 每个贡献必须能在实验中找到独立的对照、消融或真实系统证据；
- UAV-UGV 场景负责说明问题为何重要，边界对齐生成学习负责说明方法为何具有
  普遍意义。

## 7. 当前投稿判断

本文具备进入 T-RL 范围的基础：问题来自真实机器人的部分观测与几何边界，轨迹
表示具有明确的解析结构，分阶段学习利用了训练/部署信息差异，并计划接入真实系统。

最终稿件能否达到 T-RL 层次，主要取决于以下尚未完成的部分：

- 网络结构是否围绕固定窗口内规划支撑与路径状态的交互形成独立贡献；
- Stage 2 的物理有效性改善与路线覆盖收缩是否能在未参与开发的数据上共同确认；
- 轨迹表示相对 raw waypoints、端点损失、inpainting 或其他结构化 source 的
  优势能否被公平验证；
- 未见山地和 planning-support coverage 变化是否得到系统评价；
- UAV 提供观测、神经全局规划、局部规划器执行的真实闭环是否得到硬件验证。

在这些部分确定之前，论文的稳定总述是：

> 轨迹表示将已知边界写入生成空间，条件网络根据固定窗口中的当前规划支撑产生
> 路线，Stage 1 从示范获得长程路线先验，Stage 2 利用训练期完整窗口地形将该先验适配为安全导向的
> 单路径部署策略。

## 8. PathPainter 与 D-VLC 的精读写法

这两篇论文值得借鉴的主要是叙事组织，而不是其中的具体技术模块。PathPainter
采用“一个中心问题、一条系统管线、两类证据”的写法；D-VLC 采用“挑战列表、
机制对应、能力验证”的写法。MPT 应吸收这两个层面的共同原则。

### 8.1 PathPainter：先提出可执行的中心问题

PathPainter 的 Introduction 先说明 BEV 先验为什么有价值，再指出现有表示会丢失
几何细节、道路中心假设覆盖不足，随后提出一个明确的 reformulation question：能否
把 BEV 导航转化为图像生成问题，并将生成结果变成可搜索、可执行的路径。Method
紧接着给出两层 pipeline，再分别说明生成中间表示和执行时的定位接口。

对 MPT 的直接启发是：

- Introduction 先从 fixed-window global path generation under variable terrain support
  开始，不从 UAV 或数据管道开始；
- 先写当前规划条件为什么难以直接产生可执行路线，再提出“已知任务几何、可变规划
  支撑、训练期完整地形”分别应进入哪个接口；
- 每个 Method 段落按“职责 -> 操作 -> 结果接口”推进，公式只固定该接口；
- 先定义输出的可执行含义，再介绍生成器，而不是把生成模型本身作为故事中心；
- 实验先说明评价的不是单一中间量，而是中间表示对下游路径质量的作用。

PathPainter 还有一个重要的实验写法：它把所有方法的输出送入同一个 A*，从而把
“中间表示质量”和“搜索器差异”分开。MPT 也应尽量固定解码器、评价器和信息边界，
再比较表示、条件交互和训练策略。

### 8.2 D-VLC：挑战、机制和证据一一对齐

D-VLC 在 Introduction 中明确列出四个挑战：任务分解与分配、信息共享、异构协作和
统一动作执行。随后用 First、Second、Third、Finally 按相同顺序给出四个机制，
贡献列表再次复述这条映射。Method 的第一段还主动限定 decentralized 的含义，避免
初始化阶段与在线执行阶段被混为一谈。

对 MPT 的直接启发是：

- 将核心困难写成少数相互区分的问题，并保证每个困难都有一个对应的设计回答；
- 贡献列表、Method 小节和实验 RQ 使用相同顺序，不让读者自行建立映射；
- 在 Method 开头先写信息流和接口边界：部署网络看到什么，完整地形只在哪里出现，
  Stage 2 到底改变什么；
- 使用“System Setting and Information Flow”式的标题，让标题表达科学职责，而不是
  代码模块名称；
- 方程用于定义输入、输出、状态更新和约束，不用连续公式堆积来替代解释；
- 参考其保守表述，例如只称 bounded memory 就不顺带声称已经测得通信节省。MPT
  同样不能把 source noise 写成已验证的 multimodal coverage，也不能把诊断结果写成
  strict OOD 结论。

D-VLC 的实验还把共同的感知、动作专家和评价条件固定，只改变 VLM backbone 或
目标选择逻辑，再用跨场景、组件消融和能力矩阵分别回答泛化、模块必要性和系统覆盖。
这正适合 MPT 的冻结 manifest、表示消融、planning-support coverage 和 Stage 2
安全/覆盖分开报告。

### 8.3 MPT 的改稿规则

后续重写 Method 和 Introduction 时，统一使用下面这条因果链：

> 任务几何决定生成空间；当前规划支撑决定部署条件；示范提供长程路线知识；完整窗口地形提供训练期可行性监督。

每个主要小节都应完成四步：先指出一个具体学习困难，再给出一个结构或训练选择，
随后给出数学接口，最后说明该接口对推理或评价的意义。当前网络内部尚未冻结，仍
只写统一映射 $f_\theta(c_{\mathrm{obs}},z_t,t,r)\mapsto\hat y_0$ 及其职责。

因此，现有 Method 不需要增加更多模块，而需要进一步做到：

- 将“地形与轨迹表征”拆成可读者理解的职责标题，突出 boundary-aligned path space；
- 在 Stage 1 和 Stage 2 之间明确“路线先验”和“训练期可行性适配”的分工；
- 在每个方法主张后预先绑定一个可验证证据，尤其区分端点保证、路径质量、hard
  validity、source diversity 和 OOD 泛化；
- 把系统背景、实现配置和未冻结细节放到相应位置，不让它们打断中心问题的推进。
