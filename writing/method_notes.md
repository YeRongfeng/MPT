# 方法写作备注

> 本文件不进入论文正文，用于保存当前方法状态、叙事逻辑和主张边界。正文见
> [method_draft.md](./method_draft.md)，正文大纲见
> [method_outline.md](./method_outline.md)，投稿定位见
> [trl_writing_guide.md](./trl_writing_guide.md)。

## 1. 当前方法基线

当前 Method 以 **Stage 1 Path MeanFlow + Stage 2 direct privileged cost** 为
写作基线。Stage 2 直接在部署使用的单步端点上反向传播完整地形 task cost，不
包含安全目标输运、Hungarian matching、独立 critic 或候选池选择。旧方案只属于
历史探索，不再进入正文或摘要。

完整方法由三个主体部分构成：

1. **轨迹表示：** 规定网络生成什么，并解析写入任务已经确定的边界几何；
2. **网络结构：** 建立固定窗口内的规划支撑、任务条件、source 与内部路径之间的映射；
3. **训练方式：** Stage 1 从示范获得路线先验，Stage 2 用训练期完整窗口地形进行
   可行性导向的部署适应。

地形拟合和稳定性计算是三部分共享的物理接口，不作为与三项创新并列的第四部分。

## 2. 系统任务与核心问题

- **应用环境：** 无人车在起伏、非结构化山地中运动。
- **上游感知：** 无人机先行执行前沿探索与地形建图，其覆盖可能只占固定规划
  窗口的一部分。
- **规划输入：** 固定尺寸的三通道表面法向与一通道 planning-support mask。
  mask 同时界定当前已有观测且允许规划的区域。
- **本文规划层：** 神经网络根据当前规划支撑、任务位姿和 source，直接生成一条二维
  全局几何参考路径。
- **下游执行：** 局部规划器负责路径跟随和局部响应；当前生成器不输出速度、时间
  参数化或控制量。

这个问题的关键不是“地图尺寸不完整”，而是固定规划窗口内的地形支撑随 UAV
建图进程变化。三类规划信息具有不同的可用方式：

1. **部署可见信息：** 网络输入尺寸固定，但规划支撑 $\Omega_m$ 随 UAV 覆盖变化；
2. **解析任务结构：** 起点、终点和两端方向已经给定，不需要网络重新学习；
3. **训练期物理监督：** 完整窗口地形可用于评估 forbidden、倾覆稳定性和曲率，但
   部署时不可作为网络输入。

方法的总体思想是把每类信息放入与其性质相匹配的环节，而不是让单个网络和单个
损失同时发现长程路线、恢复已知边界并补偿训练/部署信息差异。

## 3. 三部分方法的科学职责

| 方法部分 | 解决的问题 | 写作中的中心含义 |
| --- | --- | --- |
| 边界对齐轨迹表示 | 原始轨迹变量混合了已知端点几何和未知内部路线 | 让任务几何定义生成空间，网络只预测内部自由度 |
| 规划支撑条件网络 | 固定窗口中的有效地形支撑、任务位姿和路径状态需要形成条件交互 | 把当前规划支撑内的地形证据和 source 映射为一条全局路线 |
| Stage 1 | 局部物理 cost 难以从随机路径提供完整的长程路线结构 | 从示范建立 source-conditioned 路线先验 |
| Stage 2 | 示范先验没有充分利用训练期完整窗口地形中的物理约束 | 在部署端点上用可微地形可行性目标形成安全导向的输出分布 |

轨迹表示回答“模型生成什么”，网络回答“当前规划支撑如何影响路线”，Stage 1 回答
“合理的长程路线从哪里来”，Stage 2 回答“完整窗口地形监督如何进入部署映射”。这些
职责构成因果链，而不是实现模块清单。

## 4. 地形拟合与稳定性接口

- **论文表述：** 参照 Capsizing-Aware Planner，先以局部点云平面拟合获得离散
  高程--法向观测，再通过其地表回归过程形成连续地形法向场；基于稳定性金字塔
  推导位置相关的可通行朝向。
- **本文接口：** 将位置--朝向不稳定集合转换为 yaw 周期的 signed-distance field，
  沿路径切向计算 yaw，并以 stability margin 定义 hard validity 和地形可行性目标。
- **正文篇幅：** Method 只保留“拟合地形 -> 法向场 -> 稳定性余度 -> Stage 2
  cost”这条接口；CAP 的回归和车辆几何推导交给引用或附录。
- **投稿前待核实：** 当前可见的上游实现使用 PCL `NormalEstimationOMP` 和栅格
  插值，尚未定位 ordinary Kriging 代码。若最终数据没有独立 Kriging 实现，正文
  应保持“沿用 CAP 的地表回归”这一概括或改为实际插值方法。
- **引用：** Wei Zhang et al., “Capsizing-Guided Trajectory Optimization for
  Autonomous Navigation with Rough Terrain,” arXiv:2508.08108, 2025。

## 5. 网络结构待定边界

- **环境输入的组织方式与编码器：** [待定]
- **自由路径状态的组织方式：** [待定]
- **任务边界、生成时刻和区间条件的嵌入：** [待定]
- **环境特征与路径状态的交互机制：** [待定]
- **生成主干、输出头和参数规模：** [待定]

在结构冻结前，正文只定义
$f_\theta(c_{\mathrm{obs}},z_t,t,r)\mapsto\hat y_0$ 及其职责，不预先写入
CNN、path token、cross-attention 或全局调制等具体模块。

## 6. Stage 2 当前解释

- **训练目标：** 从 Stage 1 checkpoint 初始化全部生成器参数，在
  $(t,r)=(1,0)$ 的部署端点上直接最小化可微地形可行性目标。
- **目标组成：** 规划支撑、倾覆稳定性和路径曲率的连续违反量；hard
  validity 仅用于独立审计。
- **信息边界：** 完整窗口地形和 cost map 只进入训练损失，生成器输入始终是带
  当前规划支撑的固定地形窗口、任务位姿和 source。
- **部署接口：** 不运行在线 cost 优化器，不增加 critic 或 learned selector；一次
  source 采样和一次网络前向输出一条路径。
- **分布效应：** 当前目标没有示范锚点、分布匹配或模式保持项。所有 source 被同一
  物理 cost 驱动后，可能集中到相同低代价走廊；现有训练中这种模式收缩较明显。

因此，Stage 2 的准确定位是：

> 将示范支持的路线先验适配为安全导向的随机单路径部署策略。

不得将最终生成器称为 multimodal planner。source noise 仍然存在，只说明网络保留
随机输入接口，不证明不同 source 对应不同路线模式。source diversity、路线覆盖和
安全率必须作为相互关联的实验量报告。

## 7. 主张边界

- 路径表示解析保证起终点位置和一阶方向；forbidden、稳定性和曲率不由该表示
  保证。
- $p'(0)=d e(\psi_s)$ 与 $p'(1)=d e(\psi_g)$ 中的 $d$ 是规范几何相位的
  导数尺度，不表示车辆速度。
- 路径坐标规范化不等同于完整规划器的严格 $\mathrm{SE}(2)$ 等变性。
- source 是自由坐标中的标准高斯输入，解码后形成边界对齐的相关路径扰动；不将其
  称为 Brownian bridge prior。
- Stage 1 的 route prior 可以具有 source-conditioned variation，但不把该结构接口
  写成已验证的模式覆盖。
- Stage 2 的 cost 下降、hard validity 改善和 source coverage 变化是三个不同结论，
  必须分别评价。
- direct privileged cost 是当前写作基线，但最终性能结论仍需由冻结协议和未参与
  开发的数据确认。

## 8. 稳定总述

当前方法的完整总述为：

> 边界对齐表示将已知任务几何解析写入生成空间；条件网络根据固定窗口中的当前
> 规划支撑产生全局路线；Stage 1 从示范获得长程路线先验；Stage 2 利用训练期完整
> 窗口地形将该先验
> 适配为安全导向、一次前向的部署策略。

最简洁的职责表达为：

> 解析结构规定生成空间，规划支撑决定路线条件，示范提供路线知识，完整窗口地形
> 提供训练期可行性监督。

## 9. 尚未确定的系统细节

- **无人机建图算法与传感器配置：** [待定]
- **地图更新频率、通信方式与在线重规划触发：** [待定]
- **全局路径与局部规划器的具体接口：** [待定]
- **局部规划器类型及其动态/避障职责：** [待定]

在这些内容冻结前，只将无人机建图写成上游观测来源、将局部规划器写成下游路径
执行模块，不把本文包装成端到端联合空地规划系统。

## 10. PathPainter 与 D-VLC 对 MPT 写法的具体影响

两篇参考论文共同采用“问题张力 -> 方法职责 -> 信息流接口 -> 对应证据”的写法，
但侧重点不同：PathPainter 用一个中心问题把中间表示、搜索和执行串成一条管线；
D-VLC 将多个挑战逐一映射到机制，并在实验中用固定其余条件的比较和针对性消融
验证这些机制。

对本稿的具体要求如下：

1. **Introduction 先写一般问题。** 从固定规划窗口和可变地形支撑下的全局路径生成
   开始；UAV 只作为一种观测来源，不作为论文主角。
2. **Method 先写信息流。** 先说明部署条件、解析边界和训练期完整地形的角色，再
   介绍生成器和损失。读者应能在进入公式前判断每类信息的可见范围。
3. **小节按职责命名。** 推荐使用 Boundary-Aligned Path Space、Mask-Conditioned
   Terrain-Path Interaction、Route-Prior Learning and Privileged Feasibility
   Adaptation 等能表达机制的标题。
4. **段落按因果链收束。** 每段应回答“解决什么困难、采用什么选择、输出什么接口、
   由什么实验验证”，避免只列出实现步骤。
5. **实验保持变量隔离。** 参考 PathPainter 固定下游搜索器、D-VLC 固定共享感知和
   执行模块的做法；MPT 的表示、条件网络和 Stage 2 对照应尽量分别改变，并继续
   保留 final-test isolation 和 frozen manifest。
6. **主张必须小于证据。** 参考 D-VLC 对 bounded memory 和未来通信测量的区分，
   本文也应把“端点由构造满足”“安全率改善”“路线覆盖变化”“OOD 泛化”作为不同
   主张，分别给出证据，不能用一个平均 cost 代替它们。

## 11. 跨章节问题--机制--证据总图

本节是后续 Abstract、Introduction、Method、Experiments 和 Discussion 的共同理解
底稿。修改任何一部分之前，先检查其问题、机制、信息边界和证据是否仍与本节一致。
它的目的不是提供可以直接粘贴到论文中的段落，而是防止把不同组件的职责混为一谈。

### 11.1 中心任务

本文的中心任务不是单独解决“部分地图”或“端点误差”，而是：在渐进建图导致规划
支撑变化的情况下，利用部署时可见的部分地形观测，生成一条具有长程路线结构、满足
任务边界条件并尽可能满足地形物理约束的全局路径。

固定的是物理范围和栅格，不是所有任务中都必须采用固定尺寸输入的学习模型。令
$\Omega$ 为固定物理范围，$\Omega_m$ 为当前已观测且允许规划的支撑。部署输入的
核心语义是当前哪些位置可以被网络作为可靠地形证据和可规划空间使用。

因此，问题应按三个层次理解：

1. **规划支撑层：** 当前可观测且允许进入的空间支撑随建图和障碍配置变化；
2. **轨迹表示层：** 起点、终点和两端方向已经由任务给定，不能交给网络软学习；
3. **学习监督层：** 专家路线提供长程路线结构，但示范目标本身不直接优化每个完整
   地形实例上的物理可行性。

这三个层次分别对应 mask-conditioned input、boundary-aligned trajectory
representation，以及 Stage 1--Stage 2 的监督分工。

### 11.2 Challenge 到方法组件的映射

| Challenge | 具体问题 | 对应措施 | 责任边界 |
| --- | --- | --- | --- |
| 规划支撑变化 | 固定网格内只有一部分区域当前可用；未知区域不能被误当作真实地形 | 显式 planning-support mask，并与法向共同输入条件生成器 | 表示当前支撑，不重建未知地形，不单独保证输出路径避开 mask=0 |
| 边界条件满足 | 普通条件生成器只接收边界位姿，端点满足依赖网络近似 | 规范化坐标、固定边界控制点、边界约束 B-spline 解码器 | 对端点位置和一阶方向提供解析保证；不保证 terrain safety |
| 长程路线结构 | 从随机路径直接优化局部物理 cost 难以发现完整的全局绕行结构 | Stage 1 用示范路线进行 Path MeanFlow imitation learning | 建立 demonstration-supported route prior；不提供 terrain safety 的硬保证 |
| 地形物理可行性 | imitation objective 不直接最小化 forbidden、stability 和 curvature violation | Stage 2 从 Stage 1 初始化，在部署 endpoint 使用训练期完整地形的可微 cost 适配生成器 | 经验性塑造安全导向输出；不提供解析安全保证，也不保证模式保持 |

### 11.3 Mask 的准确语义、作用和生成

#### 11.3.1 Mask 不是简单的空白填充

固定矩形输入只是 mask 存在的接口背景，不是完整的科学动机。网络可以接收零
填充或噪声填充，但如果没有 mask，就无法区分有效法向、未观测区域和不可进入区域。
当前实现中，mask 的统一语义是：

- $m=1$：该位置属于当前允许规划的配置空间；
- $m=0$：该位置不可作为可靠的可规划地形，原因可以是未观测或实体障碍。

当前实现有意将未观测和实体障碍合并到同一个 planning-support mask 中，模型不区分
两种原因。如果未来要分别表达“未知”和“已知障碍”，必须增加独立的观测 mask 或
障碍 mask；不能在现有单 mask 语义下暗示模型已经完成了这种区分。

#### 11.3.2 Mask 的三个作用

1. 保持固定物理坐标和固定张量尺寸，避免因观测区域变化而改变网络接口；
2. 告诉网络哪些法向是真实可用的，防止未知区域的填充值被当作地形几何；
3. 把当前规划支撑作为条件变量，使同一个生成器适配不同的观测覆盖和局部障碍布局。

因此，论文中应把 mask 称为 **planning-support mask** 或 **observed-and-admissible
support mask**，而不是笼统的 `map coverage`。其主要作用是显式表达变化的规划
支撑，而不是补全地图。

#### 11.3.3 当前代码中的 mask 生成协议

数据中若已有 mask，则直接读取；否则当前训练管线通过 `generate_random_mask` 生成
合成支撑。合成过程包括：

1. 以样本级 Bernoulli 决定使用全 1 mask 还是激活遮挡；
2. 在示范路径周围的 informed ellipse 外生成大范围缺失区域；
3. 加入小型局部椭圆障碍；
4. 按车辆半径进行配置空间腐蚀；
5. 拒绝起点、终点、起步/到达通道被阻断或起终点失去连通的候选。

Stage 1 的 `stage1_demo_valid` 语义要求示范路径保持在 mask 内；Stage 2 的
`stage2_independent` 语义允许在示范路径中段制造局部阻塞，从而为绕行和可行性适配
提供训练条件。被 mask 的法向通道使用噪声替换，mask 作为额外通道保留。

当前这套生成协议是部分观测和局部阻塞的合成训练模型，并不等同于完整的 UAV 建图
传感器仿真。若论文保留 UAV 场景，应该把 UAV 写成上游观测来源，并明确当前实验
使用的是规划支撑 mask 协议；不能声称已经实现端到端 UAV mapping。

#### 11.3.4 Mask 的预期证据

mask 机制应通过不同 mask 覆盖、形状和局部障碍条件下的性能比较验证。它主要对应
条件鲁棒性、规划支撑变化下的路线质量和可行性，不对应端点解析误差，也不自动对应
完整未知区域的安全率。

### 11.4 轨迹表示和 Gaussian trajectory prior

新的轨迹表达不是普通的输出层替换，而是先改变生成空间，再在该空间定义源先验：

1. 将路径规范化到统一几何坐标；
2. 用固定边界控制点写入起点、终点和两端方向；
3. 仅对内部控制点建立自由坐标；
4. 在自由坐标中使用 Gaussian source，并通过 B-spline 解码回物理路径。

因此，准确的说法是：生成过程仍从高斯源开始，但该高斯源经过边界对齐自由坐标
和解码器后形成 boundary-projected Gaussian trajectory prior。它不是把标准高斯
完全替换成 Stage 1 学到的 route prior。

这里必须区分：

- **Gaussian trajectory prior：** 生成过程的源分布，提供边界对齐的结构化内部形变；
- **route prior：** Stage 1 从专家路线中学习的条件路线分布。

轨迹表示提供的保证和功能包括：

- 对任意自由坐标，端点位置和一阶方向满足任务边界；
- 网络不需要通过损失恢复已知边界；
- B-spline 提供连续路径和可微的一阶、二阶几何量；
- 曲率等几何量可以接入训练期 cost。

它不保证 forbidden、stability 或 curvature threshold。后者仍然属于 Stage 2 的
训练目标和离线审计。

### 11.5 Stage 1、Stage 2 和联合训练的关系

#### 11.5.1 Stage 1 的职责

Stage 1 的主要问题是“从哪里获得长程路线结构”。局部物理 cost 从随机路径出发
通常只能提供局部梯度，不能稳定地发现完整路线的全局拓扑和绕行结构。因此 Stage 1
使用专家路线建立 demonstration-supported route prior。

Stage 1 不能被写成 terrain-feasibility guarantee。即使专家路径大多可行，网络的
近似输出、不同 source 和变化的 planning support 仍可能导致物理约束违反。

#### 11.5.2 Stage 2 的职责

Stage 2 解决的是“已有路线先验没有直接接受完整地形物理监督”。它从 Stage 1
checkpoint 初始化，在部署使用的单步 endpoint 上直接计算并反向传播：

- planning-support / forbidden cost；
- stability cost；
- curvature cost。

完整地形只存在于训练损失和离线审计中，部署时不读取完整 cost map，也不运行在线
优化器。Stage 2 的准确定位是：

> 将示范支持的路线先验适配为安全导向的随机单路径部署策略。

它是经验性 feasibility adaptation，而不是“确保所有约束满足”。

#### 11.5.3 为什么当前采用分阶段训练

分阶段训练的设计理由是监督和优化角色不同：

1. 先用示范建立可用的长程路线结构，再使用物理 cost 做局部可行性适配；
2. 避免随机路径上的局部物理梯度主导训练；
3. 把 route learning 和 feasibility adaptation 的效果分开归因；
4. 让 Stage 1 checkpoint 成为明确的 imitation-only baseline。

联合损失并非不可行。联合训练可能减少 Stage 2 对路线先验的遗忘，但需要处理 imitation
和 physical cost 的梯度冲突、损失权重、训练阶段和模式收缩问题。当前方法没有证明
分阶段训练在理论上必然优于联合训练；如果没有相应消融，论文只能把它表述为与两类
监督职责匹配的训练设计，不能写成唯一正确方案。

当前 Stage 2 不包含示范锚点、PMF anchor、分布匹配或显式模式保持项。因此它可能把
多个 source 压向同一低代价走廊，路线覆盖和多样性必须作为独立指标报告。

### 11.6 Privileged 的准确含义

Stage 2 的“特权”来自训练和部署之间的信息不对称：

- 部署输入：部分法向、planning-support mask、起终点位姿和 source；
- Stage 2 训练监督：完整地形、完整 stability/cost map 以及由其计算的违反量。

完整地形不作为网络输入，而只通过 loss 产生梯度。这属于广义的 learning with
privileged information，但当前实现没有独立 privileged teacher，也没有 teacher--student
目标。因此正文优先使用 **privileged terrain-feasibility fine-tuning**，不要称为
privileged distillation。

如果只用部署可见的部分地形计算 cost，Stage 2 仍可能作为普通 constraint-aware
fine-tuning 存在，但它只能优化已观测区域，无法评估未观测区域的真实稳定性，也不再
体现“训练期完整地形监督被摊销到部署映射”的核心信息边界。

### 11.7 组件和预期提升

| 组件 | 主要预期提升 | 不应承诺的提升 |
| --- | --- | --- |
| planning-support mask | 不同观测支撑和局部障碍下的条件鲁棒性 | 地图重建、未知区域安全保证 |
| boundary-aligned representation | 端点位置和方向误差精确满足 | terrain safety |
| Gaussian trajectory prior | 结构化、相关的内部路径形变和稳定几何接口 | 物理可行性保证 |
| Stage 1 | 长程路线结构和示范路线质量 | 所有地形约束自动满足 |
| Stage 2 | hard feasibility、Safe@1、forbidden/stability/curvature 指标 | 解析安全保证、模式保持、多模态保证 |
| 一次前向部署 | 推理效率和部署简洁性 | 不运行优化器不等于天然安全 |

每个组件应由与其职责匹配的指标验证，不能用一个平均 cost 代表所有结论。特别要
分开报告 endpoint guarantee、continuous cost、hard validity、source coverage 和
OOD generalization。

### 11.8 各论文部分的写作约束

- **Introduction：** 从“部分规划支撑下的全局约束路径生成”建立一般问题，再指出
  route structure 与 terrain feasibility 的监督缺口；不要只列 partial observation
  和 endpoint violation 两个部署问题。
- **Abstract：** 先交代中心任务和三类要求，再在方法主体中同时出现 boundary-aligned
  representation 与两阶段 feasibility adaptation；Stage 2 不能作为最后的训练细节。
- **Method：** 按信息流和职责写：planning support -> trajectory representation ->
  Stage 1 route prior -> Stage 2 privileged feasibility adaptation -> deployment boundary。
- **Experiments：** 至少分别考察 mask 条件、边界表示、Stage 1/Stage 2 和 privileged
  information 的作用；比较时固定其余输入、source 和评估协议。
- **Discussion：** 明确合成 mask 与真实 mapping 的差距、Stage 2 没有硬安全保证、当前
  没有模式保持项，以及完整地形只在训练期可用的适用范围。

### 11.9 禁止混用的表述

- 不把固定尺寸输入说成所有 learning-based planner 的共同属性；
- 不把 planning-support mask 写成地图恢复；
- 不把 Gaussian trajectory prior 写成 Stage 1 route prior；
- 不把 boundary construction 的保证归因于 Stage 2；
- 不把 Stage 2 的 soft-cost optimization 写成绝对安全保证；
- 不把 source noise 自动解释成最终生成器保持多模态；
- 不把当前合成 mask 协议包装成已完成的端到端 UAV mapping 系统。
