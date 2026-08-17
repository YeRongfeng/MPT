# 相关工作驱动的 Story 与 Baseline 调研

> 调研日期：2026-08-11
> 目的：在继续修改 Abstract、Method 和 Experiments 之前，先确定论文究竟在回答什么问题、
> 与哪些工作发生直接竞争，以及哪些公开方法值得进入实验。本文档是研究定位底稿，不是论文正文。

## 1. 先给结论

Path MeanFlow 不能再围绕“planning support”“特权信息”或“多目标梯度协调”开场。这些
都是方法内部概念，不能让普通机器人审稿人迅速理解问题。论文应从一个直接的工程矛盾
开始：**崎岖地形上的无人车既要避开不能走或容易倾覆的地面，也不能为每次规划反复花费
大量计算。**

相关工作形成了四类已经相当成熟的方案：

1. Uneven Planner、CAP 等方法在每个规划实例上显式建模地形、车辆姿态和稳定性，物理
   解释清楚，但需要搜索或迭代优化；
2. YOPO、YOPO-Rally、ViPlanner 等方法把部分规划计算或规划目标放入训练，使部署更快；
3. MPD、EDMP、FlowMP、Potential-Based Diffusion 等生成式规划器学习多模态路线，但通常
   仍需多步去噪/ODE 积分、部署期代价引导或候选筛选；
4. Kicki 等人的 neural B-spline planner 与 CNP-B 已经表明，可以解析写入 B-spline
   边界、只预测其余控制变量，并在训练时利用可微约束学习快速规划函数。

因此，Path MeanFlow 的合理问题不是“能否首次把物理 cost 摊销进网络”，因为 YOPO 和
ViPlanner 已经覆盖这一宽泛主张；也不是“能否首次做单阶段越野规划”，因为 YOPO-Rally
已经明确提出这一方向；“用样条精确满足端点”也不能单独成立，因为 MPD、CNP-B 和
neural B-spline car planner 都使用了解析边界构造。更可辩护的定位是：

> 能否先从专家路线中学习长距离、可能多模态的全局绕行结构，再把训练时可由完整地形
> 计算的多项目标摊销进同一生成器，使它在部署时只根据当前观测生成一条满足给定边界的
> 全局路径，不再计算物理代价、迭代优化或筛选多条候选路径？

这一定义把两项核心设计放到了同一条因果链上：boundary-aligned representation 负责把
已经给定的起终点位置与朝向从学习问题中拿掉；ERPL 到 PMTA 的二阶段训练先建立路线
生成器，再用三项地形目标更新其参数。两项技术各自都有强先例，当前可辩护的是其面向
部分观测轮式车全局路径的具体组合和实证结果。最终主张应是**联合可行性、路线分布漂移
与在线计算之间的经验权衡**，而不是稳定性无条件超过充分迭代的优化器。

## 2. 调研范围与筛选标准

本轮从以下五条检索路径筛选论文：

- 崎岖/不平整/越野地形上的轮式或腿式机器人规划；
- 学习式快速路径或轨迹生成；
- diffusion、flow 或其他生成模型用于 motion planning；
- 训练期 privileged cost、differentiable cost 或 expert optimization 的计算摊销；
- endpoint-constrained spline、movement primitive 与学习式 trajectory decoder。

进入核心十篇的标准不是单纯的引用量，而是以下因素的组合：

- **任务重合**：是否处理崎岖地形、地面机器人、起终点规划或全局路线；
- **机制重合**：是否学习路线分布、使用样条、物理代价梯度或二阶段训练；
- **部署重合**：是否一次前向、是否在线优化、是否生成并筛选多条候选；
- **主张威胁**：是否已经覆盖我们准备声称的创新；
- **实验价值**：是否有公开代码、清楚接口和可迁移的二维规划设置。

## 3. 筛选池：44 篇相关论文

“核心”表示进入后文重点比较的十篇；“扩展”表示应在 Related Work 中组织性引用；
“背景”表示帮助解释领域演进，但不应占据主要篇幅。

| # | 论文 | 主题 | 本文中的角色 | 级别 |
|---:|---|---|---|---|
| 1 | [Motion Planning Diffusion (MPD), T-RO 2025](https://arxiv.org/abs/2412.19948) | B-spline diffusion prior + inference cost guidance | 最近的生成式规划参照 | 核心 |
| 2 | [You Only Plan Once (YOPO), RA-L 2024](https://tju-air-lab.github.io/projects/YOPO/) | privileged cost-gradient learning + one-stage planning | 训练期代价摊销的主要先例 | 核心 |
| 3 | [YOPO-Rally, 2025 preprint](https://arxiv.org/abs/2505.18714) | single-stage off-road planning | 对“单阶段越野规划”主张的直接约束 | 核心 |
| 4 | [An Efficient Trajectory Planner for Car-like Robots on Uneven Terrain, IROS 2023](https://arxiv.org/abs/2309.06115) | terrain pose mapping + trajectory optimization | 最近的轮式崎岖地形优化器 | 核心 |
| 5 | [Capsizing-Guided Trajectory Optimization (CAP), 2025 preprint](https://arxiv.org/abs/2508.08108) | traversable orientation + capsizing constraint | 倾覆稳定性问题的最近参照 | 核心 |
| 6 | [ViPlanner, ICRA 2024](https://arxiv.org/abs/2310.00982) | differentiable semantic costmap + imperative learning | 训练期规划目标直接更新网络的先例 | 核心 |
| 7 | [EDMP, ICRA 2024](https://arxiv.org/abs/2309.11414) | ensemble-of-costs guided diffusion | 多代价生成式规划参照 | 核心 |
| 8 | [Potential Based Diffusion Motion Planning, ICML 2024](https://arxiv.org/abs/2407.06169) | composable learned trajectory potentials | 多约束组合与迭代生成参照 | 扩展 |
| 9 | [RAPiD, 2026 preprint](https://arxiv.org/abs/2602.07339) | two-step consistency student + best-of-K critic selection | 少步蒸馏和安全选择的最新先例 | 扩展 |
| 10 | [TRG-planner, RA-L 2025](https://arxiv.org/abs/2501.01806) | traversal-risk graph + global planning | 不规则地形安全全局规划参照 | 扩展 |
| 11 | [Learning to Model and Plan for Wheeled Mobility on Vertically Challenging Terrain, RA-L 2025](https://arxiv.org/abs/2306.11611) | learned vehicle-terrain dynamics + planning | 轮式车辆稳定性与可行运动参照 | 扩展 |
| 12 | [Deep-PANTHER, RA-L 2023](https://arxiv.org/abs/2209.01268) | multimodal expert imitation | 优化器摊销与多模态输出 | 扩展 |
| 13 | [ArtPlanner, Field Robotics 2023](https://arxiv.org/abs/2303.01420) | sampling planner + learned motion cost | 地形代价学习但保留在线搜索 | 扩展 |
| 14 | [Motion Planning Networks (MPNet), T-RO 2021](https://arxiv.org/abs/1907.06013) | learned recursive planner + classical repair | 通用快速神经规划 | 扩展 |
| 15 | [Motion Policy Networks (M-pi-Nets), CoRL 2021](https://mpinets.github.io/) | end-to-end motion policy from partial depth | 快速部署与部分观测 | 扩展 |
| 16 | [Path Planning using Neural A*, ICML 2021](https://arxiv.org/abs/2009.07476) | learned heuristic + differentiable A* | 易复现的学习式搜索 | 扩展 |
| 17 | [DiffusionSeeder, CoRL 2024](https://arxiv.org/abs/2410.16727) | diffusion seeds + few-step optimizer | 生成先验与在线优化的折中 | 扩展 |
| 18 | [Diffusion-ES, CVPR 2024](https://diffusion-es.github.io/) | diffusion + gradient-free test-time search | 黑盒在线代价优化 | 扩展 |
| 19 | [Learning High-Speed Flight in the Wild, Science Robotics 2021](https://doi.org/10.1126/SCIROBOTICS.ABG5810) | privileged expert imitation | 快速部署的经典学习路线 | 扩展 |
| 20 | [WayFAST, ICRA 2022](https://arxiv.org/abs/2203.12071) | self-supervised traction-aware traversability | 非几何可通行性学习 | 扩展 |
| 21 | [Learning Multiobjective Rough Terrain Traversability, 2022](https://arxiv.org/abs/2203.16354) | learned speed/energy/acceleration measures | 多目标地形评价 | 扩展 |
| 22 | [Learning Ground Traversability from Simulations, 2017](https://arxiv.org/abs/1709.05368) | robot-specific traversability prediction | 学习地形评价的早期工作 | 背景 |
| 23 | [Off-road Traversability and Planning with Deep IRL, 2019](https://arxiv.org/abs/1909.06953) | demonstration-derived terrain costs | 专家路线与地形代价学习 | 扩展 |
| 24 | [CoMPNet, 2020](https://arxiv.org/abs/2008.03787) | neural constrained motion planning | 约束下的学习式规划 | 背景 |
| 25 | [Rapid and Safe Trajectory Planning over Diverse Scenes through Diffusion Composition, 2025](https://arxiv.org/abs/2507.04384) | composed diffusion + safety filter | 候选生成和部署期筛选 | 扩展 |
| 26 | [Consistency Trajectory Planning, 2025](https://arxiv.org/abs/2507.09534) | single-step consistency trajectory planning | 少步/一步规划趋势 | 扩展 |
| 27 | [PRIMER, 2024](https://arxiv.org/abs/2406.10060) | optimization expert imitation | 高代价优化器的策略摊销 | 背景 |
| 28 | [Diffusion-Based Planning for Autonomous Driving, ICLR 2025](https://github.com/ZhengYinan-AIR/Diffusion-Planner) | guided diffusion driving planner | 大规模驾驶生成式规划 | 背景 |
| 29 | [Flow Planner, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/36d1e8aa9ceec3b781682bf5e63c31bf-Abstract-Conference.html) | flow-matching driving planner | flow-based规划的邻近方向 | 背景 |
| 30 | [Multi-Robot Motion Planning with Diffusion Models, ICLR 2025](https://arxiv.org/abs/2410.03072) | diffusion composition for multi-robot planning | diffusion planning扩展 | 背景 |
| 31 | [Reinforcement Learning for Wheeled Mobility on Vertically Challenging Terrain, 2024](https://arxiv.org/abs/2409.02383) | end-to-end RL planning/control | 地形规划与控制一体化边界 | 背景 |
| 32 | [Generative Trajectory Stitching through Diffusion Composition, NeurIPS 2025](https://comp-diffuser.github.io/) | short-segment composition for long horizon | 长距离生成的相关方向 | 背景 |
| 33 | [Learning Traversability-Aware Global Planners for Long Horizon Off-Road Navigation, 2026](https://arxiv.org/abs/2607.23743) | human-supervised global traversability from overhead data | 长距离越野全局路线参照 | 扩展 |
| 34 | [Speeding up DNN-Based Planning of Local Car Maneuvers via Efficient B-Spline Path Construction, ICRA 2022](https://arxiv.org/abs/2203.06963) | 解析边界 B-spline + car-like neural planner | 轨迹表征和车辆直接规划的最近先例 | 核心 |
| 35 | [Fast Kinodynamic Planning on the Constraint Manifold with Deep Neural Networks (CNP-B), T-RO 2024](https://arxiv.org/abs/2301.04330) | B-spline boundary construction + differentiable constraint learning | 表征与约束摊销的直接 novelty 约束 | 核心 |
| 36 | [BMP: Bridging the Gap between B-Spline and Movement Primitives, 2024](https://arxiv.org/abs/2411.10336) | boundary-satisfying B-spline movement primitives | 样条分布学习和边界条件参照 | 扩展 |
| 37 | [GCOPTER: Geometrically Constrained Trajectory Optimization for Multicopters, T-RO 2022](https://arxiv.org/abs/2103.00190) | MINCO + exact constraint elimination | 降维轨迹参数化的优化背景 | 背景 |
| 38 | [PRESTO: Fast Motion Planning using Diffusion Models, 2024](https://arxiv.org/abs/2409.16012) | diffusion seeds + constrained trajectory optimization | 生成先验仍接在线优化的参照 | 扩展 |
| 39 | [FlowMP: Learning Motion Fields for Robot Planning with Conditional Flow Matching, IROS 2025](https://arxiv.org/abs/2503.06135) | conditional flow matching + B-spline boundary parameters + inference guidance | 生成机制与轨迹表征的最直接近邻 | 核心 |
| 40 | [iPlanner, RA-L 2023](https://arxiv.org/abs/2302.11434) | differentiable planning costs + imperative learning | ViPlanner 之前的直接 cost-training 先例 | 扩展 |
| 41 | [Bi-level Trajectory Optimization on Uneven Terrains, 2024](https://arxiv.org/abs/2404.03307) | differentiable wheel-terrain interaction + bi-level optimization | 轮式车地形物理逐实例优化参照 | 扩展 |
| 42 | [TERP: Traversability Estimation and Local Path Planning on Uneven Terrain, 2021](https://arxiv.org/abs/2109.05120) | elevation-map stability-aware local planning | 有公开实现的轮式地形规划参照 | 扩展 |
| 43 | [PUTN: A Plane-fitting based Uneven Terrain Navigation Framework, IROS 2022](https://arxiv.org/abs/2203.04541) | uneven-terrain sampling and local planning | 有公开实现的不平地形框架 | 扩展 |
| 44 | [Driving on Point Clouds: Motion Planning, Trajectory Optimization, and Terrain Assessment, JFR 2017](https://doi.org/10.1002/rob.21700) | point-cloud terrain assessment + global/local planning | 粗糙地形规划的成熟系统参照 | 背景 |

## 4. 最相关十篇：问题、方法、目的与主张

### 4.1 MPD：学习路线先验，但保留部署期适配

- **提出的问题**：轨迹优化高度依赖初值；简单初值容易陷入局部最优，采样规划得到的
  初值又慢且不平滑。
- **方法**：从既有解学习多模态 diffusion trajectory prior；用 B-spline 控制点降低维度；
  部署时把 learned prior 与任务 cost gradients 结合，在多步去噪中采样 posterior。
- **目的与主张**：兼顾多模态路线先验和新场景下的任务适配，生成多条平滑、可行候选。
- **与本文重合**：专家路线分布、生成式规划、B-spline、起终点条件。
- **关键差异**：MPD 的物理适配发生在部署期，并包含多步采样、cost gradients 和候选选择；
  Path MeanFlow 试图把地形适配放到训练期并保持单路径一次前向。
- **对 story 的启示**：先从“优化器初值为什么不好”讲起，再引出 learned prior；不先讲
  diffusion 数学。

### 4.2 YOPO：训练期物理代价摊销已经存在

- **提出的问题**：感知/建图、前端搜索和后端优化的串联延迟不适合高速飞行；单纯模仿
  专家也不能准确表达安全和平滑代价。
- **方法**：网络根据深度观测预测多个 motion-primitive offsets 和 scores；训练时利用
  ground-truth ESDF 计算轨迹代价的数值梯度并反向传播，部署时一次前向产生并筛选候选。
- **目的与主张**：把传统规划流水线压缩进一个网络，降低规划延迟。
- **与本文重合**：privileged training、cost-gradient supervision、部署期不做轨迹优化。
- **关键差异**：YOPO 是四旋翼局部 receding-horizon trajectory planner，以固定权重组合
  代价，并预测和筛选多条候选；本文是轮式车全局几何路径，先学路线分布，再协调多个
  地形目标，且一次调用不做 best-of-K。
- **对 story 的约束**：不能声称“首次把规划代价摊销到网络”或“首次一次前向规划”。

### 4.3 YOPO-Rally：单阶段越野规划也不是空白

- **提出的问题**：越野环境中的复杂地形和密集障碍使模块化 terrain analysis 与
  pathfinding 难以兼顾实时性和 sim-to-real。
- **方法**：在 Unity 随机森林环境中生成 expert trajectories；以深度图、当前速度和目标
  向量为输入，用行为克隆预测多条 Hermite trajectory candidates 及其 costs，再选择路径。
- **目的与主张**：将 terrain traversability analysis 与 pathfinding 集成到单个网络，
  并从仿真直接部署到实物。
- **与本文重合**：越野地形、快速学习式规划、训练期专家、部署期直接生成。
- **关键差异**：它是传感器到局部候选轨迹的端到端系统，核心监督仍是行为克隆；本文
  处理固定窗口中的全局起终点路径，并用独立第二阶段优化多项物理目标。
- **对 story 的约束**：不能使用“第一个 single-stage off-road planner”这类主张。

### 4.4 Uneven Planner：物理建模强，但逐实例优化

- **提出的问题**：不平整地形既要求判断哪里能走，也要求考虑地形如何改变车辆 SE(3)
  姿态和可跟踪动力学；只做二维避障会过度保守或难以跟踪。
- **方法**：建立从 SE(2) 到地形相关 SE(3) 状态的 terrain pose mapping，并在此基础上
  进行车辆轨迹优化。
- **目的与主张**：生成符合车辆动力学、不过度保守且可跟踪的崎岖地形轨迹。
- **与本文重合**：轮式车辆、不平地形、姿态/稳定性和转向约束。
- **关键差异**：它在每个任务中显式构造地图映射并迭代优化；本文生成二维全局参考路径，
  物理目标仅在训练期塑形，部署端不求解该优化问题。
- **对实验的意义**：允许它获得更高物理可行性；比较问题应是这种收益需要多少在线计算。

### 4.5 CAP：把“走得稳”具体化为防倾覆方向约束

- **提出的问题**：崎岖地形规划不仅需要避障，还要避免车辆发生 tip-over。
- **方法**：分析车辆倾覆稳定性，定义每个地形位置的 traversable orientation，并把它
  写入 capsizing-safety constraint，通过图优化求解轨迹。
- **目的与主张**：在保证导航效率的同时生成防倾覆轨迹。
- **与本文重合**：稳定性目标、位置与朝向耦合的地形可行性。
- **关键差异**：CAP 直接在图优化中执行稳定性约束；本文只在训练时使用连续稳定性目标，
  部署结果没有硬安全保证。
- **对 story 的启示**：“走得稳”需要落到倾覆或姿态风险，而不是抽象的 terrain support。

### 4.6 ViPlanner：训练时直接优化规划目标

- **提出的问题**：纯几何局部规划无法理解草地、楼梯等具有不同 affordance 的地形，
  同时实际机器人需要实时规划。
- **方法**：融合深度与语义，利用 differentiable semantic costmap 和 imperative learning，
  直接根据规划任务目标更新网络权重。
- **目的与主张**：仅在仿真训练，即可快速生成兼顾几何和语义可通行性的局部路径并迁移到实物。
- **与本文重合**：训练期直接优化规划 cost，部署时由网络生成路径。
- **关键差异**：ViPlanner 关注语义可通行性的局部 legged-robot navigation；本文关注轮式
  车辆全局路线、解析起终点边界和多个物理目标间的梯度冲突。
- **对 story 的约束**：不能把“不同于 imitation，直接用 planning objective 训练网络”
  单独作为新意。

### 4.7 EDMP：多种代价已经被用于生成式规划

- **提出的问题**：经典规划能适应新场景，但缺少数据先验；学习方法有先验，却难适应训练外场景。
- **方法**：先学习一般有效轨迹的 diffusion prior，部署时用多种 collision costs 分别引导
  一批去噪轨迹，最后选择 swept-volume cost 最低的路径。
- **目的与主张**：同时获得数据先验、多模态和对新场景的显式适应。
- **与本文重合**：专家轨迹先验、多项 cost、生成式 motion planning。
- **关键差异**：EDMP 的 cost ensemble 是部署期并行引导和候选选择；PMTA 是训练期
  vector objectives 与参数梯度协调，部署时不再评价这些 costs。
- **对 story 的约束**：不能泛称“首次在生成式规划中联合多个物理代价”。

### 4.8 Neural B-spline Car Planner：解析边界和车辆直接规划已有明确先例

- **提出的问题**：直接让神经网络预测离散路径或通用多项式，既增加输出维度，也容易在
  目标位姿处产生偏差，影响训练和部署效率。
- **方法**：用单条高阶 B-spline 表示 car-like vehicle 的局部路径，按照起点和目标构型
  解析构造边界控制点，只让网络预测剩余路径参数。
- **目的与主张**：利用轨迹表征带来的 inductive bias，提高目标满足精度并把局部机动规划
  压缩到近似恒定的网络推理时间。
- **与本文重合**：轮式车辆、B-spline、解析边界、网络预测内部路径变量和直接部署。
- **关键差异**：它处理局部、城市/停车式机动，并不学习多模态长距离全局路线，也没有
  使用训练期完整崎岖地形进行第二阶段多目标适配。
- **对 novelty 的约束**：不能声称“首次通过 B-spline 边界构造保证车辆目标位姿”；本文
  必须证明的是该 reduced trajectory space 在当前全局条件生成任务中的作用。

### 4.9 CNP-B：表征与可微约束摊销已经被联合研究

- **提出的问题**：经典 kinodynamic planning 在复杂约束下计算较重，而既有学习方法又
  难以同时处理动力学、等式和不等式约束。
- **方法**：用 B-spline 解析设置位置及其高阶导数的边界变量，网络一次生成完整计划；
  训练时把多种可微约束写成 constraint-manifold losses，直接优化网络参数。
- **目的与主张**：以近似恒定的网络推理时间生成接近约束流形的计划，并支持快速重规划。
- **与本文重合**：解析边界、只学习剩余样条变量、多约束训练以及部署期直接生成。
- **关键差异**：CNP-B 面向给定任务族的机械臂 kinodynamic trajectory，并不从专家路线
  学习长距离、多模态的地形绕行结构；其约束通过可学习 scaling 构造 manifold loss，本文
  则把 expert route learning 与 privileged terrain adaptation 分成两阶段，并研究多个目标
  梯度的协调。
- **对 novelty 的约束**：本文不能把“边界满足 + constraint loss + constant-time neural
  planning”作为整体首次提出；真正需要实验证明的是长距离路线分布与地形物理适配为何
  要分阶段，以及这种组合在轮式崎岖地形全局规划上的收益。

### 4.10 FlowMP：flow matching、样条边界和内部路径变量已经组合出现

- **提出的问题**：采样或优化方法依赖初值，diffusion 又需要多步去噪；仅学习一阶轨迹
  场还可能得到不够平滑或动态不可执行的运动。
- **方法**：用 B-spline 把起终点位置与速度写成 boundary parameters，只对内部 via-points
  建模；通过 conditional flow matching 学习专家轨迹的位置、速度和加速度分布，并在部署
  时沿 ODE 多步积分，结合任务 cost gradients 进行 posterior guidance。
- **目的与主张**：学习多模态、平滑、动态可行的 motion field，并降低相对 diffusion 的
  迭代生成开销。
- **与本文重合**：专家轨迹分布、flow matching、B-spline、解析边界参数和内部路径变量。
- **关键差异**：FlowMP 面向预定义 2D/3D 环境和机械臂运动，仍需多步 ODE integration 与
  部署期 cost guidance；本文目标是把选定的地形目标在训练期摊销进参数，并在一次 endpoint
  前向中输出轮式车全局路径。
- **对 novelty 的约束**：不能把“flow matching + boundary-conditioned B-spline + expert
  trajectories”作为新组合。需要明确比较任意车辆航向、规范坐标、边界补偿/白化自由坐标
  与 FlowMP boundary-via-point parameterization 的差异，并用受控实验验证其价值。

### 4.11 核心十篇横向对比

| 方法 | 主要任务 | 轨迹/边界处理 | 物理或约束如何进入 | 部署计算 | 对本文的主要约束 |
|---|---|---|---|---|---|
| MPD | 通用/机械臂 motion planning | B-spline；固定首尾位置及零速度/加速度控制点 | 部署期 cost guidance | 多步去噪 + guidance | 内部控制点生成和解析边界并非新意 |
| YOPO | 四旋翼局部规划 | 多个 motion primitives | 训练期 privileged cost gradients | 一次网络调用 + 候选选择 | cost amortization 和快速规划已有先例 |
| YOPO-Rally | 越野局部规划 | 多条 Hermite candidates | expert behavior cloning 与 cost prediction | 一次调用 + 候选选择 | single-stage off-road planning 已存在 |
| Uneven Planner | 轮式车不平地形轨迹 | terrain-aware trajectory optimization | 逐实例地形姿态与动力学目标 | 在线迭代优化 | 物理可行性和速度应作为权衡比较 |
| CAP | 轮式车防倾覆轨迹 | 图优化轨迹 | 显式 capsizing constraint | 在线图优化 | 本文稳定性只有训练目标，不是硬保证 |
| ViPlanner | 腿式机器人语义局部路径 | 学习式 waypoint/path output | differentiable semantic planning costs | 网络直接生成 | planning objective 直接训练网络已有先例 |
| EDMP | 机械臂 motion planning | diffusion trajectories | 多种部署期 collision-cost guidance | 多批去噪 + 选择 | 多 cost 生成式规划已有先例 |
| Neural B-spline Car Planner | car-like local maneuvers | 解析目标构型边界，仅预测剩余样条参数 | 训练期可微地图/车辆约束 | 约 11 ms 网络推理 | 车辆样条边界和直接规划已有先例 |
| CNP-B | 机械臂 kinodynamic planning | B-spline 解析位置及导数边界 | 训练期 constraint-manifold losses | 一次网络推理；可选检查/修正 | boundary + constraints + constant-time 已联合出现 |
| FlowMP | 2D/3D/机械臂 motion planning | B-spline boundary parameters + internal via-points | expert flow prior + 部署期 cost guidance | 多步 ODE integration | flow、样条边界和专家分布的组合已有先例 |

## 5. 文献共同揭示的 Story 空位

### 5.1 已经被覆盖的主张

以下表述不应再作为 Path MeanFlow 的 novelty：

- 首次使用神经网络快速生成路径；
- 首次用 privileged environment information 训练规划网络；
- 首次把 trajectory cost gradient 反向传播到网络；
- 首次将在线规划计算摊销到训练；
- 首次在越野环境中做 single-stage planning；
- 首次使用 diffusion/flow 表达多模态路线；
- 首次在生成式规划中考虑多个 cost；
- 首次实现一步或一次前向的生成式轨迹策略；
- 首次用 B-spline 解析满足起终点位置或导数边界；
- 首次只预测内部样条变量，或以可微约束直接训练神经规划器。

### 5.2 当前仍有意义的组合空位

本轮没有发现一篇工作同时具备以下全部性质：

1. 面向部分地形观测下的**轮式车二维全局起终点路径**；
2. 在单步全局路线生成器中，用解析 spline decoder 精确写入车辆起终点位置与一阶方向；
3. 第一阶段学习 long-range、source-conditioned expert route distribution；
4. 第二阶段用训练期完整地形计算区域越界、倾覆稳定性和曲率三个独立目标，并将其监督
   摊销进生成器参数；
5. 不把三项目标固定压成一个标量，而是协调其参数梯度；
6. 部署时一个 source、一次前向、一条路径，无在线 cost、迭代 refinement 或 best-of-K。

这只是本轮 44 篇论文范围内的**组合差异**，不是“世界首次”的证明。尤其是第 2 至 6 项
分别已有 MPD、FlowMP、neural B-spline planner、CNP-B、YOPO 等局部先例；可能成立的
只是六项在当前任务中的组合与由此带来的经验权衡。正式论文应把贡献
写成这个组合所解决的具体矛盾，并通过消融分别证明每一步有用。

## 6. 建议采用的 Tell-Story 主线

### 6.1 第一层：先让审稿人明白现实问题

建议用下面这类普通语言建立问题，不使用 planning support：

> 崎岖地形上的无人车需要从起点到达目标，同时避开尚未看清或明确不能通行的区域、
> 容易导致倾覆的地形以及过急的转弯。搜索和优化方法可以逐条检查这些因素，但每次规划
> 都需要重新评价地形并迭代求解，在频繁重规划时会产生显著计算开销。

这里的 mask 只对应“尚未看清或不能进入的区域”，不需要在开头命名为一种新的支撑概念。

### 6.2 第二层：现有快速学习方法还缺什么

不能只说“学习方法快但不安全”，因为 YOPO、ViPlanner 等已经直接使用规划代价训练。
更准确的缺口是：

> 专家模仿能够教会模型全局上应当从哪里绕行，但路线相似性本身不直接优化当前地形上的
> 车辆风险。直接加入物理代价后，联合可行性如何变化、专家路线分布发生多少漂移，以及
> 这种变化能否换取更低的部署计算，目前需要通过实验回答，而不能由训练机制预先保证。
> 现有生成式规划器通常通过部署期多步采样、cost guidance、refinement 或候选选择来解决
> 这一矛盾，从而保留了在线计算。

这句话才自然引出“为什么需要两阶段”，而不是先介绍 ERPL、PMTA 或 MGDA 的名称。

### 6.3 第三层：两项设计分别回答两个问题

1. **已知的任务边界为什么还要让网络猜？** 通过 boundary-aligned spline decoder 解析
   固定起终点位置与朝向，使学习变量只描述中间路线。这是一种针对本文全局车辆路径任务
   的表征设计，不能脱离 MPD、FlowMP、neural B-spline planner 和 CNP-B 声称为一般性首次。
2. **如何在已有路线生成器上吸收地形物理监督，并测量相应代价？** 先用 ERPL 学习专家
   路线分布，再用 PMTA 在该生成器上协调区域、稳定性和曲率目标；实验同时报告物理可行性
   变化和路线分布漂移，不预设 PMTA 会保持原分布。

第二点的重点不是 MGDA 算法本身，而是“路线分布学习”和“物理适配”在训练过程中的职责
分离。MGDA 是实现多目标适配的手段。

### 6.4 第四层：把主张限定为可验证的权衡

最稳妥的主张形式是：

> Path MeanFlow amortizes selected training-time terrain objectives into a one-call global route
> generator with analytically fixed endpoint geometry, and empirically evaluates the resulting
> trade-off among physical feasibility, route-distribution drift, and online computation.

对应的中文含义是：用训练时可计算的地形目标更新一次调用全局路线生成器的参数，并实证
评价物理可行性、路线分布漂移与在线计算之间的权衡。这不表示模型能够恢复某个测试实例
中没有观测到的真实地形，也不表示 PMTA 保持原有路线分布。

论文不应声称：

- 比所有搜索或优化方法更稳定；
- 生成路径具有硬安全保证；
- PMTA 必然保持 ERPL 的路线分布；
- 一次前向等于完整导航系统的端到端实时性；
- 使用了完整地形训练就能恢复部署时未观测区域。

### 6.5 Abstract 的五句逻辑骨架

在结果出来前，Abstract 可以按以下逻辑组织，但暂时不写正式英文：

1. **任务**：无人车要在崎岖地形中快速得到一条兼顾到达、倾覆风险和转向能力的路径。
2. **矛盾**：逐实例搜索/优化能够显式评价这些因素但计算较重；快速生成器仅靠模仿又不能
   直接优化当前地形上的物理风险。
3. **设计一**：针对全局车辆路径生成，把给定的起终点位置和朝向解析写入轨迹表示，
   只学习内部路线；贡献落点是任务适配和实证效果，而不是 B-spline 边界条件本身。
4. **设计二**：先学习专家路线分布，再用训练期完整地形计算多目标物理监督并更新生成器；
   其作用表述为摊销这些训练目标，而不是恢复部署时不可见的实例地形。
5. **证据与边界**：报告与搜索、优化和多步生成方法相比的 Feasible@1、规划延迟和计算
   次数，明确呈现可行性与在线开销的权衡。

## 7. Baseline 复现性审查

“最相关”不等于“最值得直接复现”。例如 YOPO 与本文的训练思想很近，但车辆、输入、
输出和规划范围均不同；强行改造成全局轮式车规划器后，实验将不再代表原论文。Baseline
应同时满足任务公平性、科学解释力和工程可执行性。

### 7.1 建议的最小 baseline 组合

| 角色 | Baseline | 回答的问题 | 复现风险与报告方式 |
|---|---|---|---|
| 必须：外部搜索 | Hybrid A* | 相对标准 SE(2) 搜索的速度、路径长度和基本可行性 | [Nav2 Smac](https://docs.nav2.org/configuration/packages/smac/configuring-smac-hybrid.html) 与 [PythonRobotics](https://atsushisakai.github.io/PythonRobotics/modules/5_path_planning/hybridastar/hybridastar.html) 均可复用；unknown 设为禁止，匹配转弯半径和超时 |
| 必须：内部优化 reference | Observation-matched online B-spline optimizer | 同一观测、同一 decoder、同一三项 cost 下，训练期摊销与逐实例优化的差异 | 这是隔离 amortization gap 的受控诊断，不冒充外部 SOTA；若改用完整地形，必须单列为 privileged reference |
| 必须：受控生成 | Standard CFM/DDIM | 提升究竟来自 MeanFlow endpoint、decoder 还是训练协议 | 使用相同数据、条件、decoder、网络容量和训练预算，只更换生成目标/推理步数 |
| 候选：外部生成 | Potential-Based Diffusion 或 MPD-Splines | 一次前向相对外部多步生成与部署期 guidance 的可行性/延迟权衡 | [Potential-Based Diffusion](https://github.com/devinluo27/potential-motion-plan-release) 的 Maze2D、Colab、数据和权重更易启动；[MPD](https://github.com/joaoamcarvalho/mpd-splines-public) 更接近但依赖 IsaacGym。只有 native smoke test 通过后才承诺进入主实验 |
| privileged terrain reference | Uneven Planner | 显式地形建模和逐实例优化能换来多少物理可行性，以及在线代价是多少 | [官方代码](https://github.com/ZJU-FAST-Lab/uneven_planner) 依赖 ROS Noetic、Gazebo、OSQP；full-information 结果单列，不与 partial-information 方法做无条件排名 |

Neural B-spline Car Planner 对第一项表征最有解释力，但原论文是局部车辆机动。其官方代码
适合先做 native reproduction；若换成本文地图、数据和目标重新训练，主表中必须命名为
**Kicki-style B-spline planner adaptation**，不能把适配结果直接标成原论文方法。

### 7.2 低成本补充 baseline

- [Neural A* 官方实现](https://github.com/omron-sinicx/neural-astar) 提供 minimal branch、
  数据和测试，最容易形成学习式搜索 baseline。它适合比较搜索效率和路径长度，但不代表
  崎岖地形车辆物理规划，因此只能作为补充。
- 普通 A* 或 terrain-cost A* 可检查 Hybrid A* 的朝向/曲率建模究竟带来什么，但不需要
  在正文中占用与核心 baseline 相同篇幅。
- [CNP-B 官方实现](https://github.com/pkicki/cnp-b) 提供数据、预训练模型、Docker 和示例，
  复现资料完整，但其机械臂 task/manifold 与本文差异过大。更适合做 native reproduction
  和方法机制核验，不适合强行改造成主表中的轮式地形 baseline。

### 7.3 不建议直接作为主要 quantitative baseline

| 方法 | 原因 |
|---|---|
| YOPO | [代码公开](https://github.com/TJU-Aerial-Robotics/YOPO)，但它是深度图到局部四旋翼多候选轨迹；改造后任务已不是原方法。应作为训练机制近邻和 Related Work，而不是强行做同表排名。 |
| YOPO-Rally | 任务比 YOPO 更接近，但本轮未找到官方代码；仍是局部多候选轨迹、行为克隆和传感器到控制链路。 |
| CAP | 物理问题非常接近，但本轮未找到公开代码；若作者后续发布，应优先重新评估。当前可引用其稳定性建模，不应凭论文描述自行实现后标成“CAP”。 |
| TRG-planner | 全局地形规划任务很接近，但项目页只提供论文与视频，本轮未找到公开实现；适合定义风险与距离的对比问题，当前不能承诺直接复现。 |
| ViPlanner | [代码与模型公开](https://github.com/leggedrobotics/viplanner)，但依赖 semantic/depth local planning 和腿式机器人 affordance，信息与任务均不匹配。 |
| EDMP | 机制相关，但面向机械臂，在部署期对多批候选使用 collision-cost guidance；项目页虽标有 code，本轮没有核实到足够稳定的官方仓库入口。 |
| RAPiD | [官方仓库](https://github.com/ruturajreddy/RAPiD) 面向 nuPlan 道路驾驶；部署使用两步 consistency student 生成 $K$ 条轨迹，再由 critic 做 best-of-$K$。它限制“少步蒸馏和安全引导”的 novelty，但不适合当前崎岖地形 benchmark。 |
| Deep-PANTHER | [代码公开](https://github.com/mit-acl/deep_panther)，但处理动态障碍与四旋翼感知约束，主要价值是说明优化器模仿与多模态摊销已有先例。 |
| ArtPlanner | [代码公开](https://github.com/leggedrobotics/art_planner)，但 learned motion cost 和权重针对 ANYmal，官方也说明无法提供重训练环境；不适合作为轮式车公平 baseline。 |

### 7.4 推荐的执行顺序

1. 先完成 Hybrid A* 和统一 evaluator adapter。
2. 搭建 observation-matched online B-spline optimizer，并补齐 standard CFM/DDIM 受控基线。
3. 原生复现 Neural B-spline Car Planner；适配版只能作为明确重命名的机制对照。
4. 先 smoke test Potential-Based Diffusion，再测试 MPD-Splines；根据实际可运行性决定外部
   生成式主 baseline，不在 smoke test 前承诺 MPD。
5. 最后尝试 Uneven Planner；它可以原生运行时作为 terrain-aware privileged reference，
   不能运行时则如实保留为 Related Work，而不是用不完整复刻冒充原方法。

### 7.5 统一比较协议与必须补充的消融

- 对原生生成并筛选多条候选的方法，同时报告 $K=1$ 的单样本结果和原生
  $K$ + selection 结果；采样、cost evaluation 和 selection 全部计入端到端延迟。
- 第一项表征不能只比较 whitened/unwhitened。至少增加“网络预测全部控制点 + endpoint
  loss”以及“预测全部控制点后做解析 boundary projection”两个受控变体，保持网络与训练
  预算可比，才能说明 reduced boundary-aligned coordinates 的实际贡献。
- FlowMP/MPD 形式差异应单独列清：边界变量、自由变量、source covariance、推理步数、
  是否使用部署期 guidance，以及是否做 candidate selection。

## 8. 对现有研究定位底稿的直接修正

后续重写 `meeting_writing_notes.md` 时至少需要改四处，但本轮先不改：

1. 将“兼顾当前规划支撑”改成普通语言，例如“避开尚未观测或明确不能通行的区域”；
2. 将“MPD 是最接近的方法”改成多轴定位：MPD/FlowMP 对应生成式路线先验与样条表征，
   YOPO/ViPlanner 对应训练期 cost supervision，Uneven/CAP 对应地形物理；
3. 将第二项创新从笼统的 cost amortization 收紧为“expert route learning 后的 privileged
   multi-objective terrain adaptation，以及由此形成的单路径一次前向部署边界”。
4. 将第一项创新从“首次满足边界的 B-spline 表征”收紧为“面向轮式车全局条件生成的
   boundary-aligned reduced trajectory space”，并必须与 MPD、FlowMP、Neural B-spline
   Car Planner 和 CNP-B 做正面对比。

只有先确认这条 story 与 baseline 组合，才应该回到 Abstract、Pipeline 图和 Method 大纲。
