# Abstract Writing Notes

> 本文件不进入论文正文，用于记录摘要的叙事逻辑、方法职责和主张边界。可直接
> 使用的英文摘要见 [abstract_draft.md](./abstract_draft.md)，纯内容大纲见
> [abstract_outline.md](./abstract_outline.md)。

## 1. 摘要真正要建立的问题

本文中的“不完整”不是输入张量尺寸不足，也不是环境地图天然残缺。就本文规划器的
输入接口而言，网络始终接收固定尺寸的地形窗口；随着渐进式建图（无人机前视测绘只是其一种实验实现），窗口
内只有部分栅格具有真实地形观测，其余栅格属于未观测或禁入区域。问题可以用实现
直接对应的语言描述：

> 就本文的固定输入接口而言，网络必须依据掩码标记的当前观测
> 区域生成路径。

掩码通道 $m$ 直接区分已观测且允许规划的区域与未观测/禁入区域；建图进度改变的
是窗口中 $m=1$ 的栅格集合，而不是网络输入的空间尺寸。

该任务需要将三类知识组织成一条规划链：当前支撑内的地形证据决定路线条件，给定
端点位姿定义路径空间，示范提供长程路线先验，完整窗口地形则在训练阶段提供可微
可行性监督。摘要需要描述这条转化关系，而不是罗列 UAV、B 样条、MeanFlow 和多个
cost 名称。

## 2. 各方法部分的科学职责

| 方法部分 | 它解决的问题 | 摘要中应表达的意义 |
| --- | --- | --- |
| 渐进式建图与 mask 通道 | 固定尺寸窗口内的有效观测覆盖随建图进程变化 | mask 显式标记当前已观测且允许规划的区域，是网络的实际输入通道 |
| 边界对齐轨迹表示 | 原始轨迹生成把已知端点几何和未知内部路线混在同一学习空间 | 解析写入起终点位置与方向，使网络只生成内部路线自由度 |
| 地形支撑条件网络 | 同一固定窗口在不同建图阶段具有不同的有效支撑 | 学习规划支撑内的地形证据和 source 到内部路线的映射 |
| Stage 1 | 从随机路径出发，仅靠局部物理 cost 难以获得示范中的长程路线结构 | 从示范建立带有 source-conditioned variation 的 route prior |
| Stage 2 | 示范先验尚未吸收完整窗口中的物理可行性信息 | 用可微地形可行性目标适配部署映射；可能牺牲 Stage 1 的路线覆盖 |
| 局部规划器 | 全局生成器输出的是几何参考路径，而不是完整控制策略 | 只交代下游接口，不将局部控制写成本文方法贡献 |

这里最重要的因果关系是：规划支撑决定**哪些地形证据当前有效**，轨迹表示决定
模型**生成什么**，Stage 1 决定路线知识**从哪里获得**，Stage 2 决定这些路线
**如何适应完整地形可行性**。它们共同把渐进式地形观测转化为一次式部署路径。

## 3. 推荐叙事顺序

1. 从具体挑战开始：崎岖地形下，端点约束与地形可行性必须同时成立。
2. 给出两个困难：本文规划器的固定输入窗口可能填不满（掩码语义）；端点作为学习条件
   （边界违例）。无人机只是渐进式建图的一种实验来源，不承担叙事重心。
3. 提出核心洞见：把已知任务结构移出网络——端点属于表示层，观测覆盖属于输入层。
4. 提出 Path MeanFlow 规划器，机制逐一回应两个困难：边界控制点保证端点，mask
   表达观测覆盖，两阶段训练引入完整地图代价。
5. 如部署效率或信息边界是论文主张，再补充部署承诺；否则不要让摘要尾部变成系统管线清单。
6. 用冻结实验设置、核心数字和一句总体意义收束。

## 4. 当前英文摘要的中心句

当前候选版本用一个任务句和一个缺口句完成开场：

> A global planner for rough terrain must generate a traversable path from the terrain
> observations available at planning time while satisfying prescribed start and goal poses.
> In our setting, progressive mapping may leave only part of the planner's fixed-size window
> covered by observed terrain, while boundary poses enter the model only as conditioning inputs,
> leaving endpoint satisfaction to learning rather than guaranteeing it by construction.

方法总述由核心洞见句和机制句承担：

> We propose a Path MeanFlow planner that treats task geometry as part of the generative
> space and conditions path generation on terrain observations with an explicit mask. A
> boundary-constrained spline representation satisfies endpoint position and heading
> constraints exactly by construction; demonstration-based route-prior pre-training is then
> followed by adaptation with a differentiable terrain-feasibility objective computed from
> complete terrain available only during training.

机制不再逐句展开：

> 方法一句完成，不再把 Stage 1 与 Stage 2 拆成独立句子；代价名称（禁入区域、
> 倾覆稳定性、曲率）不在摘要中逐项罗列。

开场解决“固定窗口为什么仍会出现观测覆盖变化”的歧义并给出两个困难，方法段落再
按“生成空间 -> 输入条件 -> 训练监督”展开。完整地形只作为训练期目标的来源出现，
不另造抽象概念。

> 基线以 [method_draft.md](./method_draft.md) 为准：Stage 2 是完整地形可微
> 可行性代价在部署时使用的单步输出上直接反向传播；部署为一次网络前向，不包含
> 安全目标输运、critic 或候选选择器。

## 5. 不应采用的摘要写法

- 不按“B 样条 -> 网络 -> MeanFlow -> 三个 cost -> 五个指标”的顺序流水记账。
- `fixed-size` 只描述本文规划器的输入接口，不是 learning-based global planning 的普遍属性；任务句先写崎岖地形规划的要求，再在本文设置中引入固定窗口。
- 不无条件使用 incomplete map 或 partial map；写成“渐进式建图过程中前沿扫描地形
  可能尚未填满固定尺寸输入窗口”，掩码区分已观测与未观测/禁入区域。
- 不把 mask 写成普通第四通道；它区分已观测区域与未观测/禁入区域。
- 不把边界表示写成全文唯一矛盾；它只解决已知任务几何不应被重复学习的问题。
- 不把 Stage 1 和 Stage 2 写成两个训练步骤名称；需要说明前者提供 route prior，
  后者利用完整地形进行物理塑形。
- 不把最终生成器称为 multimodal。Stage 2 没有模式保持目标，而且实际训练会显著
  收缩 Stage 1 的路线覆盖；source noise 仍然存在不等于保留了多模态分布。
- 不把 UAV 建图或局部规划器写成本文算法贡献，除非后续确实纳入联合设计。
- 不把无人机作为摘要的开头或主角；它在本文中只是观测来源的实现方式，摘要应
  从规划问题本身切入。
- 不使用 support-conditioned、planning support、规划支撑 等不直接对应实现的
  抽象词；术语应能对应到固定尺寸窗口、mask 通道、边界控制点、完整地形代价等
  具体机制。
- 不把“完整地形提供特权监督”当作口号；直接写完整地形图只用于训练期计算禁止
  区域、稳定性和曲率可微代价。
- 不把部署句写成系统管道；“一次前向、无在线优化、端点由构造保证”才是承诺。
- 三个代价项要么在问题段先引出，要么在摘要中合并为 terrain-feasibility costs，
  不在括注里生硬堆叠。
- 避免 “endpoints hold by construction”“forbidden clearance”“arbitrary
  coverage”“obstacles here” 这类生硬或口语化搭配；用 “endpoint constraints are
  satisfied exactly”“clearance”“varying coverage” 等自然表达。
- 方法总述不能写成“结合 A、B、C”的平铺；应以“不同信息放到相应环节”的平行结构
  给出，三个分句各带实义动词，覆盖轨迹表示、条件生成与两阶段训练。
- 中文避免“承载”“净距”“观测覆盖形态”等非常用搭配；分别改为“可用的地形观测”
  “禁入区域距离”“不同程度的地图覆盖”，稳定性术语用“稳定裕度”。
- 中文避免“推进到”“部署端点”“参考路径”等直译搭配；分别改为“先……再……”
  “部署时使用的单步输出”“全局路径”。
- “两阶段”只修饰训练，不写“two-stage planner/两阶段规划器”，避免被读成串联的
  双网络结构。
- 代价由部署时使用的单步输出计算并反向传播，不写“反传至单步输出”。
- 不单独写部署句“一次网络前向”；单次前向是 MeanFlow 的固有属性，无需在摘要中
  说明。
- 英文输出用 global path，不用 reference path（避免与 Stage 1 的示范轨迹混淆）。
- 英文摘要不写 Stage 1/Stage 2，用 pre-training / post-training 表达两阶段训练。
- 摘要不写“以部署时使用的单步输出”，统一写“计算可微特权代价并反向传播”。
- 困难句只描述“前沿扫描地形可能尚未填满固定窗口”，掩码出现在方法句（区分已观测/
  未观测/禁入栅格），不把解法写进困难。
- 方法句中的掩码表述避免“掩码化地形观测区分……作为路线生成条件”的缠绕句式；
  写成“路线生成以地形观测和掩码为条件，掩码区分已观测栅格与未观测或禁入栅格”。
- 方法句中不写“上下文约束的样条表示让网络只生成……”，写成“端点位姿由上下文
  约束的样条表示解析确定，网络只需生成内部路线几何”。
- 开头把要求嵌进困难句，避免 is required to / are difficult to satisfy 等低信息量
  短语；用 incomplete terrain evidence、learned rather than by construction
  等高密度表述。
- 起终点位姿用 prescribed，不用 given，强调由任务规定。
- 第一句不用 “remains/ is feasible … on the ground/terrain”；写成 “is traversable”，
  前面已有 “rough terrain”，句尾不再重复地形。
- “扫描地形”写成“前沿扫描地形 / frontier-scanned terrain”，对应无人机前沿建图
  语义，不直接用 scanned。
- 英文不用 demonstration path（可能被理解为示教轨迹数据集），统一用 expert routes。
- 结果句先写假想结论占位：预期结论是“特权后训练提升部署路径的硬可行性”，端点
  约束是表示构造保证、在不同覆盖程度下成立，而不是实验“保持”出来的结果；实验
  冻结后替换为实际数字与结论。
- 摘要逻辑链保持：任务需求 → 两个困难（不掺解法）→ 方法逐一回应 → 训练动机
  （完整地形仅训练期可见）→ 预期结论。
- 首句与困难句之间加桥接：英文 “meeting both requirements is difficult for two
  reasons”，中文“这两个要求并不容易同时满足”，避免跳变。
- 困难二的原因从句用 since 引导（不用 because），并用 “its satisfaction depends
  on learning” 表达（避免重复前面已用的 leaving）。
- 困难表述用 Learning-based planners face two challenges；避免 “only partially
  filled”“leave … unresolved”等口语化/模糊搭配。
- 两个困难不用 First/Second（其一/其二），用“冒号 + 分号 + and/；”自然并列；
  困难二的违例前置只用于英文强调，中文按因果自然顺序（原因在前、违例在后）。
- 中英不必逐句同序：中文按话题-述题组织（如“全局路径规划需要在……约束下生成
  ……”），英文保持主谓宾结构；两版各自自然，不做镜像翻译。
- 两个困难拆成 First/Second（其一/其二）两句，端点上下文句不加 “usually
  supplied to the network” 等冗余修饰。
- 部分观测问题写为“在渐进式建图过程中，前沿扫描地形可能尚未填满固定尺寸输入
  窗口，掩码区分已观测与未观测/禁入区域”；不要写成“往往填不满”或“填充范围随
  建图进度变化”——问题是在建图过程中会出现填不满的阶段，掩码因此成为必要条件。
- 部署承诺只写“一次网络前向”；随机源是标准高斯 source，不是学习得到的隐变量，
  摘要中不写“隐变量采样”这类易误导的实现细节。
- 不在结果冻结前使用“safer”“superior”“generalizes better”等结果性判断。
- 不在摘要中逐项罗列 forbidden、stability 和 curvature；将其概括为
  differentiable terrain-feasibility objective，具体组成放在 Method。
- 不在摘要中展开 terrain fitting、TailMean、softplus、MeanFlow JVP 或 B 样条
  控制点公式；这些属于 Method。

## 6. 实验结果句待填内容

最终结果句不应罗列所有评价指标。优先选择能够闭合摘要故事的三类证据：

1. 边界表示确实保持端点位置与方向，并相对无结构表示改善学习或有效性；
2. 不同规划支撑覆盖率下的全局规划质量，以及未见地形上的表现；
3. Stage 2 相对 Stage 1 的物理有效性变化及路线覆盖损失，同时报告单步推理时延。

若这些结果尚未在冻结协议上确认，摘要继续保留 `[settings]` 和 `[results]`，
不使用诊断实验代替最终证据。结果填入前的正文
应控制在约 170 词，为最终实验设置和定量结论预留 20--30 词。

## 7. 参考摘要的可借鉴结构

- **DARE：** 可借鉴“部分观测先于生成方法”的顺序，但本文必须进一步说明固定窗口
  与可变空间支撑，不能直接沿用 partial map 的宽泛说法。
- **SPARTA：** 解析结构应被解释为解决具体学习困难的机制，而不是预处理步骤。
- **NeuPAN：** 先给统一物理问题，再让数学表示、网络和系统接口承担明确职责。
- **YOPO：** 清楚区分训练期 privileged information 与部署期有限观测。
- **Motion Planning Diffusion：** 从数据驱动 route prior 的必要性自然引出生成模型。
- **BridgeFlow：** 可借鉴“生成空间应包含已知任务结构”的观点，但不迁移其
  Brownian bridge、OT efficiency 或完整 SE(2) equivariance 主张。

## 8. PathPainter 与 D-VLC 的摘要逐句拆解

### 8.1 PathPainter：中心问题驱动的系统摘要

PathPainter 的摘要不是把所有模块平均分配篇幅，而是沿着“先验价值 -> 转化困难 ->
生成式中间表示 -> 执行接口 -> 证据 -> 意义”推进。其句子功能可以概括为：

| 句子功能 | 摘要中的写法 | 可迁移到 MPT 的原则 |
| --- | --- | --- |
| 建立任务价值 | BEV images provide valuable global priors | 第一行说明输入/问题为什么重要，同时点出规划目标 |
| 收窄核心缺口 | two key challenges remain: how to ... and how to ... | 用平行的 how-to 结构写两个真正不同的困难，不写泛泛的 difficult |
| 给出中心 reformulation | can ... be reformulated as ...? | 方法出现前先给一个可理解的核心问题或洞见 |
| 定义方法输出 | infer the target region, generate a traversability mask, and apply A* | 用一个主语配多个并列实义动词，说明输入到输出的转换链 |
| 补上执行接口 | During execution, ... align ... and mitigate ... | 只有当部署接口是论文主张的一部分时，才单独写执行句 |
| 说明证据范围 | We conduct extensive benchmark experiments ... | 说明验证覆盖什么，不用空泛的 comprehensive evaluation |
| 给出可核验结果 | Using only ... successfully completes ... | 用具体平台、距离、成功率或延迟承载强结论 |
| 提炼总体意义 | This work demonstrates how ... | 最后一行回答这项工作为什么具有迁移价值 |

PathPainter 的 Method 和摘要之间存在严格对应：摘要中的 traversability mask 是
Method 的中间表示，A* 是固定的下游接口，cross-view localization 是执行阶段的
补偿机制。因此，摘要没有把“生成模型很强”作为单独结论，而是说明生成结果如何被
转换为可搜索、可执行的路径。

### 8.2 D-VLC：挑战与机制逐句对齐的摘要

D-VLC 使用另一种压缩方式：先给系统价值，再依次写 rule-based limitation、LLM 的
语言能力、VLM 的视觉扩展和仍未解决的系统缺口，最后用一长句压缩四个机制，再用
结果句收束。其句子功能为：

| 句子功能 | 摘要中的写法 | 可迁移到 MPT 的原则 |
| --- | --- | --- |
| 说明系统收益 | heterogeneous swarms improve efficiency through complementary capabilities | 先说明任务中为什么需要学习系统，而非先介绍网络 |
| 指出旧范式限制 | conventional rule-based methods rely on predefined ... | 用具体依赖解释旧方法为什么不能覆盖目标场景 |
| 引入已有能力 | LLMs introduce ...; VLMs further ... | 若需要技术演进，只写与本文缺口直接相关的两步 |
| 保留剩余缺口 | Nevertheless, existing ... depend on known maps ... | 用 Nevertheless 把“已有能力”与“本文仍要解决的问题”分开 |
| 压缩方法机制 | combines A, B, C, and D, enabling ... | 机制使用并列结构，结果能力用 enabling 从句承接 |
| 给出经验边界 | Experiments across ... show ... | 同时交代场景、模型/基线范围和主要结果 |

D-VLC 的写法提醒我们：摘要中的机制顺序必须与 Introduction 的挑战、Method 的
小节和 Experiments 的 RQ 保持一致。MPT 不应在摘要中先写 B-spline，再跳到 mask，
再突然写 MeanFlow；应该沿着“任务几何、观测条件、路线知识、训练期可行性监督”的
因果顺序展开。

## 9. MPT 摘要的句子合同

MPT 适合使用六句主结构，必要时增加一句部署接口，而不是固定追求某个句子数量：

1. **任务句：** 写 global path planning 的目标和两个必须同时满足的条件。使用
   `must generate ... while satisfying ...`，避免 `faces difficulties in producing`。
2. **缺口句：** 用 `Two factors make this difficult:` 引出两个具体困难；两个分句
   使用相同语法层级，并且只描述问题，不提前写 mask、样条或损失。
3. **洞见句：** 用 `We propose ... that ...` 给出方法和核心表示原则；不要写
   `novel`、`powerful` 或 `effectively`。
4. **表示句：** 单独说明边界约束表示保证什么；这句话只承担解析保证，不要同时塞入
   训练过程。
5. **训练句：** 写路线先验如何获得、完整地形如何进入监督，同时明确
   `available only during training`。
6. **结果句：** 写实验范围、相对基线、指标和条件。结果未冻结时使用 `[settings]`、
   `[metric]` 或 `[results]`，不要用 `better`、`safer`、`generalizes better` 代替证据。
7. **可选部署句：** 仅当一次前向、无在线优化或下游执行接口是明确贡献时加入；否则
   把部署信息边界并入训练句，避免摘要尾部变成工程管线清单。

当前候选摘要的句法链是：

> `must generate` -> `Two factors make this difficult` -> `We propose` ->
> `satisfies ... by construction` -> `is pre-trained ... and then adapted ...` ->
> `Experiments show`。

这条链每句只有一个主要动作，句间由因果和对比连接，而不是由模块名称连接。

## 10. 学术英文的措辞界定

| 语义功能 | 推荐表达 | 使用边界 |
| --- | --- | --- |
| 任务要求 | `must generate`, `under prescribed start and goal poses` | `must` 只描述任务约束，不暗示方法已满足 |
| 条件变化 | `may leave only part of the window covered by observed terrain evidence` | 说明固定窗口和可变覆盖，避免 `partial map` |
| 方法原则 | `treats task geometry as part of the generative space` | 解释设计思想，不使用 `naturally` 或 `inherently` |
| 输入条件 | `conditions path generation on terrain observations with an explicit mask` | 明确 mask 是输入条件，不写成轨迹先验自带观测 |
| 解析保证 | `satisfies endpoint position and heading constraints exactly by construction` | 只用于表示确实保证的端点性质 |
| 训练信息边界 | `computed from complete terrain available only during training` | 明确 privileged information 的可见范围 |
| 训练顺序 | `route-prior pre-training is followed by adaptation with ...` | 写训练阶段，不把方法称为 two-stage planner |
| 实证改善 | `improves`, `reduces`, `achieves`, `outperforms` | 必须紧跟指标、基线或实验条件 |
| 证据总结 | `experiments show`, `we validate`, `the results demonstrate` | `demonstrate` 不能替代具体结果 |

尽量使用“主体 + 实义动词 + 对象”的句子。`generate`, `infer`, `condition`,
`encode`, `satisfy`, `adapt`, `evaluate` 比 `is designed to`, `is able to`,
`makes use of` 更紧凑。并列动作保持同一形式，例如
`infer the target, generate the mask, and extract the path`，不要在一个列表中混用
名词、被动句和不定式。

## 11. 当前摘要的具体修正

- `faces two difficulties in producing` 信息密度低，已改为以任务要求开头的
  `must generate ... while satisfying ...`。
- `partial terrain observation` 容易被理解为输入尺寸变化，已改为固定窗口中只有
  部分区域被 observed terrain evidence 覆盖。
- `endpoint violations persist` 把待验证现象写成了无条件事实，已改为
  `endpoint constraints are difficult to enforce when ...`。
- `the trajectory prior inherently encodes the boundary poses` 的责任主体不准确；
  应由 boundary-constrained representation 保证端点，而不是由 route prior 保证。
- `privileged costs` 过于像代码标签，已改为完整地形上计算的
  `differentiable terrain-feasibility objective`，并明确完整地形只在训练期可见。
- 结果句保留“post-training vs. imitation-only training + hard-feasibility metric +
  varying map coverage”的槽位，待冻结实验后填入数字和准确结论。
