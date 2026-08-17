# Method Figure Reference v1

这份资料用于交给网页 GPT 或其他绘图模型生成 Method 总览图。图中只应呈现
当前 Method 已定义的数据流和训练/部署边界，不应根据常见扩散模型自行补充模块。

## 一句话方法概括

Path MeanFlow 是一个面向崎岖地形全局路径生成的条件生成器：它用边界约束 B 样条
把给定的起终点位置和朝向写入路径解码器，再通过“专家路线学习 -> 特权多目标地形
适配”的两阶段训练，把长程绕行结构和地形物理目标结合到一次前向路径生成中。

## 总体数据流

### 部署可见输入

规划器每次 replanning invocation 接收：

1. 固定大小窗口中的当前部分地形观测；
2. 四通道观测表示 `X_obs = [normal_x, normal_y, normal_z, mask]`；
3. 起点位姿 `S=(x_s,y_s,psi_s)` 和终点位姿 `G=(x_g,y_g,psi_g)`；
4. 一个标准高斯 source `xi ~ N(0,I)`，作用在内部自由路径坐标上。

`mask=1` 表示该位置已经观测且车辆中心允许进入；`mask=0` 表示未知或已知禁入，
两者都不作为可靠规划支撑。输入窗口尺寸固定，但 mask 支撑区域可以是不规则的子区域。

### 训练信息边界

完整窗口地形 `X_full` 只在训练阶段用于稳定性/倾覆目标和离线审计。它不是生成器
的输入，也不能连接到部署分支。训练图中应将 `X_full` 画成一条虚线旁路，只进入
`rollover stability` 目标。

## 两项核心设计

### 1. Boundary-aligned trajectory representation

规划任务先被变换到由起终点定义的任务规范坐标系，减少平移、旋转和统一尺度差异。
路径采用一阶边界约束 clamped cubic B-spline：起点和终点的位置以及两端一阶方向由
四个边界控制点解析固定，网络不预测这些已知边界量，只预测中间控制点的自由坐标。

生成过程可以表示为：

`standard Gaussian source -> generator predicts free path coordinates -> analytic
boundary-aligned B-spline decoder -> geometric global reference path`

解码后的每个样本都属于给定端点和航向的边界路径空间。该表示不等于地形安全保证；
规划支撑、稳定性和曲率性质仍由训练目标塑形并由实验评价。

### 2. Two-stage learning

**Stage 1: Expert Route-Prior Learning (ERPL)**

专家路径先被转换到同一个边界对齐的自由坐标空间。Path MeanFlow 从标准高斯 source
学习在当前地形观测和任务位姿条件下的路线分布，使模型获得长距离绕行和到达目标的
整体路线结构。该阶段的核心作用是学习路线先验，不应在图中宣称已经保证全部物理约束。

**Stage 2: Privileged Multi-objective Terrain Adaptation (PMTA)**

PMTA 从 Stage-1 生成器初始化，并在与部署一致的 endpoint `(t,r)=(1,0)` 上生成路径，
然后分别计算三个路径级目标：

- `planning-support violation`：路径是否离开当前可靠规划支撑；
- `rollover stability`：路径上的车辆姿态是否接近或进入倾覆风险区域；
- `curvature violation`：路径是否超过车辆转向能力。

三个目标保持独立，不先固定压成一个全局加权和。MGDA 在每次更新时协调它们的参数
梯度，并更新同一个生成器。完整地形只为训练期稳定性监督提供特权信息；部署时仍只
使用 `X_obs`、mask、起终点位姿和 source。

## 网络结构信息（适合画成小插图）

网络细节只能画到下面这个层级：

1. `X_obs` 的四个通道进入卷积地形编码器，得到空间 `map tokens`；
2. MeanFlow 状态 `z_t` 按内部控制点组织为 `path/query tokens`；
3. 归一化起终点位姿、MeanFlow 时间 `t` 和时间差 `t-r` 组成全局条件，调制路径主干；
4. 任务/时间条件化的 progress queries 先通过 cross-attention 读取 `map tokens`，
   形成路径对齐的地形引导；
5. 路径主干再次读取这些 guidance tokens；
6. 逐 path-token 的预测头输出二维 free path coordinates；
7. 输出交给解析 B-spline decoder，得到二维几何全局参考路径。

推荐在网络小插图中使用以下简化链路：

`masked terrain (4 ch) -> CNN terrain encoder -> map tokens`

`free-coordinate state + task poses + (t, t-r) -> path tokens -> cross-attention with map tokens -> coordinate head`

`predicted free coordinates + fixed boundary controls -> analytic cubic B-spline decoder -> path`

层数、宽度、注意力头数、具体激活函数和优化器参数不属于总览图必须信息；除非另画
网络结构附图，否则不要凭空标注这些数值。

## 推荐版式

画一张横向、上下两带的图：

- 上带标题 `OFFLINE TRAINING`：`Expert routes -> ERPL -> PMTA (three objectives + MGDA) -> Adapted generator`；
- 下带标题 `ONLINE PATH GENERATION`：`Masked terrain + task poses + Gaussian source -> Adapted generator -> Boundary-aligned representation -> Generated path`；
- 用橙色突出边界约束 B 样条模块，用红色或砖红色突出 PMTA，用蓝色突出 ERPL，用青绿色突出部署生成器；
- `complete terrain - training only` 用虚线接到 `rollover stability`，并在图上明确断开部署连接；
- 用同一个 adapted generator 表示训练后的部署模型，不画 teacher/student 两个网络；
- 输出路径叠加在与输入相同的 masked terrain thumbnail 上，显示起点、终点和两端航向箭头。

## 必须避免的误解

- 不要把 mask 画成独立的第三项方法贡献；它是部署输入和信息边界。
- 不要把完整地形画进部署输入；生成器部署时看不到 `X_full`。
- 不要把 source 画成 Brownian bridge、扩散多步去噪链或在线采样循环；这里是一次 endpoint forward pass。
- 不要画在线 ESDF/cost evaluation、轨迹优化、候选路径排序或 best-of-K 选择。
- 不要把 B-spline 模块画成单独的网络；它是解析解码器。
- 不要把输出称为带物理时间的 timed trajectory；输出是二维 geometric global reference path。
- 不要添加 UAV、Mars、wolf、raven 等故事元素到技术 pipeline 图中；这些内容如需使用，应放在 Introduction 的背景图，而不是 Method 图。

## 可直接粘贴给网页 GPT 的英文提示词

Create a clean, wide, publication-quality two-band method overview for a rough-terrain global path generator. The two core contributions are (1) a boundary-aligned first-order clamped cubic B-spline representation and (2) two-stage learning: Expert Route-Prior Learning (ERPL) followed by Privileged Multi-objective Terrain Adaptation (PMTA). The mask is only an input condition and information boundary, not a third contribution.

Top band, titled “OFFLINE TRAINING”: expert route demonstrations -> ERPL learns long-range route structure -> PMTA adapts the same generator with three independent path-level objectives: planning-support violation, rollover stability, and curvature violation. Show MGDA gradient coordination below the three objectives. Draw a dashed input “complete terrain - training only” entering only the rollover-stability objective. Then output one “adapted generator”.

Bottom band, titled “ONLINE PATH GENERATION”: masked partial terrain observation in a fixed-size window + start/goal poses + one standard Gaussian source -> adapted generator, one endpoint forward pass -> boundary-aligned cubic B-spline decoder -> generated 2D geometric global reference path. The decoder analytically fixes endpoint positions and headings with boundary control points; the network predicts only free interior control-point coordinates. Overlay the output path on the same masked terrain thumbnail and show start/goal heading arrows.

Use a white background, flat vector-like academic style, restrained blue/teal/orange/brick-red palette, dark arrows, readable sans-serif labels, and no decorative scenery. Do not add a UAV, Mars, wolf, raven, Brownian bridge, diffusion denoising chain, online optimizer, ESDF cost loop, best-of-K selector, teacher/student pair, runtime safety certificate, or invented encoder/attention submodules. Do not connect complete terrain to deployment. Keep arrows left-to-right and make the training-only information boundary visually unmistakable.
