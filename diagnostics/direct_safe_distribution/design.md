# 直接安全分布路线：审计与受控实验设计

## 审计结论

当前主模型是 `PathDiffusionTransformer`，输入地图为
`(normal_x, normal_y, normal_z)` 三通道 100×100 栅格，经 CNN 得到 12×12
地图 token。高程存在于 `map.p` 和 dataloader 返回值中，但不进入当前生成器；
它参与地形数据定义，而 yaw-aware stability/ESDF 由三个法向通道生成。

物理地图使用中心坐标 `[-10,10]²`、分辨率 0.2 m。像素约定为列对应 x、行
对应 y；第一个像素中心是 `(-10,-10)`，最后一个中心是 `(9.8,9.8)`。
`map.p["tensor"]` 的通道顺序是 elevation、nx、ny、nz。pose 和路径是
`(x,y,yaw)`，yaw 用弧度。

训练标签先把原始 100 点路径用三次 B 样条最小二乘拟合为 26 个控制点，再把
中间点除以 `coordinate_scale=10`。完整控制点由真实起点、24 个中间点、真实
终点组成。表示层计算 25 条边，减去均匀弦线边后乘 25，并沿边维投影到零和
子空间。解码时累积弦线边与 `residual/25`，因此终点是结构性固定的。

源变量是 `(B,25,2)` 标准高斯沿边维减均值后的 projected Gaussian。它不是
learned latent。训练是此显式 residual 空间中的 pixel/one-step MeanFlow
目标；论文与代码文档应称为 **Trajectory MeanFlow**。

现有 radial feasible layer 沿零和 residual 方向计算到轴对齐
`[-1,1]²` 的最大半径。它保证的是当前归一化画布边界，不理解旋转后的原地图
多边形。

## Canonicalization 插入点

变换必须发生在坐标归一化、B 样条拟合和 residual 编码之前：

1. 从原始 `(M,S,G,trajectory)` 计算 `theta=atan2(G-S)`。
2. 对 pose、完整训练轨迹和 teacher 控制点应用
   `p_c=R(-theta)(p-S)`，yaw 减 `theta` 并 wrap。
3. 对地图在 canonical 输出栅格的每个像素做逆映射
   `p=R(theta)p_c+S` 后重采样。
4. 在采样位置对 `(nx,ny)` 应用 `R(-theta)`；`nz` 和 elevation 是标量，
   只重采样、不旋转也不减起点高度。
5. 在 canonical 坐标拟合/编码 26 控制点和 25 residual。
6. source residual 是向量场样本：若从全局任务变换则旋转；严格对照训练可在
   canonical frame 直接复用同一 projected Gaussian 数组。
7. 网络输出先在 canonical frame 解码，再对控制点和稠密 B 样条轨迹应用
   `R(theta)` 并加回 S。

`geometry/canonicalization.py` 是唯一几何实现。`enabled=False` 不执行
grid-sample 或算术，直接返回原地图，保证 legacy 路径逐位不变。

## 固定画布与 radial layer 的兼容性

严格的起点原点化不能直接塞进当前 10 m 半边长画布。本数据 20,000 个训练
条件的弦长中位数为 14.06 m，最大 21.15 m，因此大多数 canonical 终点超出
`[-10,10]²`。此外，旋转和平移后的原地图有效域是一个任意放置的旋转方形，
而当前 radial layer 只能约束轴对齐方形。

主 2×2 实验不能通过夹紧 goal 或丢弃长任务来掩盖此问题。受控实现采用共同
画布策略：

- A/B/C/D 都使用同一个固定物理画布、同一个 `coordinate_scale` 和同一个
  100×100 栅格采样密度；
- 画布半边长至少为原地图对角线 28.285 m，从而容纳任意
  `p-S`；
- A/C 做恒等朝向的相同画布嵌入，B/D 做 task-centric 变换，避免只有
  canonical 组承受分辨率变化；
- 栅格外区域使用零法向并保存 `valid_mask`。Oracle 将无效域视为越界，不能
  把 padding 当作安全地形；
- 网络头中的 radial layer 保持原实现和参数，先保证共同画布边界。四组网络
  输出随后统一进入同一个解析部署包装：canonical residual 先旋回全局（global
  组是恒等），按 `canvas_scale/10` 换算为原地图 normalized residual，再调用
  **同一个未修改的** `radial_project_residual`，以原始全局 S/G 和
  `[-10,10]²` 做最终投影。随后用原表示解码。因此 A/B/C/D 的最终控制点仍被
  严格限制在原地图边界；validity/OOB 指标继续审计 raster padding 和曲线。

这一区别必须写入报告：连续点/向量公式严格 SE(2) 等变，离散 raster
grid-sample 只有近似等变。

## 严格数据协议

现有 `data/dataset0/train` 与 `data/dataset0/val` 各自包含全部 100 个相同
map ID；抽查及 hash 证明同 ID 的 `map.p` 相同。旧 checkpoint 的
`model_params.json` 也记录 Stage-1 train/val 都使用 100 张地图。因此新实验
使用单一物理数据根和一个不可变的 map-level manifest：

- 70 train maps：Stage-1 与 safety teacher 唯一可访问的地图；
- 15 validation maps：早停和超参数；
- 15 test maps：只允许最终一次评估；
- split 由 `SHA256(seed|map_id)` 稳定排序生成；
- manifest 保存每张 `map.p` 的 SHA-256、路径数、split 清单及 manifest
  自身 hash；
- 每个入口都通过 manifest 取 env list，不接受目录中的“全部环境”默认值。

现有地图已经生成，因而只能保证训练流程隔离，不能追溯性声称 test 在原始
地形生成阶段未曾存在。最终论文级协议应冻结新的 test 生成 seed 并在训练
结束前不生成/不挂载其文件。

## 安全多模态数据

每个 train 条件固定保存 K=8/16 个 projected Gaussian，并用仅 train maps
训练的 Stage-1 生成 proposal。现有 safety-priority teacher 的双局部起点、
自适应 1/2/3 m 位移信赖域、1.10/1.17/1.25 长度限制、安全最大违规优先和
不跨侧投影原样复用。数据文件区分：

- `strict_safe=true`：可进入 strict-safe 训练集；
- `strict_safe=false, improved=true`：只进入显式允许 improved 的配置；
- 其余只保留审计，不作为安全目标。

逐候选字段为用户指定的 16 项，并额外保存 teacher 配置 hash、split manifest
hash、checkpoint hash 和优化终止原因。条件汇总保存安全数量、三类 mode
计数、是否有安全解及是否有两个安全 mode。

## 2×2 控制

A 原始目标/全局，B 原始目标/canonical，C 安全 teacher/全局，D 安全
teacher/canonical。四组从相同初始化策略完整训练，不从旧 Stage-1 微调。
网络参数量、源数组、batch、epoch、optimizer、学习率、候选 K 和推理函数
完全相同；唯一数据差异是 target 选择与坐标变换。外部 baseline 读取旧
Stage1+Stage2 checkpoint，但不参与四组预算相等声明。

首轮继续当前独立 source-target 配对，不引入 OT。每种子保存 config、
checkpoint、逐候选 parquet/npz、条件汇总和 wall-clock/GPU memory。

## 评价与判定

单候选同时报告 strict Valid Rate、最大 ESDF 违规、safe cost、长度、曲率、
jerk、端点 yaw、自交和异常。集合在 K=1/4/8/16/32 报告 Safe@K、
Best-Cost@K、安全数量、三类 mode 覆盖、去重有效数量、延迟和显存。所有指标
按 train/validation/strict test/随机 SE(2) 分层。

Oracle 只用 privileged ESDF 选离线 best candidate。Safe@32 低时不研究
scorer；Safe@K 高而 Valid Rate 低时才说明排序可能有价值；Safe@K 高但模式
覆盖低时优先诊断 collapse。

等变性脚本对控制点与稠密曲线计算相对 RMS，并分别报告 mode、strict-safe、
cost 的变化和角度曲线。栅格误差必须与连续解析变换误差分开报告。

桥先验诊断不训练网络。对当前 source 使用
`Y_k=(1/n)sum_{i<k}R_i`，理论协方差为
`sigma²/n²(min(k,l)-kl/n)I`；Monte Carlo 报告均值、方差、完整协方差误差，
并与绝对控制点高斯和 waypoint bridge-style noise 对照 source-target 距离、
MeanFlow target vector norm、位置速度幅值、端点、长度与异常率。
