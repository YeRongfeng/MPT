# 网络结构候选审计

状态：**网络结构未冻结；当前优先确认条件注入机制，禁止进入 Stage 2**  
日期：2026-08-13

本文只讨论当前 Boundary-Constrained Path MeanFlow 的生成器结构。它不改变
一阶边界约束 B-spline 表示、44-D free path coordinates、Path MeanFlow 目标、
部署端点 `(t=1,r=0)` 或 Stage 2 准入协议。

## 1. 当前实现审计

当前生产类是 `dit.Models.PathMeanFlowTransformer`。默认配置为
`n_layers=6, n_heads=8, d_model=512, d_inner=1024`，输入为四通道
`100x100` map，路径状态为 `22x2`。当前 forward 包含：

1. 四级地图 CNN，将输入依次压到 `50x50 -> 25x25 -> 12x12`，最后保留
   `144` 个 map tokens；
2. 独立的 `ImplicitGuidanceEncoder`，用进度、起终点和时间 query 读取 map；
3. 每个 `TrajDiTBlock` 同时执行 path self-attention、map cross-attention、
   FFN，并为三条分支各生成一组 AdaLN shift/scale/gate；
4. 一个较宽的条件 MLP，将 `t`、`t-r`、start、goal 四个 `512-D` 向量先扩展到
   `5*512` 再压回 `512`；
5. 两层 path patchify 和两层 prediction head。

按当前张量尺寸静态计算，默认模型约 **50.8M** 参数。其中 6 个主 block 约
`33.1M`，guidance encoder 约 `6.3M`，条件 MLP 约 `6.6M`，地图 CNN 约
`3.5M`。因此容量主要来自宽度和重复的条件路径，而不是 44-D 输出空间。

另外，地图 CNN 使用 `BatchNorm2d`，而 Stage 2 默认 context batch 为 4；这会
使 batch statistics 成为额外的训练/推理变量。当前实现同时保留了多个历史类和
radial 兼容接口，不能把这些兼容代码当作最终方法结构。

## 2. 受测的最小生成器候选

本轮建立了一个 `compact` 候选进行结构验证，但它同时改变了容量和条件路径，
因此不能作为机制结论：

| 部件 | 建议配置 |
|---|---|
| 路径 token 数 | `22` 个，每个 token 表示一个二维 free coordinate |
| token width | `d_model=256` |
| 主干深度 | `n_layers=4` |
| attention heads | `n_heads=4` |
| FFN width | `d_inner=768`，即 `3*d_model` |
| map input | 4 通道：masked normals + planning-support mask |
| map encoder | 3 个 stride-2 Conv-GN-SiLU block，最后用 `1x1` 投影到 `256-D` |
| map tokens | 约 `13x13` 的空间 tokens；保留二维 sinusoidal position encoding |
| path/map fusion | **只在主干前做一次 path-to-map cross-attention** |
| path backbone | 4 个 path-only DiT block，每层为 self-attention + FFN + AdaLN-Zero |
| output | 逐 token `256 -> 128 -> 2`，直接预测 `y_0` |
| output constraint | 保持当前无约束 free coordinates；不启用 radial output 或历史 zero-sum projection |

实际实现为 **5.64M** 参数，比当前默认模型减少约 89%。地图仍以空间 tokens
输入，因此不会把
planning-support mask 压成一个无法定位局部阻塞的全局标量。

## 3. 条件注入方式

只保留一种全局条件通路：

\[
c = \operatorname{MLP}\bigl([
e_t(t),e_t(t-r),e_p(S),e_p(G)
 ]\bigr),
\]

其中 `e_t` 是共享的时间嵌入，`e_p` 是共享的起终点 pose 嵌入。`c` 仅通过每个
path-only block 的 AdaLN-Zero 调制进入主干。地图条件只通过前置的一次
cross-attention 进入 path tokens；不再额外建立 guidance query、自注意力 guidance
层或每层重复的 map cross-attention。

这样保留了三个必要语义：

- `t` 与 `t-r` 仍分别可见，满足 Path MeanFlow 的时间接口；
- 起终点位姿仍是网络条件，但边界满足性仍由解析 decoder 保证；
- mask 与法向场仍以空间 map tokens 提供，网络可以定位局部规划支撑。

## 4. 不建议继续保留的设计

- `ImplicitGuidanceEncoder`：它与主干 cross-attention 提供重复的地图读取路径；
- 每层的三路 AdaLN 调制：对当前 22 个短序列过于宽裕；
- `BatchNorm2d`：改为 GroupNorm 或 LayerNorm 风格的无 batch-statistics 归一化；
- `d_model=512` 和 6 层深度：在没有容量不足证据前不应作为默认值；
- `use_radial_output` 及旧 projection：继续留在兼容接口，但不得进入 compact。

## 5. 冻结与迁移规则

1. 先把 `compact` 作为新的结构名称写入 checkpoint metadata，并对现有
   `model_args` 做严格保存；
2. 旧 50.8M checkpoint 只作为历史/基线资产读取，不允许以新结构加载；
3. 在未完成结构 smoke test、参数量检查和固定 validation 对照前，不启动长程
   Stage 1 或 Stage 2；
4. 结构验证只比较 compact 与当前默认结构，固定数据划分、损失、优化器、
   mask 语义和随机 source，不打开 final test；
5. 只有 compact 在 Stage 1 的训练稳定性、endpoint 输出和边界解码检查通过后，
   才把它写入正式 Method 的 Experimental Setup。Stage 2 仍保持 validation-gated。

## 6. 验证结果与当前结论

GPU 0 上将 compact 从 update 1200 续训至验证耐心停止（停止于 update 11000，
最佳验证点为 update 9000）。固定源 MeanFlow validation loss 在独立复算中为
`0.1062`；对 20 个与本次 compact 训练集互斥的验证地形、400 个固定上下文进行
部署端点检查后：

| 指标 | compact | 现有生产 Stage 1 checkpoint |
|---|---:|---:|
| Safe@1 | 14.25% | 25.50% |
| Safe@8 | 35.00% | 52.75% |
| curvature hard pass（K=1） | 20.50% | 34.75% |
| forbidden hard pass（K=1） | 92.00% | 90.50% |
| stability hard pass（K=1） | 80.50% | 81.25% |
| endpoint yaw pass（K=1） | 100.00% | 100.00% |

因此此前基于 update 1200 得出的“compact 失败”结论已撤销。长程训练后 compact
的物理指标显著改善，但仍低于旧生产 checkpoint，尚不能冻结，也不得进入 Stage 2。
旧 checkpoint 没有保存可核对的环境清单，且训练预算为 `13750` updates、compact
最佳点为 `9000` updates，故这里只能说 compact 在较充分训练后仍未追平该诊断基线，
不能据此完成严格的同预算架构排名。

另外，最初 1200-update 运行在第一次验证后没有恢复 `model.train()`，导致后续步骤
关闭 dropout；本次续训已修复该问题，但仍继承了这个起点。因此下一次正式比较应从
头按修复后的脚本训练 compact，并为旧模型建立相同的环境划分和训练预算记录。

## 7. 下一步：先确认条件注入

当前不继续做容量插值。`compact` 把模型从约 50.8M 参数降到 5.64M，同时把
`ImplicitGuidanceEncoder`、每层地图 cross-attention 和 12x12 地图读取改成一次
path-to-map 读取；这三个变化无法从一次结果中分离。中等容量试验也已在 update 250
主动停止，只有启动日志和 checkpoint，不构成证据。

下一轮应使用接近生产的固定容量外壳（`d_model=512`、6 层、8 头、`d_inner=1024`），
固定 44-D 表示、MeanFlow 目标、优化器、数据划分和训练预算，只做两个机制因素的
独立对照：

1. 地图条件路径：重复的 progress-aligned guidance tokens 逐层 cross-attention，
   与一次 path-to-map 读取进行对照；
2. 地图 token 分辨率：现有约 12x12 下采样与保留 25x25 空间 token 进行对照。

每次只改变一个因素，并用固定验证集上的 MeanFlow、曲率通过率、strict-valid、
Safe@1 和 Safe@8 共同选点。机制结论确认前，不再讨论减少层数、宽度或 FFN 容量。
