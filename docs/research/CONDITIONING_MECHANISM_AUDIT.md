# 条件注入机制审计

状态：**机制候选，尚未进入训练；网络容量暂不作为本轮变量**  
日期：2026-08-13

## 1. 图中方法的条件信息流

图中最重要的设计不是某一个模块名称，而是条件信息的职责分离：

1. `Start & Goal` 与 `Obstacle Points` 先由 `Task Encoder` 编码为一个明确的
   task latent (M)。这个 latent 是任务条件，不是路径序列中的伪位置 token。
2. Primitive generator 接收 (M)、Gaussian source、time embedding 和
   positional encoding，Transformer decoder 通过 SA/CA 生成 primitive path。
3. 下方的 Point Transformer 同时处理真实点云和生成的 keypoint sequence，使用
   positional encoding、SA 和 CA 融合局部空间信息与任务 latent。
4. 轨迹优化器位于神经网络之后，负责模型外的物理修正；它不能被当作网络条件
   注入机制本身。

因此，图中没有把“路径系数编号”解释成物理行进进度，也没有把同一个由地图和任务
混合得到的静态 guidance 序列重复送入每一层作为唯一地图上下文。

## 2. 我们当前实现的实际信息流

当前生产 `PathDiffusionTransformer` 在 [dit/Models.py] 中执行：

1. 四通道地图经过四段 CNN 和池化，最终只保留约 `12x12=144` 个地图 token。
2. `ImplicitGuidanceEncoder` 用可学习的 `22` 个 `progress_embed` 作为 query，
   同时接收 start、goal、(t)、(t-r)，再对地图 token 做一次 CA，得到 `22` 个
   guidance token。
3. noisy path 的 `22x2` free coordinates 被映射为 `22` 个 path token。
4. 每个 `TrajDiTBlock` 都对同一组 guidance token 做 CA；全局
   ([t,t-r,S,G]) 又通过 AdaLN 调制每一层。

这不是图中那种“任务 latent + 空间条件”的清晰分工，而是：

\[
  (M_{\rm map},S,G,t,t-r)
  \longrightarrow
  H_{\rm guidance}^{22}
  \longrightarrow
  \text{每层重复 CA},
\]

同时再走一条全局 AdaLN 条件支路。

## 3. 当前机制的可疑点

### 3.1 `progress_embed` 的语义可能错误

`22` 个 free coordinates 是 B-spline 内部自由变量的坐标索引，不是沿路径采样的
`22` 个空间点，也不是弧长进度。将它们命名为 progress-aligned guidance token
会把“系数序号”误当成“物理位置”。这样得到的 guidance token 可能具有稳定的
索引模式，却没有可靠的局部障碍对应关系。

### 3.2 起终点和时间被重复注入

start、goal、(t)、(t-r) 先调制 guidance query，再经过 `cond_mlp` 调制每个
DiT block。两条支路的功能没有明确分工，网络可以学习到重复或相互冲突的条件映射。

### 3.3 地图信息被过早压成静态 guidance

地图先被 guidance encoder 读取一次，之后所有 block 看到的是同一组已混合的
guidance token，而不是直接看到保持空间布局的地图 token。path query 虽然在每层
变化，但 key/value 已经失去原始地图 token 的清晰语义。

### 3.4 `12x12` 下采样可能抹掉窄障碍和 mask 边界

100x100 输入经过三次 stride-2/max-pool 后变成约 12x12。对于规划 support mask，
这会把窄的禁入区域和局部可通行边界混入同一个低分辨率 token。图中的 Point
Transformer 保留点级局部特征，不能直接支持我们把地图压到 12x12 的选择。

### 3.5 compact 结果不能单独归因于容量

compact 同时移除了 guidance encoder、改成单次 map read、改变地图下采样，并将
容量从约 50.8M 降到 5.64M。它的长程 Safe@1 改善到 14.25%，但仍低于旧生产诊断
checkpoint 的 25.5%；这只能说明当前组合尚未被验证，不能说明哪一个机制负责差异。

## 4. 推荐的最小机制

优先测试 **直接空间地图 CA + 全局任务 AdaLN**。它保留图中清晰的 SA/CA 结构，
但不引入新的抽象模块：

### 地图条件

地图 CNN 输出带二维位置编码的空间 token (H_M\in\mathbb R^{N_M\times d})。
第一轮机制实验固定网络容量，只比较 `12x12` 与 `25x25` 的 (N_M)。

### 任务条件

将 start、goal、(t)、(t-r) 通过一个 MLP 得到单个全局向量：

\[
  c=\operatorname{MLP}([e_t(t),e_t(t-r),e_p(S),e_p(G)]).
\]

这个向量只用于 AdaLN；地图不再进入这个向量，progress token 也不再作为伪路径
进度使用。

### 每层数据流

对每一个 path block，保持明确的三步顺序：

\[
\begin{aligned}
X &\leftarrow X+\operatorname{SA}(\operatorname{AdaLN}(X,c)),\\
X &\leftarrow X+\operatorname{CA}(\operatorname{AdaLN}(X,c),H_M,H_M),\\
X &\leftarrow X+\operatorname{FFN}(\operatorname{AdaLN}(X,c)).
\end{aligned}
\]

这里 path token 只表示 free-coordinate index；它不再被称为 progress token。每层
直接访问空间地图 token，path query 可以根据当前 noisy path 状态重新选择地图
区域，地图位置语义也不会被 guidance encoder 提前混合掉。

## 5. 已实现的首个机制候选

代码中新增 `SpatialMapPathMeanFlowTransformer`，它保持生产模型的地图 CNN、
`6` 层主干、`8` 头和 `512` 宽度，只改变条件路径：

- 删除 `ImplicitGuidanceEncoder` 和 `progress_embed`；
- 保留带二维位置编码的 `12x12` map tokens；
- 每一层由当前 path tokens 直接 query map tokens；
- start/goal/time 只经过 `cond_mlp` 进入 AdaLN；
- 保持当前 44-D free path coordinates、MeanFlow 接口和单次部署端点。

该候选已通过 GPU smoke test：forward、backward、map token shape 和边界解码
均正常（`tests/test_compact_path_meanflow.py`, 3/3）。随后在固定 80/20 terrain
split、400 个 validation contexts、同一 source/optimizer/early-stopping 规则下
完成了 GPU 0 Stage 1 训练：最佳 update 为 `7750`，固定 MeanFlow validation loss
为 `0.1116`（独立复算）。

## 6. 最小可验证对照

不引入新容量搜索，不使用版本编号，先固定接近生产模型的容量、数据、优化器、
source、验证划分和训练预算，只做以下机制对照：

| 候选 | 地图条件 | 全局条件 | 目的 |
|---|---|---|---|
| 当前生产机制 | `12x12` guidance tokens，每层 CA | guidance 与 AdaLN 重复注入 | 现有基线 |
| 直接空间 CA | 原始空间 map tokens，每层 CA | 只在 AdaLN 注入 start/goal/time | 检验 guidance token 是否造成语义错位 |
| 直接空间 CA + 高分辨率 | `25x25` 原始空间 map tokens，每层 CA | 同上 | 检验地图下采样是否损失局部约束 |

第一轮只需要比较第二项与当前生产机制；第三项在第二项机制成立后再运行。评价
必须同时保留 MeanFlow、endpoint loss、曲率通过率、strict-valid、Safe@1 和
Safe@8。Safe@8 仅作覆盖诊断，不能替代部署主指标 Safe@1。

本轮第一项对照已完成，且两者使用同一 `12x12` 地图下采样：

这里的“直接空间 CA”特指生产宽度的 `SpatialMapPathMeanFlowTransformer`，参数量
约 `44.5M`；它不是前文的 `compact`（约 `5.64M`）。此前将这两类结果都简称为
“直接 Map CA”是不准确的。

| 指标 | 直接空间 CA | 现有生产 guidance |
|---|---:|---:|
| Safe@1 | 23.00% | 25.50% |
| Safe@8 | 47.50% | 52.75% |
| 曲率通过率（K=1） | 33.25% | 34.75% |
| 禁入区通过率（K=1） | 91.50% | 90.50% |
| 稳定性通过率（K=1） | 79.25% | 81.25% |

这组结果不能证明直接空间 CA 优于生产 guidance，但它已经接近生产基线，且
没有引入伪 progress token，支持将其作为更清晰的条件机制候选。由于地图下采样
在两边完全相同，本轮结果不包含任何关于 `12x12` 是否足够的判断。

## 7. 当前判断

最值得优先验证的不是“再减少多少层”，而是：

> path token 应直接查询带空间位置的地图 token；start/goal/time 作为全局任务
> 条件调制主干；不要用 B-spline free-coordinate index 构造伪 progress guidance。

这个改法比当前 guidance encoder 更容易解释，也比 compact 的一次性融合更容易
定位失败原因。它仍然是候选机制，未经训练验证前不写入正式 Method，也不启动
Stage 2。
