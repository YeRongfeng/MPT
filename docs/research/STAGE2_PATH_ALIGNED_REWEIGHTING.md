# Stage 2 Path-Aligned Privileged Reweighting

状态：**已通过探索准入和独立预留环境确认；final test 未打开**  
固定日期：**2026-08-03**

## 方法定义

Stage 2 不修改 Stage 1 的生成分布。给定部署可见输入
`(observed_map, start, goal)`：

1. 冻结的 Stage 1 从 16 个独立高斯 source 生成候选路径；
2. 路径-地图对齐 critic 只根据部署可见地图、起终点和候选几何预测风险；
3. 将 critic energy 转成满足
   `KL(q || Uniform) <= 0.2` 的候选权重；
4. 从该权重分布抽取一条路径作为部署输出。

critic 训练时使用完整地形上的 forbidden、stability、curvature 和连续违反量
作为特权标签。完整 cost map 不能进入 critic 输入；部署时不需要特权地图、
cost 反向传播或在线优化器。

## 冻结实验协议

- Stage 1 checkpoint：`data/path_meanflow/stage1_best.pth`；
- 候选数：`K=16`；
- KL 预算：`0.2`；
- 每个 condition 使用 4 个独立 Stage-2 mask；
- 同一物理 condition 的不同 mask 使用相同 candidate sources；
- 物理环境分区：64 fit、4 internal selection、16 admission validation、
  16 independent confirmation；
- final-test 环境不参与训练、选模、阈值设定或当前报告。

fit、selection、validation 和 confirmation 必须按物理环境互斥；mask 不是物理
环境，不能用不同 mask 冒充跨环境验证。checkpoint 固定取最后一个预注册 epoch，
不得根据 validation 曲线事后挑选 epoch。

## 训练目标

critic 同时预测连续风险和 strict-safe logit。训练损失由三部分组成：

- 连续风险的 Smooth-L1 regression；
- strict-safe 的加权二元分类；
- 同一 condition 内的 strict-first、连续风险次级 pairwise ranking。

候选归档保存部署输入和特权标签，但 formal trainer 明确分离两者。加载归档时
必须校验 Stage 1 哈希、候选数、mask 参数、车辆半径和环境 manifest。

## 准入结果

探索 admission validation：

- uniform safe mass：`18.55%`；
- weighted safe mass：`24.20%`；
- 增益：`+5.64 pp`，environment bootstrap 95% CI
  `[+2.53, +8.85] pp`；
- oracle gain retention：`53.4%`；
- strict-to-invalid：`0%`；
- gate：`PASS`。

冻结 critic 后，在未参与训练和选模的 16 个 confirmation 环境上：

- uniform safe mass：`16.31%`；
- weighted safe mass：`22.43%`；
- 增益：`+6.12 pp`，environment bootstrap 95% CI
  `[+2.95, +9.70] pp`；
- oracle gain retention：`62.7%`；
- strict-to-invalid：`0%`；
- ESS/K 最小值：`0.634`；
- mode coverage retention：`94.3%`；
- gate：`PASS`。

这些结果支持将方法接入正式代码，但不等同于 final-test 结论。

## 正式代码与资产

- 训练入口：`train_flow.py --workflow stage2`；
- 历史训练实现：`legacy/stage2_reweighting/stage2_reweighting_training.py`；
- 历史 critic、KL solver 和部署组合器：`legacy/stage2_reweighting/stage2_reweighting.py`；
- 已确认 critic：
  `data/path_meanflow_stage2_reweighting/stage2_critic_confirmed.pth`；
- 确认元数据：
  `data/path_meanflow_stage2_reweighting/stage2_method_confirmed.json`。

`stage2_critic_last.pth` 保存 optimizer、Torch/CUDA RNG 和 DataLoader generator
状态，可严格续训。`stage2_critic_best.pth` 仅在冻结 gate 通过时生成，不包含
optimizer 状态。

## 历史结果边界

D0 pure direct-cost 的 paired-regression gate 仍是 `FAIL`。更早的 candidate
selector admission 也仍是 `FAIL`。当前结果来自不同的路径-地图对齐 critic、
KL 有界权重、环境互斥协议和独立 confirmation，属于新的独立正证据，不能用于
事后把旧实验改判为通过。
