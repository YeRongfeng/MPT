# Stage-2 coupling 受控实验结论

## 实验边界

- 固定 `data/sim/stage1_best_model.pth`、当前 DiT、12×12 地图编码、
  条件注入、B 样条、零和边残差和径向可行层。
- 完全关闭 FIM、dropout、weight decay 和学习率调度。
- Stage-2 训练地图：`env000015/env000040/env000064`。
- Stage-2 held-out 地图：`env000068/env000085`，与训练地图无交集。
- 这里的 held-out 仅指未参与 Stage-2 微调；当前 Stage-1 checkpoint 的配置
  显示它训练过 dataset0 的全部 100 张地图，因此这不是“Stage-1 从未见过的
  新地形”测试。
- 每张地图选 3 个高风险条件，每个条件固定 4 个源噪声；训练/held-out
  共 36/24 个 source–trajectory 样本。
- C 使用同一条件内的固定循环错排，不跨地图、不跨起终点；D 保留
  `(noise_i, R_i*)` 一一配对。

## Teacher 有效性

硬信赖域 teacher 优化 100 步：

- 100% 样本满足左右侧不跨零、路径长度比不超过 1.17、最大控制点位移
  不超过 2 m；
- 100% 样本的物理 safe cost 均有改善；
- safe cost 中位数：训练集 `5.417 -> 2.069`，held-out
  `5.695 -> 1.743`；
- teacher 的严格三类模式身份保持率为 91.7%；信赖域约束的是“不跨到
  另一侧”，靠近阈值时允许 left/right 变为 near-straight。

因此 R* 是有效的局部改进目标，但在当前最难条件上 `safe@4` 仍为 0：
teacher 只降低了风险，并未生成完全零危险点的候选。

## 主要结果

### 250-step 随机 coupling 与优化诱导 coupling

| 指标 | C 随机，train | D 诱导，train | C 随机，held-out | D 诱导，held-out |
|---|---:|---:|---:|---:|
| safe cost 中位数 | 2.744 | 2.668 | 5.247 | 5.531 |
| Recovery 中位数 | 0.740 | 0.720 | 0.159 | 0.048 |
| 相对 teacher 误差 | 0.830 | **0.539** | 1.007 | 1.014 |
| 模式保持率 | 80.6% | **86.1%** | 70.8% | **79.2%** |
| best-of-4 safe cost | 2.345 | **1.952** | **4.803** | 5.220 |
| 条件内多样性 RMS | **0.477 m** | 0.407 m | 0.356 m | **0.431 m** |
| 异常率 | 0% | 0% | 4.2% | 4.2% |

D 在训练条件上明显学到了实例级配对：相对 teacher 误差、模式身份和
best-of-K 都优于 C。但它没有在 held-out 地图上获得更高 Recovery 或更低
cost。因此当前结果只支持“optimizer-induced coupling 提供了可学习的
候选身份监督”，不支持“它已经带来跨地图物理收益”。

### cost-only 参考

此前完成的同数据 B（400 步）输出，用本次 100% 有效 teacher 重新计算
Recovery：

- train：safe cost `0.916`，Recovery `1.133`，模式保持 77.8%；
- held-out：safe cost `5.473`，Recovery `0.271`，模式保持 83.3%。

B 能在训练地图强力降低 cost，但 held-out 收益很小，并伴随训练集模式身份
下降。这个结果与“网络可表达、优化梯度有效，但跨地图修正方向泛化不足”
一致。

## 可证伪判断

1. **D 是否优于 C？**  
   在训练条件上的候选级监督指标成立；在 held-out 物理指标上不成立。
   因而暂时不能把 optimizer-induced coupling 写成已验证的核心贡献。

2. **失败是否只是网络容量或 cost 梯度问题？**  
   不像。B 在训练地图 Recovery 超过 1，D 也显著降低 teacher 误差，说明
   当前网络能够拟合修正；主要缺口出现在 held-out 地图。

3. **是否应立即修改地图编码？**  
   现在已有进入问题三的证据，但下一步仍应先把配对实验扩到更多地图和随机
   种子。若 D 持续表现为 train teacher-error 明显下降、held-out Recovery
   接近 0，再只增加“沿 R0 轨迹采样的高分辨率局部特征”，与当前 12×12
   编码作单变量对照。

## 产物

- `experiment_stage2_coupling.py`：可复现 A/B/C/D、硬信赖域 teacher 和
  同条件随机错排。
- `stage2_coupling_abcd_trust100/`：完整 100-step 同轮 A/B/C/D。
- `stage2_coupling_C250/`、`stage2_coupling_D250/`：严格同数据的 250-step
  C/D。
- 每个目录包含配置、逐样本 CSV、规范配对 NPZ、模型输出 NPZ、汇总 JSON
  和简表。
