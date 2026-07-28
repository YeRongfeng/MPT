# 零和边增量桥先验诊断

- Monte Carlo 样本：100,000
- `max |E[Y_k]|`：4.343e-04
- 协方差 RMSE：1.890e-05
- 协方差最大绝对误差：5.672e-05
- 内部点方差平均/最大相对误差：0.33% / 0.78%
- 中点经验/理论方差：0.009985 / 0.009984

理论与实证均支持：当前 projected Gaussian edge source 在控制点偏移空间中就是离散 Brownian bridge；端点方差为零、中间方差最大。

## 同一目标数据上的 source 对照

下表先按模型 `coordinate_scale=10 m` 将三种 normalized source 统一换算到物理控制点空间。

| Source | physical CP squared distance | native vector norm | endpoint error | length | abnormal |
|---|---:|---:|---:|---:|---:|
| absolute_control_point_gaussian | 6143.403 | 7.798 | 14.303 | 254.286 | 100.0% |
| independent_waypoint_bridge_marginals | 65.793 | 0.798 | 0.000 | 23.971 | 99.0% |
| projected_edge_increment_bridge | 63.928 | 8.314 | 0.000 | 16.061 | 31.4% |

绝对控制点高斯不固定端点；另外两种桥源固定端点。独立 waypoint 对照只匹配每个位置的边际方差，不匹配跨控制点协方差，因此不能据此 替换当前 source。表中 distance 是物理控制点距离；vector norm 是各表示真正训练时的 native MeanFlow 状态范数（绝对/waypoint 为 normalized control state，当前方法为 25× scaled residual），两者不能混作同一单位。逐控制点物理速度 RMS 保存在 JSON 中。
