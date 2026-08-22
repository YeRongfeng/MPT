# Uneven Planner

## 原版 Uneven 在表里扮演什么角色

回答：省掉逐实例地形优化以后，可行性大概少多少。

原版 Uneven 是崎岖地形上的起终点轨迹优化。它读完整地形，必须标成
**Privileged**，不能和只看部分观测的方法排总名次。

## 原版复现要求

原版应使用上游仓库 `/home/sdu/reference_project/gaofei_uneven_planner_ws/uneven_planner`
中的 ROS `KinoAstar + ALMTrajOpt` 链路，在同一张 20 m 地图和同一组
start/goal pose 上重新求解。原版输出再转换为统一评价器需要的二维路径。
不能把当前 fork 中的 `isOccupancy` / 连续 stability validation 改动带入
原版结果。

原版源码副本位于 `baselines/third_party/uneven_planner_original`，一次性 ROS
入口位于 `uneven_baseline_cli`，输出和计时写入
`baselines/evaluation_results/`。外部参考工作区不参与构建写入。

## 当前接入

入口已接到统一评价器：

```bash
source /opt/ros/noetic/setup.bash
source baselines/uneven_planner_ws/devel/setup.bash
.venv/bin/python -m baselines.evaluate \
  --method uneven \
  --dataFolder data/dataset1 \
  --split val \
  --environments 1 \
  --paths-per-env 1 \
  --mask-mode full
```

Original Uneven 是 privileged baseline，必须使用完整高程；评价入口会拒绝
`--mask-mode partial`。每个环境的高程先转成 PCD，原版 `UnevenMap` 再按
自身的邻域拟合和 `SE(2)` 地形构图流程处理。

`env000000/path_0` 的 smoke 已找到路径，前端 200 点，ALM 外层 8 次收敛，
端到端约 2.96 s；统一稳定性检查未通过，规划区域、曲率和终点 yaw 通过。
该结果仅为单任务诊断，不是全量验证结果。
