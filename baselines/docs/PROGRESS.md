# 基线复现进度

这份记录用普通语言说明：我们在比什么、已经做完什么、卡在哪里。
技术命令可以回到对应的方法说明页。

当前完整的比较清单和主表/supplementary/diagnostic 分层见
[`BASELINE_MATRIX.md`](BASELINE_MATRIX.md)。

## 我们在做什么

论文要回答的不是“谁在所有指标上都最好”，而是：

> 在崎岖地形上，一次前向生成的全局路径，相对在线搜索/优化，可行性差多少、时间省多少。

所以外部方法必须接到**同一套硬指标**上：规划区域、倾覆稳定性、曲率、起终点朝向。找不到路或超时，一律记为不可行。

当前确定要准备接入的外部 baseline 包括：

1. Original Uneven Planner（privileged 原版 ROS）
2. Neural A*
3. Kicki neural B-spline
4. MPD

此外，Uneven Planner 需要以**原版 ROS 规划链路**单独复现；当前数据生成时
使用的是加入了新的 occupancy/连续稳定性判据的修改版，不能直接当成原版
baseline。

仓库里此前准备过的 T-Hybrid terrain-aware 对照可以作为额外搜索行；TRG
已从 `baselines/` 清除，不替代上面的生成式/学习式 baseline 计划。

当前本地实现/材料的状态是：

1. T-Hybrid A*（已接 ROS smoke）
2. Original Uneven Planner（原版核心已在 `baselines/` 内编译并完成一条 ROS smoke）
3. Neural A*（官方源码已固定，MPT NumPy 转换层已完成；checkpoint 待定）
4. Kicki（官方 `bspline` 源码已固定；MPT 输入适配和 checkpoint 待定）
5. MPD（官方源码已固定；IsaacGym/模型目录和 MPT 适配待定）

Path MeanFlow 的 Stage 1 / Stage 2 权重在 `data/path_meanflow/`，用来和这些方法坐同一张表，不是再训练一遍。

---

## 2026-08-17 第一步：统一评价接口

没有统一接口，各方法各报各的数字，后面没法写论文。

所以先做了 `baselines/common.py`：读同一套 val 任务、同一套 mask、同一套 `trajectory_validity_metrics`。

约定：

- 部分观测时，mask=0 的格子（没看到或不能走）对搜索器就是障碍。
- 完整地形只给 Uneven 这类 privileged 方法，以及离线打分。
- 输出都变成二维路径，再用同一套硬检查。

---

## 2026-08-17 至 2026-08-20：保留的 terrain-aware 对照

当前保留的搜索对照是官方 T-Hybrid ROS 节点。Original Uneven 按原版 ROS 链路
在 `baselines/` 内接入，使用完整高程并
标记 privileged。

---

## 2026-08-17 第四步：Path MeanFlow 接到同一张表

Stage 1、Stage 2 的权重都能被 `load_model` 读进来，表示版本和当前代码一致。

在 val/env000000 的 path_0 上各跑了一次（一次前向，一个 source）：

- 都能出路径，规划区域和端点朝向通过
- 倾覆、曲率未通过
- Stage 1 约 1.3 s（含加载），Stage 2 约 0.7 s

这只是一条任务，不能当成第二阶段失败或成功的结论。它只证明：自己的方法和别人已经走同一套打分。

要用 conda 环境 `vim`，并且加上  
`LD_LIBRARY_PATH=$CONDA_PREFIX/lib`，否则 Pillow 会报 GLIBCXX 错误。项目里的 `.venv` 缺 `timm`，跑不了 Path MeanFlow。

说明页：[path_meanflow.md](path_meanflow.md)

---

## 2026-08-20 搜索基线改成 T-Hybrid A*

已选定 Liu et al., IROS 2023 的 T-Hybrid A*：官方搜索代码 + 我们的 20 m / κ=2.1 地图车辆数字。占据来自 mask，俯仰/滚转/粗糙度来自高程和法向，评分仍用余量 > 0。

复现必须走官方 ROS 节点，不能剥 ROS。剥掉以后 smoother、tf、官方 `planner.cpp` 都不在回路里，和 Uneven 也无法在同一 ROS 环境下比。

评价入口仍是 `--method t_hybrid`，但改为 `roslaunch Thybrid_astar path_plan_eval.launch`，发 `/map`、`/initialpose`、`/move_base_simple/goal`，收 `/sPath`。地形文件用 ROS 参数 `terrain_data_path`，不再写死 `/home/nubot/...`。

无 ROS 的 CLI 只当诊断，主表不用。catkin 包已编在 `baselines/t_hybrid_ws`。ROS smoke（val/env000000 path_0，部分观测）：找到路，约 14 s；规划区域和倾覆通过；曲率和朝向不过。官方 smoother 让最大曲率从剥 ROS 时的几百降到约 5.5，路径已经不是同一条。

说明页：[hybrid_astar.md](hybrid_astar.md)。

后续和 Uneven 的公平对比也放在 ROS（或 ROS 仿真）里做，而不是各写一个脱 ROS 的命令行。

## 2026-08-21：Original Uneven 原版 ROS smoke

在 `baselines/third_party/uneven_planner_original` 内复制了上游的
`UnevenMap + KinoAstar + ALMTrajOpt` 源码，并增加一次性 ROS 入口；外部
`/home/sdu/reference_project/...` 保持只读。高程网格先转为 PCD，空 map 文件
让原版按自己的构图代码重新计算地形表示。

`val/env000000/path_0` 的 smoke：前端 200 点，ALM 外层 8 次收敛，规划时间
约 2.96 s；找到路径，规划区域、曲率、终点 yaw 通过，统一稳定性检查未通过。
这只是单任务诊断，不能代表全量 val 结果。

---

## 现在还卡住的

T-Hybrid 已在 ROS 下 smoke 一条，全量 val 还没跑。  
Original Uneven 已完成一条原版 ROS smoke。Neural A* 官方源码已固定到
`baselines/third_party/neural-astar`，MPT 栅格转换层已完成，但
MPT-compatible checkpoint 尚未取得；在训练预算获批准前不启动新训练。Kicki 官方
`bspline` 分支也已固定到 `baselines/third_party/kicki-neural-path-planning`
（`fc61a01`），其预训练模型和 256x128 车辆局部规划输入仍未适配 MPT。MPD 官方
`mpd-splines-public` 也已固定到 `baselines/third_party/mpd-splines-public`
（`3676cbf`），但它的原生推理依赖 IsaacGym、机器人环境和外部模型目录，尚未
改写成 MPT 适配器。
