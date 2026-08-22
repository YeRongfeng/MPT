# 搜索基线：T-Hybrid A*

## 选定

论文主表的搜索行写 **T-Hybrid A\***（Liu et al., IROS 2023），使用其官方 ROS 节点。

原因：

- 崎岖地形上的 Hybrid A* 已经有论文和代码，不能自己再实现一版格子搜索然后叫 Hybrid A*。
- Uneven 的 kino_astar 虽然也是 Hybrid A* 前端，但和 Uneven Planner 是同一套仓库、同一套 terrain pose mapping。主表里再单列一行，读者很难看出搜索和优化差在哪里。
- T-Hybrid 是独立工作：Kurzer Hybrid A* + 混合 2D/2.5D 地图，用粗糙度和投影后的俯仰/滚转做可通行性。正好回答“标准车式搜索加上地形可通行性，可行性如何”。

官方仓库：`https://github.com/nubot-nudt/T-Hybrid-planner`  
本地：`baselines/third_party/T-Hybrid-planner`  
接到评价器的入口：`--method t_hybrid`

## 官方代码实际在做什么

搜索还是 Kurzer Hybrid A*：Ackermann 运动基元、倒车、2D 有障启发、Reeds-Shepp 启发、72 个航向。可通行性是他们加的：

- 每个体素：`static_traver`、粗糙度、法向 `(nx,ny,nz)`
- 扩展节点时，用法向和当前航向算车体俯仰/滚转
- 超过阈值（俯仰约 -0.3/0.25 rad，滚转 0.18 rad，粗糙度再缩放）直接丢掉
- 没超的话，把 `1 - (0.4 pitch + 0.4 roll + 0.3 roughness)` 写进 g 代价

2D 占据层和 2.5D 体素是分开的。我们这边：mask 看不见或不能走的格子当占据；看得见的自由格用高程和法向写成他们的 `terrainData.txt`。

官方仓库绑在 ROS 上：`/map`、`/initialpose`、`/move_base_simple/goal` 进，`/path` 和 `/sPath` 出，中间有 Voronoi 平滑。**主表复现走这条 ROS 节点，不剥 ROS。** 剥掉 ROS 会跳过 smoother、tf、官方 `planner.cpp` 的起终点检查，和 Uneven 等 ROS 方法也不在同一运行环境里，后面没法公平对比。

地形文件原仓库写死成 `/home/nubot/...`。接到评价器时改成 ROS 参数 `terrain_data_path`（launch 传入），仍由官方节点自己读。

官方 `hybridAStar` 里 Dubins shot 函数在，主循环没调用。终点判定是格子距离 < 1（在 0.2 m 格子上约 0.2 m），不查航向。这些保持原样。迭代次数到了却还没靠近终点时，命令行记为失败，不当成找到路。

## 相对原论文改了什么（T-Hybrid-adapted）

| 量 | 原仓库 | 接到 dataset1 |
|---|---|---|
| 地图 | 27 m，0.3 m 格 | 20 m，0.2 m 格 |
| 最小转弯半径 | 注释写 6 m，运动基元按 6 格（约 1.8 m） | 按我们的 κ=2.1，世界半径 0.476 m，仍用他们的“r 是格子单位”约定 |
| 车体 | 0.5 × 0.5 m | 0.4 × 0.4 m（对应 0.2 m 半径） |
| 俯仰/滚转/粗糙度阈值 | 原值 | 原值，不用我们的 `cost_map` 当搜索约束 |
| 输入 | ROS OccupancyGrid + 写死的 terrain 文件 | mask 占据 + 从高程/法向写出的 terrain 文件 |

打分仍是统一硬指标：规划区域、余量 > 0、曲率、端点朝向。T-Hybrid 自己的滚转/俯仰阈值只用于搜索，不代替评分。

还有两处和原仓库不完全一样，写进表里要叫 **T-Hybrid-adapted**：

- 原代码把路程代价乘上 `(1-可通行性)`。完全可走的格子 g=0，A* 会在平地上空转。我们改成「路程 + 他们的可通行性惩罚」，否则平坦区域搜不到。
- 原仓库有 Dubins shot 函数和 `dubinsShot=true`，主循环没调用，终点只看格子距离。我们没擅自接上。所以端点朝向、折线曲率仍可能不过。

## 怎么跑（ROS）

需要本机 Noetic。第一次先编官方包：

```bash
source /opt/ros/noetic/setup.bash
mkdir -p baselines/t_hybrid_ws/src
ln -sfn "$(pwd)/baselines/third_party/T-Hybrid-planner/src/path_planner" \
  baselines/t_hybrid_ws/src/Thybrid_astar
cd baselines/t_hybrid_ws && catkin_make
```

评价器会自己 `roslaunch Thybrid_astar path_plan_eval.launch`（不启 RViz），往 `/map` 发占据，往起终点话题发姿态，收 `/sPath`。也可用官方 launch 加 RViz 做可视化，但批处理不要开 RViz。

```bash
.venv/bin/python -m unittest tests.test_t_hybrid
.venv/bin/python -m baselines.evaluate \
  --method t_hybrid \
  --dataFolder data/dataset1 \
  --split val \
  --environments 1 \
  --paths-per-env 1 \
  --mask-mode partial
```

`baselines/t_hybrid_cli` 是之前剥 ROS 的诊断二进制，**不能当论文复现，也不参与和 Uneven 的公平对比。**

## ROS smoke（val/env000000 path_0，部分观测）

官方节点 `Thybrid_astar` + `/sPath`，约 14 s：

- 找到路
- 规划区域通过
- 倾覆通过
- 曲率不过
- 终点朝向不过

联合可行仍是 0。官方 smoother 把最大曲率从剥 ROS 时的几百降到约 5.5，仍然超过 κ=2.1，但已经说明：**不走 ROS 节点会把路径形态改掉，不能当复现。**

主表只使用上方的官方 T-Hybrid ROS 节点；不再保留仓库内的手写 Hybrid A*
探针或其独立评估入口。
