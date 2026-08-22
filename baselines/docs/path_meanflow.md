# Path MeanFlow（自己的方法，接到同一张表）

权重：

- Stage 1 / ERPL：`data/path_meanflow/stage1_best.pth`
- Stage 2：`data/path_meanflow/stage2_best.pth`

两个 checkpoint 的轨迹表示版本与当前代码一致（`gauge_fixed_first_order_projected_bridge_44d_v1`）。

评价时也走 `baselines.evaluate`，每次只采一个 source，对应 Feasible@1。  
Stage 1 训练时 `p_mask=0.5`；Stage 2 为 `p_mask=1.0`。和别人比的时候，mask 协议必须写在表注里。

这一步的目的不是再训练，只是让自己的数字和 Hybrid A*、Uneven 来自同一套硬检查。

## 怎么跑

需要 conda 环境 `vim`，以及：

```bash
export LD_LIBRARY_PATH=/home/sdu/miniconda3/envs/vim/lib:$LD_LIBRARY_PATH
cd /home/sdu/MPT
python -m baselines.evaluate --method path_meanflow_stage1 \
  --p_mask 0.5 --mask-seed 2026 --environments 1 --paths-per-env 1
python -m baselines.evaluate --method path_meanflow_stage2 \
  --p_mask 1.0 --mask-seed 2026 --environments 1 --paths-per-env 1
```

## smoke（val/env000000 path_0，一条任务）

两条都能出路径，禁入和朝向通过，倾覆和曲率未通过。  
一条样本不能用来判断第二阶段有没有用。
