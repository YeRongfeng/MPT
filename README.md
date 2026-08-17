# Boundary-Constrained Path MeanFlow

本仓库当前已经成立的主方法为：

**Boundary-Constrained Path MeanFlow**

中文名称：**基于一阶边界约束路径表示的条件 Path MeanFlow**。

当前主方法由以下模块组成：

1. Canonical geometric path parameterization；
2. First-order boundary-constrained B-spline representation；
3. First-order boundary-projected Gaussian path prior；
4. Conditional Path MeanFlow pretraining（Stage 1）。

当前 Stage 1 部署形式为**单次路径生成**。Stage 2 的当前训练候选为
`Direct Privileged Cost`：在部署点直接优化特权 task cost，同时保持部署输入不变。
完整地形只用于 Stage 2 训练损失和离线 validation，不能进入部署模型输入。
Stage 2 仍处于验证阶段，final test 未打开，不能把训练结果写成最终方法结论。

模型输出的是供局部规划器跟随的二维全局几何参考路径。曲线参数 \(t\)
不表示真实时间，模型不生成速度、时长或控制输入。

术语规范详见
[docs/research/METHOD_NOMENCLATURE.md](docs/research/METHOD_NOMENCLATURE.md)。

## 训练

先进入环境：

```bash
cd /home/yrf/MPT
source /home/yrf/miniconda3/etc/profile.d/conda.sh
conda activate vim
```

Stage 1 训练命令：

```bash
python3 train_flow.py \
  --workflow stage1 \
  --fileDir data/path_meanflow \
  --stage1_epochs 20
```

继续当前已保存的 Stage 1（旧目录名只作为资产路径保留）：

```bash
python3 train_flow.py \
  --workflow stage1 \
  --fileDir data/path_meanflow \
  --resume data/path_meanflow/stage1_last.pth \
  --stage1_epochs 20
```

### Stage 2：Direct Privileged Cost

当前 Stage 2 从 Stage 1 checkpoint 初始化，在部署点 `t=1,r=0` 的生成路径上直接
反向传播 privileged task cost。完整 cost map 和独立 mask 只用于训练损失与离线
validation；推理输入仍然是部分观测地图、mask、起终点和 source noise。该方法不使用
Stage 1 anchor、replay、expert、Hungarian matching 或 target archive。

Stage 2 默认启用三目标 MGDA（`--stage2_use_mgda`，默认值为 `True`）。每次更新先
对未加权的 `forbidden_region`、`stability`、`analytic_curvature` 分别求共享生成器
全参数梯度，再用三目标 exact active-set solver 求 simplex 系数 `alpha`，最后把组合
梯度交给 plain SGD 更新（无 momentum、无 weight decay）。这是为了保持实际参数更新
方向与 `-g_MGDA` 共线；global gradient clipping 仍保留，因为它只做统一标量缩放。
MGDA 不改变 forward cost、数据协议或 checkpoint 选择规则。它是当前暂时采用的实验
方案，不代表已经通过最终验收。

原来的 fixed direct-cost scalarization 保留为对照，可使用
`--no-stage2_use_mgda`（等价于对 `5F+1S+0.5K` 反传），该对照仍使用 Adam。MGDA 和
fixed 是两个不同的训练协议；不要在两种模式之间续训，分别从同一个 Stage 1 checkpoint 启动，并使用
不同的 `--fileDir`。checkpoint 会记录 `stage2_use_mgda` 和实际 optimizer，续训时会
拒绝模式或 optimizer 不一致的 checkpoint。旧 MGDA+Adam checkpoint 不能续训到当前
MGDA+SGD 协议，必须重新从 Stage 1 启动。

正式长程配置使用 `data/dataset1/train` 的全部 100 个训练环境，并从独立的
`data/dataset1_val/train` 使用 50 个 validation 环境和 50 个固定 validation contexts；
每次更新使用 4 个 source/context；MGDA arm 使用 plain SGD，fixed arm 使用 Adam，
二者学习率均为 `1e-5`，梯度裁剪均为 `1.0`。训练长度由 `stage2_epochs` 控制；可选的
`stage2_max_updates` 只用于固定 update 预算的审计。固定 validation 默认每 100 updates
执行一次，终端摘要默认每 100 updates 打印一次，TensorBoard 仍记录每个 update。
每个 epoch 还会记录 source、condition 和 environment 三层 task-cost 分布，环境明细位于
`stage2_direct_cost/train_epoch/environment_detail/<env>/task_cost/`。
`stage2_best.pth` 按 validation `task_cost` 最低选择，`Safe@1` 和 `strict-valid`
仅作为并列时的次级排序；strict/forbidden/stability/curvature regression 保留为
诊断指标，不阻止 best checkpoint 保存。`stage2_last.pth` 始终保存最近一次训练状态。

长程训练 Stage 2 MGDA arm（默认模式）：

```bash
python3 train_flow.py \
  --workflow stage2 \
  --dataFolder data/dataset1 \
  --fileDir data/path_meanflow_stage2_mgda_long \
  --prior_checkpoint data/path_meanflow/stage1_best.pth \
  --stage2_validation_data data/dataset1_val \
  --stage2_validation_split train \
  --stage2_split_seed 20260802 \
  --stage2_train_environments 100 \
  --stage2_validation_environments 50 \
  --stage2_validation_contexts 50 \
  --stage2_validation_sources 16 \
  --stage2_epochs 30 \
  --stage2_use_mgda
```

长程训练 Stage 2 fixed 对照 arm：

```bash
python3 train_flow.py \
  --workflow stage2 \
  --dataFolder data/dataset1 \
  --fileDir data/path_meanflow_stage2_fixed_long \
  --prior_checkpoint data/path_meanflow/stage1_best.pth \
  --stage2_validation_data data/dataset1_val \
  --stage2_validation_split train \
  --stage2_split_seed 20260802 \
  --stage2_train_environments 100 \
  --stage2_validation_environments 50 \
  --stage2_validation_contexts 50 \
  --stage2_validation_sources 16 \
  --stage2_epochs 30 \
  --no-stage2_use_mgda
```

训练输出包含 `stage2_best.pth`、带 optimizer 状态的 `stage2_last.pth`、
`stage2_method.json` 和 `tensorboard/stage2_direct_cost/`。只有从同一 direct-cost
协议生成的 `stage2_last.pth` 才能续训。如果需要固定为 1200 updates、并保持短程
审计的每 5 updates validation 频率，可以显式加上；续训时必须保持原 arm 的
`--fileDir` 和 MGDA/fixed 参数：

```bash
python3 train_flow.py \
  --workflow stage2 \
  --dataFolder data/dataset1 \
  --fileDir data/path_meanflow_stage2_mgda_long \
  --stage2_resume data/path_meanflow_stage2_mgda_long/stage2_last.pth \
  --stage2_validation_data data/dataset1_val \
  --stage2_validation_split train \
  --stage2_train_environments 100 \
  --stage2_validation_environments 50 \
  --stage2_validation_contexts 50 \
  --stage2_epochs 30 \
  --stage2_max_updates 1200 \
  --stage2_eval_every_updates 5 \
  --stage2_use_mgda
```

`train_flow.py` 是当前 Stage 2 生成器训练的正式入口；`train_stage2_critic.py` 和
source-transport 脚本不属于 direct-cost 训练流程。

### 当前 Stage 2 状态

当前 direct-cost 入口已经接入固定的独立 train-root/validation-root 协议，
并保留 Stage 1 paired baseline、逐约束 regression、Safe@1、strict-valid 和
source diversity 诊断。旧 source-transport/critic 验收报告不作为 direct-cost 的
证据；direct-cost 的新训练结果仍需按该协议独立记录，final test 继续关闭。

## 可视化

### Stage 1

```bash
python3 vis_dit.py \
  --workflow stage1 \
  --environment env000060 \
  --paths 0 3 6 9 12 15
```

Stage 1 默认读取 `data/dataset1/val`，使用当前 Stage 1 checkpoint 并生成 32 条
候选。可用 `--mask_mode full` 检查完整地图输入。

### Stage 2

`vis_dit.py` 可以加载 direct-cost checkpoint，从 `data/dataset1/val` 读取数据，
使用 Stage 2 独立 mask（`p_mask=1.0`）生成候选。direct-cost 训练不包含 critic
selector，使用 `--without-selector` 只检查生成器输出和离线几何指标：

```bash
MPLCONFIGDIR=/tmp/matplotlib-vis-stage2 python3 vis_dit.py \
  --workflow stage2 \
  --checkpoint data/pmf_s2_stability_only/stage2_best.pth \
  --dataset-root data/dataset1 \
  --without-selector \
  --environment env000040 \
  --paths 0 1 2 3 4 5
```

图中：

- 青色：通过 forbidden 硬过滤、仍可参与选择的候选；
- 灰色：被 forbidden 硬过滤排除的候选；
- `Privileged truth`：只用于离线事后诊断，不参与生成器输入。

如果全部候选都未通过 forbidden，脚本会明确输出 `REJECT`，不会回退选择非法
轨迹。PNG 和逐候选 JSON 默认保存到：

```text
predictions/path_meanflow_stage2/<environment>/
```

## 地图处理
先离线生成所有 train/val 的 stability map：

```bash
cd /home/yrf/MPT
source /home/yrf/miniconda3/etc/profile.d/conda.sh
conda activate vim

python3 -m tools.data.generate_stability_maps \
  --dataset-root data/dataset1 \
  --splits train val
```

## Repository layout

- 根目录只保留当前训练、评估、模型和几何主入口。
- `tools/` 只保存长期维护的命令行工具。
- `visualization/` 保存画图和结果查看脚本。
- `predictions/` 保存常规推理与可视化产物，不属于诊断实验。
- `tests/` 统一保存测试相关内容：根目录是正式回归测试，`tests/legacy/`
  是旧手工检查，`tests/audits/` 是一次性研究试验区。
- `docs/` 保存当前正式协议和旧方法文档。
- `legacy/` 保存已废弃的 OMPL、RRT*、Dubins、MPNet 等实现。
- `tests/audits/` 只保存一次性研究审计、配套测试、协议和报告，该子目录不进入
  Git。

常规预测默认写入 `predictions/`，可通过 `MPT_PREDICTIONS_DIR` 重定向；
一次性研究审计默认写入 `tests/audits/`，可通过 `MPT_TEST_DIR` 重定向。


## Legacy methods

The original OMPL, RRT*, Dubins-car, and baseline MPT instructions are archived in [docs/legacy/ORIGINAL_MPT_README.md](docs/legacy/ORIGINAL_MPT_README.md). They are not part of the current terrain-aware Stage 1 workflow.
