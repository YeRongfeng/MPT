# Direct-safe Trajectory MeanFlow 实验入口

本目录只把完整 A/B/C/D 同 seed 结果视为主实验。旧 checkpoint、smoke 和
旧 Stage‑2 held-out 均是诊断，不是 strict OOD。

## 已完成

- `design.md`：实现审计、canonical 插入点和固定画布约束；
- `split_manifest.json`：70/15/15 map-level split、每图 SHA-256 和 manifest
  hash；
- `canonicalization_tests.md`：连续几何、raster、B 样条、source 和 disabled
  行为测试；
- `bridge_prior_report.md`：100,000 样本协方差验证；
- `equivariance_current_global/`：旧全局 Stage‑1 的旋转诊断；
- `current_stage1_candidate_diagnostic/`：旧 Stage‑1 的 K=1/4/8/16/32
  诊断（旧模型见过所有地图，不是 OOD）；
- `experiment_matrix.json`：三 seed 受控矩阵；
- `comparison_report.md`：只在发现完整矩阵时才给研究结论。

## 执行顺序

先训练仅使用 manifest train maps 的 legacy proposal Stage‑1：

```bash
python train_direct_safe_distribution.py \
  --variant A \
  --legacy-global-input \
  --manifest diagnostics/direct_safe_distribution/split_manifest.json \
  --model-template data/sim/model_params.json \
  --epochs 250 --batch-size 16 \
  --output-root diagnostics/direct_safe_distribution/proposal_stage1 \
  --device cuda:0
```

用其 checkpoint 生成 train-only safety teacher：

```bash
python generate_safe_multimodal_dataset.py \
  --model-params diagnostics/direct_safe_distribution/proposal_stage1/experiment_A/seed_20260728/model_params.json \
  --checkpoint diagnostics/direct_safe_distribution/proposal_stage1/experiment_A/seed_20260728/final_model.pth \
  --k-teacher 8 --conditions-per-map 16 \
  --output-dir diagnostics/direct_safe_distribution/safety_teacher_dataset \
  --device cuda:0
```

然后对每个 seed 运行 A/B/C/D。C/D 默认只读
`training_eligible_strict`，不会把 unsafe/improved 冒充安全目标：

```bash
python train_direct_safe_distribution.py \
  --variant D \
  --manifest diagnostics/direct_safe_distribution/split_manifest.json \
  --model-template data/sim/model_params.json \
  --teacher-dataset diagnostics/direct_safe_distribution/safety_teacher_dataset/candidates.npz \
  --safe-target-policy strict \
  --epochs 250 --batch-size 16 --seed 20260728 \
  --device cuda:0
```

validation 可重复运行；test 必须显式确认：

```bash
python evaluate_direct_safe_distribution.py \
  --checkpoint diagnostics/direct_safe_distribution/experiment_D/seed_20260728/final_model.pth \
  --split validation --device cuda:0

python evaluate_direct_safe_distribution.py \
  --checkpoint diagnostics/direct_safe_distribution/experiment_D/seed_20260728/final_model.pth \
  --split test --allow-test --device cuda:0
```

等变性和最终汇总：

```bash
python evaluate_se2_equivariance.py \
  --checkpoint diagnostics/direct_safe_distribution/experiment_D/seed_20260728/final_model.pth \
  --device cuda:0

python compare_direct_safe_2x2.py --split test
```

不要在主矩阵中加入 OT、FIM、Refiner、scorer、CFG、局部高分辨率分支或新的
source prior。若 Safe@32 仍低，停止 scorer 路线；若 Safe@K 高而 Valid
Rate 低，才把排序作为后续独立问题。

共同扩展画布只服务于 task frame 表示。A/B/C/D 推理都会把输出 residual
换算回原全局 normalized residual，并再次调用现有、未修改的 radial projector，
所以最终 26 控制点仍硬约束在原 `[-10,10]²` 地图边界内。
