# Method nomenclature

## Current method status

当前已经成立并可用于部署评价的方法是：

**Boundary-Constrained Path MeanFlow**

中文：**基于一阶边界约束路径表示的条件 Path MeanFlow**。

当前部署形式是单次路径生成，主指标为 Safe@1。Oracle Safe@K 仅为覆盖
诊断；候选选择器不属于当前 Method。

以下完整研究计划名称仅在 Stage 2 通过正式准入后使用，当前不得写成已经
成立的方法：

**Boundary-Constrained Path MeanFlow with Privileged Constraint
Distillation**

中文：**基于一阶边界约束路径表示与特权约束蒸馏的条件 Path
MeanFlow**。

## Required terminology

| Concept | Canonical term | 中文 |
|---|---|---|
| Geometric preprocessing | Canonical geometric path parameterization | 几何路径规范参数化 |
| Task frame | Task-canonical coordinate frame | 任务规范坐标系 |
| Path representation | First-order boundary-constrained B-spline representation | 一阶边界约束 B 样条路径表示 |
| Source path distribution | First-order boundary-projected Gaussian path prior | 一阶边界投影高斯路径先验 |
| Generated variable \(y\) | Free path coordinates | 自由路径坐标 |
| Stage 1 | Conditional Path MeanFlow pretraining | 条件 Path MeanFlow 预训练 |
| Stage 2 research hypothesis | Relative-entropy-constrained feasibility reweighting and distribution distillation | 相对熵约束的可行性重加权与分布蒸馏 |
| Training-only optimizer | Privileged path corrector | 特权路径修正器 |
| Local Stage 2 target | Same-source bounded local improvement target | 同 source 有界局部改进目标 |
| Reweighting baseline | Feasibility-reweighted Path MeanFlow (D) | 可行性重加权 Path MeanFlow（D） |
| Primary Stage 2 candidate | Local support expansion plus feasibility reweighting (B+D) | 局部支持扩充与可行性重加权（B+D） |
| Optional warm start | Feasibility-oriented distribution warm start | 面向可行域的分布预热 |
| Model output | Global geometric reference path | 全局几何参考路径 |

## Terms not used for the current method

- Do not call the source prior a Brownian bridge, conditional Gaussian bridge,
  or standard Gaussian bridge.
- Do not call the 44-D variables residual edges. They are free path
  coordinates, \(y\in\mathbb R^r\).
- Do not describe Stage 2 as teacher–student learning or expert imitation.
  The formal term is privileged constraint distillation; the training-only
  module is a privileged path corrector.
- Do not describe relative-entropy-constrained distribution distillation as
  an established method property before it passes the admission criteria in
  `STAGE2_RESEARCH_PROTOCOL.md`.
- Do not describe full-correction target imitation or iterative on-policy data
  aggregation as the current default Stage 2. They remain experimental
  controls.
- Do not use DAgger as the method name. On-policy data aggregation is only a
  possible later experimental mechanism after the fixed D/B+D admissions.
- Do not report Oracle Safe@K as deployed system performance. The deployed
  metric is Safe@1.
- Do not call D a selector. It changes the training target distribution;
  deployment still draws one path from one Path MeanFlow forward pass.
- Do not claim that empirical candidate-pool KL equals the true KL between
  successive Path MeanFlow distributions.
- Do not call the output a timed trajectory. It is a geometric path:
  \(p:[0,1]\to\mathbb R^2\), and \(t\) is not physical time.
- Do not claim general Wasserstein contraction or universally faster
  training. The proven statement is the per-sample \(M\)-metric projection
  inequality for boundary-feasible targets.

## Symbol convention

- \(c_{\mathrm{obs}}=(M_{\mathrm{obs}},m,S,G)\): local observation condition.
- \(m=1\): observed and traversable; \(m=0\): unknown or forbidden.
- \(y_0\sim\mathcal N(0,I_r)\): source free path coordinates.
- \(y_1\): demonstration path coordinates.
- \(y^{\mathrm{cand}}\): student candidate path coordinates.
- \(y^{\mathrm{ref}}\): privileged-corrected path coordinates.
- \(y^+\): stopped-gradient, same-source bounded local improvement target.
- \(\bar w_i\): per-condition normalized feasibility weight.
- \(z_i'\): newly sampled source, independent of the source that generated
  candidate \(y_i\).
- \(D_c(y)\): condition-dependent affine path decoder.
- \(p,p',p''\): analytic B-spline geometry.

## Authoritative Stage 2 protocol

The current research status, A/B/C/D/B+D controls, dual-budget comparison,
metric priority, and admission criteria are defined in
[`STAGE2_RESEARCH_PROTOCOL.md`](STAGE2_RESEARCH_PROTOCOL.md).

## Compatibility note

Historical filenames, directories, checkpoint semantic tokens, and import
aliases containing `gauge`, `bridge`, `residual`, or `dagger` remain readable
only to preserve existing assets. They are compatibility identifiers, not
publication-facing terminology. New production code should import
`boundary_constrained_path` and use the canonical names above.
