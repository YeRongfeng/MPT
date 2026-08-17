# Stage 2 试错复盘：哪些路走过，为什么停下

> 截止日期：2026-08-05。本文是实验史和自查材料，不是新的 Method 结论。
>
> 最重要的一句话：**目前没有一条路线通过完整的 Stage 2 端到端准入；最接近通过的是 path-aligned privileged reweighting，但 sealed-system acceptance 仍未确认。**

## 怎么读这份记录

先看第 1 节的结论地图，再按自己关心的路线看第 2--7 节。每条路线都按同一个顺序写：

1. 想解决什么问题；
2. 做出了什么结果；
3. 为什么没有继续；
4. 哪些正面结果可以保留，哪些不能外推。

正文只保留能改变判定的数字。更细的变体数字放在“补充证据”中，原始审计报告集中列在最后。

## 1. 结论地图

| 路线 | 原本想解决的问题 | 当前判定 | 一句话原因 |
| --- | --- | --- | --- |
| 直接 privileged-cost 反传 | 用训练期可见的 privileged cost 直接改进 Stage 1 proposal | `FAIL` | 短程局部有效，但长程 Safe@1 不稳；跨 condition 会失效；保护项要么仍有回归，要么几乎不更新。 |
| 候选 selector | 从候选池中自动选出安全 candidate | `FAIL` | Oracle 证明池里有好候选，但可部署 scorer 选不出来，Safe@1 增益 CI 也不可靠。 |
| D0/D1 重加权 | 先在有限候选池中重排，再让网络学会重排后的分布 | D0 `PASS-EXPLORATION`；D1 `FAIL` | 池内重排确实有效，但网络只恢复了很小一部分收益，绝对 Safe@1 没有确认提升。 |
| teacher / operator / residual student | 先用 privileged 信息得到安全修正，再摊销到 student | teacher `PASS-LOCAL`；student `FAIL` | teacher 能修一部分样本，但修正不稳定，student 不能稳定复制 teacher 的安全收益。 |
| 分布蒸馏、XMeanFlow、NFT、transport | 通过更丰富的目标或参考轨迹扩展安全 support | `DIAG` / `FAIL` | 目标池或上限有时很好，但 student、长期 cost、mode retention 或正式 CI 没过。 |
| mask-conditioned M0/M1 | 让网络根据 mask 学会不同约束组合 | M0 `FAIL`；M1 `BLOCKED` | M0 的 B+C train/validation gap 为 33.4 pp，超过 20 pp 门槛；按协议没有训练 M1。 |
| path-aligned privileged reweighting | 用 deployment-visible 的 path/map 几何做 critic，privileged terrain 只提供训练标签 | 探索与独立确认 `PASS`；sealed system `NOT CONFIRMED` | 这是目前最强候选，但更高层端到端 Safe@1/16 的 CI 下界仍为 -0.21 pp。 |

### 1.1 先记住六个结论

1. **Stage 2 不是“完全没有信号”。** direct cost、teacher 和 D0 都出现过明确的局部正向结果。
2. **问题在于正向信号不能稳定穿过下一层。** 典型链条是：候选池有好目标 -> teacher 能找到目标 -> student 学不会 -> 完整系统没有提升。
3. **D0 通过不等于 D1 通过。** D0 只回答“有限候选池重排有没有用”；D1 才回答“网络能不能摊销这个重排”。
4. **teacher/Oracle/Best-of-K 不是部署结果。** 它们可以作为上限或机制证据，不能替代部署时的 Safe@1。
5. **M0 失败后没有运行 M1 是协议要求，不是遗漏。** 不能把“没有 M1 结果”写成 M1 失败，也不能写成 M1 成功。
6. **所有 final-test 结论都还关闭。** 目前最多能说探索 admission 或独立 confirmation 通过，不能说正式 Stage 2 已确认。

### 1.2 术语先说明

- **Safe@1**：部署时只选第一个 candidate 后的安全率，是最接近真实使用的指标。
- **strict-valid**：所有冻结约束同时满足的比例，比只看单个安全指标更严格。
- **回归**：原本安全的 proposal 经过 Stage 2 后变得不安全，例如 `strict-to-invalid`、stability regression、curvature regression。
- **teacher / student**：teacher 可以使用 privileged 信息生成或筛选目标；student 只能使用部署时可见输入，负责把 teacher 的效果学出来。
- **D0 / D1**：D0 是不训练网络的有限候选池重加权审计；D1 是训练网络摊销 D0 目标。
- **environment-bootstrap CI**：按 environment 重采样得到的置信区间，用来避免 pooled point estimate 掩盖跨环境不稳定。
- **observed-zero-safe**：当前有限候选池中没有观察到安全候选；这不等于证明真实 support 不存在。

## 2. 路线一：直接 privileged-cost 反传

### 想解决什么

最直接的想法是：训练时用完整地图计算 privileged cost，直接把 cost 梯度反传到 proposal network；推理时仍只用原有 deployment-visible 输入。这个方案没有 anchor、replay 或额外 expert，目的是先做最小可归因实验。

### 做出了什么结果

- **10-step smoke** 看起来有改善：held-out Safe@1 `9.4% -> 15.6%`，strict-valid `7.0% -> 10.9%`。
- **100-step long run** 没有保住改善：Safe@1 `9.4% -> 9.4%`，task cost `2.4579 -> 2.5297`，path drift `0 -> 0.167 m`。
- 修正曲率软梯度和低速 yaw 反向分母后，数值链路可以通过等价性、hard-field、strict 和有限差分审计；但 corrected long run 的 Safe@1 `25.00% -> 18.75%`，`strict-to-invalid=45.28%`，stability regression `34.44%`，curvature regression `31.58%`。
- 多 pair 结果显示局部泛化很有限：trained pair `0 -> 12.5%`，same-condition new source `12.5 -> 12.5%`，new condition `12.5 -> 0%`，同时 cost 增加 `+0.7332`。

### 为什么停下

1. **短程正向不代表长程有效。** 继续更新后，主指标回到原点甚至恶化。
2. **跨 condition 的梯度方向不一致。** 同 condition 还有小幅 cost 改善，new condition 则出现 Safe@1 归零和 cost 上升。
3. **加入保护项仍未形成可接受解。** safe replay、A-GEM、backtracking 等版本有些提高 Safe@16，但同时出现 curvature/stability regression；safe-core 版本虽能把回归压到 0，accepted step fraction 只有约 `0.0005`，基本等于冻结。
4. **不能事后挑最好 update。** update 1030 曾有 Safe@1 `43.75%`，但 strict-to-invalid `47.17%`，不满足冻结回归上限；因此正式 checkpoint 只能保留 update 0。

### 能保留什么

可以保留：direct cost 的入口、梯度和数值审计基本成立，Stage 1 source 也没有完全失效。不能保留：把短程 Safe@1 增长或某个漂亮 update 当成完整方法通过。

<details>
<summary>直接 cost 的补充变体数字</summary>

| 变体 | 正向现象 | 仍未通过的原因 |
| --- | --- | --- |
| FULL ALL-DIRECT 4x8 | Safe@16 `50.00% -> 56.25%` | mean cost `1.1119 -> 1.2257`；safe regression `27.54%`；curvature regression `31.65%`。 |
| 32 conditions x 1 source | 扩大 condition coverage | safe regression `30.43%`；curvature regression `30.38%`。 |
| conditioning / normmean / norm-PCGrad | 部分版本把 curvature regression 降到约 `8.86%` | confirmatory 仍为 `False`，没有同时获得可靠 Safe@1/16 增益。 |
| safe-core / worst-core | F/S/K regression 可到 `0/0/0` | accepted step fraction 约 `0.0005`，没有真正推动主指标。 |
| A-GEM / safe replay | Safe@16 最高 `62.50%` | F/S/K regression 仍约 `1.77/4.35/13.92%`，confirmatory `False`。 |
| curvature margin | proposal safe `23.05%`，Safe@16 `56.25%` | F/S/K regression `1.77/5.43/29.11%`。 |
| unsafe-only / anchor | Safe@16 可到 `62.50%` | environment CI 包含 0，不能证明 anchor 或 unsafe-only 有可归因收益。 |
| external V11 | proposal safe `16.86 -> 17.04%` | mean cost `2.393285 -> 2.394511`，mean-cost gate 失败。 |

</details>

原始记录：[`direct_privileged_cost_no_anchor_20260729`](../../tests/audits/direct_privileged_cost_no_anchor_20260729/REPORT.md)、[`a0_multipair_controllability`](../../tests/audits/a0_multipair_controllability/REPORT.md)、[`direct_cost_condition_margin_v2_20260805`](../../tests/audits/direct_cost_condition_margin_v2_20260805/REPORT.md)、[`all_direct_v11_external_admission_20260805`](../../tests/audits/all_direct_v11_external_admission_20260805/REPORT.md)。

## 3. 路线二：候选 selector 与 D0/D1 重加权

### 想解决什么

direct cost 的问题是直接改网络太激进，于是改成两步：先在有限候选池中找到更安全的 candidate，再让网络学习这个分布。这样可以把“找好目标”和“训练网络”分开审计。

### 3.1 旧 selector：Oracle 有机会，但 scorer 选不出来

旧 selector 的 full regime 结果是：Safe@1 `8.3%`，Oracle Safe@K `26.7%`，selected Safe@K `11.7%`；`R_pick=43.8%`，低于 `70%` 门槛，`R_gain=18.2%`，低于 `50%` 门槛，selected gain 的 CI `[-5.0%, 11.7%]` 也跨过 0。

后续 S0 候选池确实很有潜力：Stage 1 Safe@1 `20.31%`，Oracle Safe@32 `73.44%`，oracle gain CI `[42.19,64.06]`。但两个可部署 scorer 仍不行：

| scorer | selected Safe@1 | 相对 Stage 1 | 主要问题 |
| --- | ---: | ---: | --- |
| global map/path | `15.62%` | `-4.69 pp` | 直接低于 baseline。 |
| path-map aligned | `25.00%` | `+4.69 pp` | gain CI `[-7.81,14.06]`；stability/curvature/strict regression 分别为 `9.38/10.94/9.38%`。 |

**停线理由：**候选池有好东西，但 selector 的排序能力和约束保持都不够；不能拿 Oracle 数字代替部署 Safe@1。

### 3.2 D0：有限池重排通过独立确认

第一轮使用 KL=`0.3` 时，Safe@1 `7.71% -> 14.01%`，CI `[+2.44,+10.76] pp`，但 mode retention `89.9% < 90%`，所以失败。

独立 confirmation 换了 condition、source lineage 和 mask randomness，并将 KL 降到 `0.2`：Safe@1 `15.21% -> 22.87%`，增益 `+7.66 pp`，CI `[+4.37,+11.35] pp`，minimum ESS/M `0.6395`，mode retention `92.0%`。因此 **D0 的非参数有限池重加权通过了自己的探索确认**。

### 3.3 D1：网络没有学会复现 D0

D1-U 是 uniform control，D1-W 是 weighted training：

| 比较 | 结果 | 应如何解读 |
| --- | ---: | --- |
| D1-W - D1-U | `+2.29 pp`，CI `[+0.63,+4.58]` | 加权相对普通自蒸馏确有可归因收益。 |
| D1-W - Stage 1 | `+1.67 pp`，CI `[-1.46,+4.37]` | 相对原始 baseline 的绝对收益没有确认。 |
| D0 目标 -> D1 student | `R_amort=22.1%`，amortization gap `5.89 pp` | student 只恢复了有限一部分目标池收益。 |

**停线理由：**D0 证明的是“候选池重排有效”，D1 没有证明“网络能把这个重排摊销出来”。后续 conservative RWPMF 仍出现 task-cost gate 失败，因此不能把 D0 的正证据升级成 Stage 2 通过。

### 3.4 当前最强候选：path-aligned reweighting

这条路线用 deployment-visible 的 path/map 几何做 critic，privileged terrain 只用于训练期产生标签，并用 KL=`0.2` 限制重加权。

- admission：uniform `18.55%`，weighted `24.20%`，增益 `+5.64 pp`，environment CI `[+2.53,+8.85] pp`，通过；
- independent confirmation：uniform `16.31%`，weighted `22.43%`，增益 `+6.12 pp`，CI `[+2.95,+9.70] pp`，通过；
- sealed-system acceptance：Stage 2 相对 Stage 1 的 Safe@16/selected Safe@1 delta 为 `+1.88%`，CI `[-0.21,+4.38]%`，状态 `STAGE2_SYSTEM_NOT_CONFIRMED`。

**当前结论：**它是唯一同时有探索 admission 和独立 confirmation 正证据的主线，但还不能写成端到端确认，更不能打开 final-test。

### 3.5 为什么 critic 和 curvature filter 也没有救回来

- stability critic V1/V2 在 external admission 的增益只有 `-0.15%` 或 `+0.21%`；V3 new-vs-old 为 `-0.06%`，`new_beats_old` 和 ESS gate 失败。
- pool-KL v1 相对 baseline `-2.51%`；v3 在已打开 admission 上虽有 `+1.80%`，但 calibration 失败且增益集中在单一 environment。
- curvature soft filter 能提高 finite-pool strict-safe mass，但 hard filtering 会降低 coverage 和 mode retention；它是 support 诊断，不是部署方案。

原始记录：[`candidate_selector_admission_20260730`](../../tests/audits/candidate_selector_admission_20260730/REPORT.md)、[`d0_confirm_kl02_20260730`](../../tests/audits/d0_confirm_kl02_20260730/REPORT.md)、[`d1_uniform_weighted_20260730`](../../tests/audits/d1_uniform_weighted_20260730/REPORT.md)、[`stage2_path_aligned_reweighting_v1_20260803`](../../tests/audits/stage2_path_aligned_reweighting_v1_20260803/REPORT.md)、[`stage2_path_aligned_reweighting_confirm_20260803`](../../tests/audits/stage2_path_aligned_reweighting_confirm_20260803/REPORT.md)、[`stage2_sealed_system_acceptance_20260804`](../../tests/audits/stage2_sealed_system_acceptance_20260804/REPORT.md)。

## 4. 路线三：teacher、operator 和 residual student

### 想解决什么

如果直接训练 proposal 太难，就先让 privileged teacher 做安全修正，再训练 student 模仿这个修正。这里的关键不是 teacher 能不能找到一个好结果，而是修正是否稳定、是否能被 deployment-visible student 学会。

### 4.1 operator：能修，不代表修得稳定

不同 proximity weight（`0.002/0.2/2/20/100`）的审计反复发现：coordinate 版本 corrected strict 可到 `35.9%`，但 idempotence path RMS q95 `0.770 m`、local amplification q95 `382`；physical weight `0.2` 时分别为 `0.413 m` 和 `277`；weight `20` 仍有 amplification q95 `74`。所有版本都没有满足 robust identity、q95 位移 `<=0.10 m` 和 amplification `<=20`。

**停线理由：**局部修正存在，但不是稳定的近端 operator，重复应用或换 condition 后会放大误差。

### 4.2 structured/joint correction：专项 teacher 和 joint teacher 都没有形成可摊销安全裕量

- structured correction 的 cross-constraint forbidden regression `11.1%`，stability regression `33.1%`，超过 5% 门槛；joint teacher 的 V-gain retention 只有 `14.0%`，低于 80% 门槛。
- joint operator 的 validation safe radius 中位数只有 `0.0015 m`，student error/safe-radius ratio 中位数 `619.874`，safe-tube hit rate `0%`。
- 旧 A/B/C/D coupling 中，D held-out Recovery `0.214`，teacher error `1.005`，safe@4 `0%`，mode keep `82.5%`；B 虽在训练集 Recovery `0.949`，held-out abnormal 却达 `35%`，mode keep `72.5%`。

**不能误读：**C+ overfit 在固定 32 条样本上能恢复 teacher 改善的 `84.9%/93.3%`，说明模型有小数据拟合能力；它没有提供跨地图、跨 noise、held-out 泛化证据。

### 4.3 PPCD：corrected teacher 通过，student 失败

早期 PPCD 因错误使用 historical zero-sum projection，在当前 44-D free-coordinate state 上产生约 `1.25 m` reconstruction error；这些早期目录标为 `INVALID`。

修正后结果：

| 版本 | teacher strict-valid | teacher 回归 | student strict-valid |
| --- | ---: | ---: | ---: |
| single-step | `3.12% -> 6.25%` | `0%` | `0.78%` |
| three-step | `3.12% -> 10.94%` | `0%` | `0%` |

teacher 的 proximal inequality pass `100%`，并保留了已安全 proposal；但 student 只覆盖约 `58%~60%` 的 teacher-improved 样本，不能达到 strict amortization。

**这一组实验真正证明的是：**privileged teacher 在局部可以做出安全修正；**没有证明：**deployment-visible student 能可靠复制修正。

### 4.4 teacher 动力学和 coarse-to-fine

- R1 重复更新在 K=`20` 时 nominal strict `8.6%`、robust `5.5%`，但 constraint regression `17.2%`、reversal `25.8%`；严重 reversal 中 `83.4%` 来自 same-field update reversal。
- R0-P 只有局部 teacher gate 通过，R0-G regression `7.8%` 失败；没有 student 或正式环境准入证据。
- coarse-to-fine 的筛选目标 Safe@1 `5.6% -> 48.4%`，invalid-to-valid `45.3%`，但 R2-A 未执行且 PMF 未接入，最终 gate `False`。
- boundary residual 在 development 条件上 Safe@1 `+4.30 pp`，CI `[+0.78,+8.60] pp`，但仍未达到预注册 Safe@1 门槛；其他版本 CI 下界为 0。

原始记录：[`stage2_operator_audit_physical_w02_20260729`](../../tests/audits/stage2_operator_audit_physical_w02_20260729/REPORT.md)、[`structured_constraint_correction_20260731`](../../tests/audits/structured_constraint_correction_20260731/REPORT.md)、[`joint_operator_amortization_20260731`](../../tests/audits/joint_operator_amortization_20260731/REPORT.md)、[`ppcd_corrected_main_20260730`](../../tests/audits/ppcd_corrected_main_20260730/REPORT.md)、[`ppcd_corrected_k3_student_20260730`](../../tests/audits/ppcd_corrected_k3_student_20260730/REPORT.md)。

## 5. 路线四：扩大目标分布、latent transport、XMeanFlow 和 NFT

### 想解决什么

前几条路线可能不是“优化不够”，而是候选池太窄或目标不够丰富。因此尝试扩大 target pool、寻找 latent repair、蒸馏 teacher distribution、使用 XMeanFlow/NFT 或 reference-guided transport。

### 5.1 共同结果：目标上限有时很好，但 student/正式 gate 没过

| 方向 | 看起来有希望的地方 | 没有通过的依据 |
| --- | --- | --- |
| latent repair | finite-budget 上限：existing pool `62.5%`，latent search `81.25%`，path correction `100%` | observability AUC `0.557`；repair NRMSE `1.000`；improved-over-identity `0%`；student 没有恢复 teacher gain。 |
| broad latent target | teacher audit 可通过 | student Safe@1 与 baseline 相同，CI `[0,0]`；repair NRMSE 约 `1.0`。 |
| source-coupled repair | 有 privileged transport 目标 | Safe@1 `18.75% -> 17.19%`，CI `[-4.69,0]`；student safe-to-unsafe `7.04%`。 |
| teacher distribution | 个别 teacher 相对 Stage 1 有 `+6.25~+7.81 pp` 点估计 | teacher-vs-uniform CI 下界为 0；部分版本 mode retention `0.757`，S->I `9.09%~17.27%`。 |
| XMeanFlow | 个别 K8-K1 只有小幅正差值 | XM-T0 `+0.23 pp`，CI `[-0.62,+1.17]`；Broad K8-K1 `+1.05 pp`，CI `[-0.90,+3.01]`，同时 curvature/diversity gate 失败。 |
| NFT | gradient contract、finite difference/JVP 一致 | held-out cost `1.8603 -> 1.8966`；累计 gain CI `[0,0.586]`；fixed-point 和 frozen-source improvement 失败。 |
| reference-guided transport | 59/64 条件有 safe reference；294/294 source-reference pair 可做 hard-safe transport | 这是 privileged teacher 上限；尚未训练出可部署 student。 |
| source pairing | 短训 PMF pairing 相对 geometry pairing Safe@16 `+3.57 pp` | CI `[0,+10.71]`；medium conditional gain `+1.47 pp`，CI `[-1.47,+5.15]`，不是完整分布准入。 |

### 5.2 这组实验的结论

这些路线支持一个较窄的结论：**当前候选池和 privileged target 还有可挖掘空间。** 但它们没有证明 deployment-visible 输入能够识别 latent safe direction，也没有证明 student 能以足够小的误差复制 teacher。不能用 teacher upper bound、target coverage 或 reference transport 成功来替代 Safe@1 admission。

原始记录：[`stage2_latent_repair_observability_v1_20260803`](../../tests/audits/stage2_latent_repair_observability_v1_20260803/REPORT.md)、[`stage2_broad_latent_targets_v1_20260803`](../../tests/audits/stage2_broad_latent_targets_v1_20260803/REPORT.md)、[`stage2_broad_latent_target_student_v2_gated_20260803`](../../tests/audits/stage2_broad_latent_target_student_v2_gated_20260803/REPORT.md)、[`stage2_teacher_distribution_lastblock_v6_20260803`](../../tests/audits/stage2_teacher_distribution_lastblock_v6_20260803/REPORT.md)、[`privileged_xmeanflow_broad_pmf_b1_20260804`](../../tests/audits/privileged_xmeanflow_broad_pmf_b1_20260804/report.md)、[`privileged_path_meanflownft_accumulated_20260803`](../../tests/audits/privileged_path_meanflownft_accumulated_20260803/report.md)、[`reference_guided_transport_20260805`](../../tests/audits/reference_guided_transport_20260805/REPORT.md)。

## 6. 路线五：mask-conditioned M0/M1

### 想解决什么

让模型根据 mask 区分不同约束组合，期待它比一个统一的 correction 更容易学习。

### 实际判定

M0 按协议只做 audit，不训练网络；M1 只有在 M0 全部通过后才能训练。

- 80 个 mask groups，accepted attempts `210/320`；
- B+C 的 train consistency `54.2%`，validation `20.8%`，gap `33.4 pp`，超过 `20 pp` 上限；
- D 的 train/validation 都为 `25.0%`；target sensitivity 为 `95.9%/92.6%`；
- 唯一失败 gate 是 `bc_train_validation_consistency`。

**结论：**M0 `FAIL`，原因是 B+C 跨 split 不一致，不是 mask 已被证明不可行。M1 `BLOCKED`，student updates 为 `0`；不能把未运行的 M1 写成失败实验，也不能写成成功实验。

原始记录：[`m0_mask_conditioned_targets`](../../tests/audits/m0_mask_conditioned_targets/REPORT.md)、[`m1_mask_conditioned_stage2`](../../tests/audits/m1_mask_conditioned_stage2/REPORT.md)。

## 7. 支持上限、证据边界和无效记录

### 7.1 support 能说明什么，不能说明什么

- proposal repairability：同模式 nominal repairable `33.8%`，robust target coverage `36.4%`，当前预算下 unresolved `55.8%`。
- 从 16 增加到 512 draws 后 support rate `81.25% -> 96.81%`，仍有 51 个 input observed-zero-safe。
- `observed-zero-safe` 只表示有限池没有观测到 safe candidate；不能写成真实 support 不存在。

把当前证据分成三层更准确：

| 层次 | 已经知道什么 | 还不知道什么 |
| --- | --- | --- |
| 候选池 | 某些 condition 的池内安全质量可以被重排 | 是否有更多未采样的 safe support。 |
| teacher/target | privileged teacher 能在部分样本找到更安全结果 | deployment-visible 输入能否判断该朝哪个方向修。 |
| student/system | D1、residual student、sealed system 尚未确认绝对增益 | 是否存在可稳定部署的 Stage 2。 |

### 7.2 这些记录不能混用

- 旧 `stage2_coupling` 使用的 Stage 1 checkpoint 训练过 dataset0 全部 100 张地图；它不是严格 unseen-map OOD，只能作机制诊断。
- 旧 `direct_safe_distribution` 的 2x2 A/B/C/D 没有完整同 seed 矩阵，不能回答最终研究问题。
- reference contract 在 2026-07-31 至 2026-08-02 发生过重验和修正；旧 direct coarse、旧 repairability headline 不能直接沿用到当前 cost/target semantics。
- `Best-Cost@K`、Oracle Safe@K、teacher safe rate、frozen target weighted mass 都是 privileged upper bound/diagnostic，不是部署 Safe@1。

### 7.3 工程错误和科学失败要分开

- PPCD 早期误用 historical zero-sum projection，造成约 `1.25 m` reconstruction error；非 `corrected` 目录是 `INVALID`，不能与 corrected archive 合并统计。
- M0 初始 batch geometry decode 没有扩展 start/goal，触发 `start_direction must have shape (B,2)`；修复后按同一 frozen manifest 重跑。
- D0 confirm 初期 aggregate reporting 遇到空 train group 的 `KeyError: baseline_safe_mass`；修复为跳过空 split，不改变实验判定。
- direct-cost 曾出现 `PytorchStreamWriter failed writing file`，原因是磁盘空间不足；这是 artifact 风险，不是性能证据。
- 当前工作区仍有用户/历史删除和未跟踪变更；本文没有恢复或清理它们，只读取现有审计、当前 docs 和 Git 中的旧报告。

## 8. 以后重新提案时的自查顺序

把一次新的 Stage 2 尝试按下面顺序检查；任何一层失败，都不要用下一层的正向数字替它改判。

1. **support**：有限候选池是否真的有可用 safe support？若没有，是否只写 `observed-zero-safe`？
2. **target**：teacher 或 target operator 是否稳定？是否报告了回归、mode retention 和重复应用后的行为？
3. **student**：student 是否能在独立 source/condition 上复制 target？是否有 uniform D1-U control？
4. **system**：部署可见输入下的 Safe@1、strict-valid 和 environment-bootstrap CI 是否相对 Stage 1 为正？
5. **隔离**：environment-disjoint manifest、train/validation/test、selection/fit、source lineage 是否分开？
6. **冻结**：Stage 1 checkpoint、cost contract、mask、source、checkpoint selection 规则是否预先冻结？
7. **回归**：是否同时报告 invalid-to-strict、strict-to-invalid、单约束 regression、diversity、mode retention 和 source sensitivity？
8. **停止规则**：前置 gate 失败后是否停止后续 student training；是否保留失败 checkpoint 和 exact manifest，而不是事后挑漂亮 update？

最终判定顺序应保持为：

```text
有限池 support
      -> teacher / target 稳定
            -> student 能摊销
                  -> 完整系统 Safe@1 通过
```

目前已有的是部分箭头上的正证据，最后一个箭头尚未通过。

## 9. 原始记录索引

### 直接 cost 与保护变体

- [`direct_privileged_cost_no_anchor_20260729`](../../tests/audits/direct_privileged_cost_no_anchor_20260729/REPORT.md)
- [`draft_direct_cost_paired_20260802`](../../tests/audits/draft_direct_cost_paired_20260802/report.md)
- [`a0_balanced_direct_cost`](../../tests/audits/a0_balanced_direct_cost/REPORT.md)
- [`a0_multipair_controllability`](../../tests/audits/a0_multipair_controllability/REPORT.md)
- [`direct_cost_condition_margin_v2_20260805`](../../tests/audits/direct_cost_condition_margin_v2_20260805/REPORT.md)
- [`all_direct_v11_external_admission_20260805`](../../tests/audits/all_direct_v11_external_admission_20260805/REPORT.md)
- [`all_direct_condition_coverage_v1_20260805`](../../tests/audits/all_direct_condition_coverage_v1_20260805/REPORT.md)
- [`all_direct_conditioning_normmean_worstcore_v11_checkpointed_20260805`](../../tests/audits/all_direct_conditioning_normmean_worstcore_v11_checkpointed_20260805/REPORT.md)
- [`all_direct_safe_core_backtracking_v7_20260805`](../../tests/audits/all_direct_safe_core_backtracking_v7_20260805/REPORT.md)
- [`all_direct_safe_replay_projection_v2_20260805`](../../tests/audits/all_direct_safe_replay_projection_v2_20260805/REPORT.md)
- [`unsafe_only_direct_cost_v2_20260805`](../../tests/audits/unsafe_only_direct_cost_v2_20260805/REPORT.md)

### selector、D0/D1、critic 与系统 gate

- [`candidate_selector_admission_20260730`](../../tests/audits/candidate_selector_admission_20260730/REPORT.md)
- [`stage2_candidate_selection_v12_20260802/DECISION`](../../tests/audits/stage2_candidate_selection_v12_20260802/DECISION.md)
- [`d0_nonparametric_reweighting_20260730`](../../tests/audits/d0_nonparametric_reweighting_20260730/REPORT.md)
- [`d0_confirm_kl02_20260730`](../../tests/audits/d0_confirm_kl02_20260730/REPORT.md)
- [`d1_uniform_weighted_20260730`](../../tests/audits/d1_uniform_weighted_20260730/REPORT.md)
- [`conservative_rwpmf_d1_m0_20260803`](../../tests/audits/conservative_rwpmf_d1_m0_20260803/report.md)
- [`conservative_rwpmf_target_v1_20260803`](../../tests/audits/conservative_rwpmf_target_v1_20260803/report.md)
- [`stage2_pareto_adapter_v2_20260802`](../../tests/audits/stage2_pareto_adapter_v2_20260802/REPORT.md)
- [`stage2_path_aligned_reweighting_v1_20260803`](../../tests/audits/stage2_path_aligned_reweighting_v1_20260803/REPORT.md)
- [`stage2_path_aligned_reweighting_confirm_20260803`](../../tests/audits/stage2_path_aligned_reweighting_confirm_20260803/REPORT.md)
- [`stage2_sealed_system_acceptance_20260804`](../../tests/audits/stage2_sealed_system_acceptance_20260804/REPORT.md)
- [`stage2_stability_critic_external_v3_20260803`](../../tests/audits/stage2_stability_critic_external_v3_20260803/REPORT.md)
- [`stage2_stability_critic_pool_kl_v3_20260803`](../../tests/audits/stage2_stability_critic_pool_kl_v3_20260803/REPORT.md)

### teacher、修正、latent 与 student

- [`stage2_operator_audit_physical_w02_20260729`](../../tests/audits/stage2_operator_audit_physical_w02_20260729/REPORT.md)
- [`structured_constraint_correction_20260731`](../../tests/audits/structured_constraint_correction_20260731/REPORT.md)
- [`joint_operator_amortization_20260731`](../../tests/audits/joint_operator_amortization_20260731/REPORT.md)
- [`direct_coarse_to_fine_20260731`](../../tests/audits/direct_coarse_to_fine_20260731/REPORT.md)
- [`ppcd_corrected_main_20260730`](../../tests/audits/ppcd_corrected_main_20260730/REPORT.md)
- [`ppcd_corrected_k3_student_20260730`](../../tests/audits/ppcd_corrected_k3_student_20260730/REPORT.md)
- [`stage2_broad_latent_targets_v1_20260803`](../../tests/audits/stage2_broad_latent_targets_v1_20260803/REPORT.md)
- [`stage2_broad_latent_target_student_v2_gated_20260803`](../../tests/audits/stage2_broad_latent_target_student_v2_gated_20260803/REPORT.md)
- [`stage2_latent_repair_observability_v1_20260803`](../../tests/audits/stage2_latent_repair_observability_v1_20260803/REPORT.md)
- [`stage2_latent_transport_v1_20260803`](../../tests/audits/stage2_latent_transport_v1_20260803/REPORT.md)
- [`stage2_source_coupled_repair_v1_20260803`](../../tests/audits/stage2_source_coupled_repair_v1_20260803/REPORT.md)
- [`stage2_multistart_target_sets_v1_20260803`](../../tests/audits/stage2_multistart_target_sets_v1_20260803/REPORT.md)

### mask、support 与其他分布机制

- [`m0_mask_conditioned_targets`](../../tests/audits/m0_mask_conditioned_targets/REPORT.md)
- [`m1_mask_conditioned_stage2`](../../tests/audits/m1_mask_conditioned_stage2/REPORT.md)
- [`stage1_proposal_repairability_20260731`](../../tests/audits/stage1_proposal_repairability_20260731/REPORT.md)
- [`stage1_large_capsizing_support_v1_20260803`](../../tests/audits/stage1_large_capsizing_support_v1_20260803/REPORT.md)
- [`stage2_teacher_distribution_lastblock_v6_20260803`](../../tests/audits/stage2_teacher_distribution_lastblock_v6_20260803/REPORT.md)
- [`stage2_curvature_soft_filter_v2_20260803`](../../tests/audits/stage2_curvature_soft_filter_v2_20260803/REPORT.md)
- [`stage2_continuous_alpha_audit_20260803`](../../tests/audits/stage2_continuous_alpha_audit_20260803/REPORT.md)
- [`stage2_t2_critic_hard_selection_20260804`](../../tests/audits/stage2_t2_critic_hard_selection_20260804/REPORT.md)
- [`privileged_xmeanflow_broad_pmf_b1_20260804`](../../tests/audits/privileged_xmeanflow_broad_pmf_b1_20260804/report.md)
- [`privileged_path_meanflownft_accumulated_20260803`](../../tests/audits/privileged_path_meanflownft_accumulated_20260803/report.md)
- [`reference_relative_online_meanflownft_m0_128_20260803`](../../tests/audits/reference_relative_online_meanflownft_m0_128_20260803/report.md)
- [`reference_guided_transport_20260805`](../../tests/audits/reference_guided_transport_20260805/REPORT.md)
- [`reference_guided_pmf_error_calibration_20260805`](../../tests/audits/reference_guided_pmf_error_calibration_20260805/REPORT.md)
- [`low_cost_source_pairing_short_v2_20260804`](../../tests/audits/low_cost_source_pairing_short_v2_20260804/REPORT.md)

### 当前协议

- [`STAGE2_RESEARCH_PROTOCOL.md`](STAGE2_RESEARCH_PROTOCOL.md)
- [`STAGE2_PATH_ALIGNED_REWEIGHTING.md`](STAGE2_PATH_ALIGNED_REWEIGHTING.md)
- [`STAGE2_POSTERIOR_TRANSPORT_SURVEY_20260802.md`](STAGE2_POSTERIOR_TRANSPORT_SURVEY_20260802.md)
- Git HEAD 中的旧 coupling 结论（当前工作树已标记删除）：`diagnostics/stage2_coupling_conclusion.md`、`diagnostics/stage2_failure_decomposition_report.md`
