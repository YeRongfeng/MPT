# Stage 2 正式研究协议

状态：**当前有效，待实验验证**  
固定日期：**2026-07-30**  
当前修订：**D0 direct-cost 长程审计完成；历史 D / B+D 结果保留**

本文是当前 Stage 2 研究路线的唯一权威说明。它定义研究假设、对照、预算、
指标和准入规则，但不声称 Stage 2 已经成立或已经进入正式 Method。

## 0. 2026-08-02 D0 执行状态更新

### 0.1 长程 direct-cost 结论

在修复解析曲率和低速 yaw 的病态反向梯度后，使用冻结环境划分、source、
学习率和 1200-update 预算完成了正式长程复现。修复只改变软目标的梯度条件：

- hard curvature 仍以精确解析审计 `max(kappa) <= 2.1` 判定；
- stability map 的 yaw 前向值仍是精确 `atan2(p'_y,p'_x)`；
- 曲率软项改为同零集的代数裕度并使用 `log1p` 压缩远离边界的异常梯度；
- yaw 仅在低于既有最小线段速度时对反向分母设置下限。

最终运行的裁剪前梯度中位数为 25.75、q95 为 85.97、最大值为 177.79，
不再出现旧实现的 95,118 尖峰。训练数值链路因此判定为正常。

但是方法准入没有通过。固定 validation 上：

- task cost：`1.7663 -> 0.8894`；
- strict-valid：`20.70% -> 28.52%`；
- Safe@1：`25.00% -> 18.75%`；
- invalid->strict：`21.67%`；
- strict->invalid：`45.28%`；
- stability regression：`34.44%`；
- curvature regression：`31.58%`。

update 1030 曾得到 Safe@1 `43.75%`、strict-valid `36.72%`，但同时
strict->invalid 为 `47.17%`、stability regression 为 `33.89%`，不能被
事后提升为通过。全部非零 update 中没有任何一个同时满足冻结回归上限；
`stage2_best.pth` 因而正确保留 update 0 基线。

当前结论是：

> 可微 privileged cost 能够直接驱动 Path MeanFlow 改善平均 cost 和部分
> 分布级安全率，但无锚点、无分布保持项的 pure direct-cost 会以大量原安全
> source 回归为代价。它是成立的优化机制，不是可准入的完整 Stage 2。

该结论阻断继续增加 pure D0 预算，但不否定以后单独预注册带分布保持约束的
direct-cost 变体。final-test environment 继续保持关闭。

### 0.2 短程接入依据（历史状态）

在完成环境互斥的短程 direct-cost 配对验证后，当前正式代码中的可执行
Stage 2 候选改为 D0：在部署点 `t=1,r=0` 上直接反向传播 privileged task
cost。该决定覆盖本文后续章节中“direct-cost 仅作为机制对照”的即时执行
安排，但不改写任何历史 gate 结果。

D0 的独立正证据是 held-out proposal strict-valid 从 17.97% 提高到 26.56%，
且短程内没有观察到 source-response collapse；其冻结 screen 仍因
strict->invalid=6.52% 和 curvature regression=6.85% 而失败。因此 D0 被接入
正式训练代码供扩大训练观察，不等于 Stage 2 已准入或最终 Method 已成立。

正式训练必须保留：Stage-2 物理环境互斥划分、固定 source 配对、初始 Stage-1
baseline、invalid->strict、strict->invalid、单约束 regression、source 路径
距离和 covariance effective rank。超过冻结 regression 上限的模型只能保存为
`stage2_last.pth`，不得覆盖 `stage2_best.pth`。

## 1. 当前正式状态

当前 Method 为：

$$
\boxed{\text{单次条件路径生成，部署主指标为 Safe@1。}}
$$

对条件 \(c\)，Stage 2 真正需要改善的是：

$$
p_\theta^{\mathrm{safe}}(c)
=
\Pr_{z\sim\mathcal N(0,I)}
\left[G_\theta(c,z)\in\mathcal F\right].
$$

Oracle Safe@K 只用于诊断：

- 条件分布中是否存在尚未利用的可行路径；
- 多次采样是否提供额外覆盖；
- Stage 2 是否发生明显模式收缩。

Oracle Safe@K 不是部署性能。候选选择器不进入训练链路、推理链路或主结果
表。其独立准入试验没有通过，证据见
[`tests/audits/candidate_selector_admission_20260730/REPORT.md`](../../tests/audits/candidate_selector_admission_20260730/REPORT.md)。

Stage 2 当前的**待验证假设**是：

$$
\boxed{
\begin{gathered}
\text{在保持 Stage 1 条件分布基本结构的前提下，}\\
\text{通过局部可行支持扩充与相对熵约束的分布重加权，}\\
\text{逐步提高安全路径所占的条件概率质量。}
\end{gathered}}
$$

不得把这一假设写成已经成立的方法性质。

## 2. 问题的两种视角

### 2.1 逐路径移动

直接 cost 或修正目标方法把问题写成：

> 怎样把每条路径沿违反量下降的方向移动？

这类方法可以扩展当前支持，但分别存在无约束收缩或移动目标不连续的问题。

### 2.2 条件分布质量重分配

分布方法把问题写成：

> 怎样在保留当前多模态路径分布的前提下，把更多概率质量分配给安全或
> 低违反路径？

这一表述更直接对应 Safe@1、无选择器和防坍缩要求。当前首要候选不再是
B 单独使用，而是：

$$
\boxed{
\text{B+D：局部可行支持扩充}
+
\text{相对熵约束的可行性重加权分布蒸馏}.
}
$$

## 3. 相对熵约束的分布更新

固定条件 \(c\)，令当前模型诱导：

$$
y=G_{\theta_k}(c,z),\qquad
y\sim q_k(y\mid c).
$$

设 \(V(y,c)\ge0\) 为特权违反量。考虑：

$$
\begin{aligned}
q_{k+1}^{\star}
=
\arg\min_q\quad&
\mathbb E_{y\sim q(\cdot\mid c)}[V(y,c)],\\
\text{s.t.}\quad&
D_{\mathrm{KL}}
\left(q(\cdot\mid c)\,\|\,q_k(\cdot\mid c)\right)
\le\varepsilon .
\end{aligned}
$$

其拉格朗日形式为：

$$
\min_q\;
\mathbb E_q[V]
+
\eta D_{\mathrm{KL}}(q\,\|\,q_k),
$$

非参数最优解为：

$$
\boxed{
q_{k+1}^{\star}(y\mid c)
=
\frac{
q_k(y\mid c)\exp[-V(y,c)/\eta]
}{
Z(c)
}.}
$$

该更新：

- 增加当前支持中低违反路径的概率；
- 减少高违反路径的概率；
- 以当前分布为参考限制单轮信息损失；
- 不需要定义逐路径的远距离修正映射；
- 不把 privileged cost 梯度反向传播到生成器。

它受 REPS/MPO 一类保守分布更新思想启发，但当前 Path MeanFlow 问题不是
标准 MDP 策略优化，不能直接继承这些算法的策略改进保证。

## 4. D：可行性重加权 Path MeanFlow

### 4.1 固定候选池

冻结 \(G_{\theta_k}\)，对每个训练条件采样：

$$
y_i=G_{\theta_k}(c,z_i),\qquad i=1,\ldots,M.
$$

使用训练期特权信息计算：

$$
V_{\max,i},\qquad
V_{\mathrm{int},i},\qquad
F_i^{\mathrm{robust}}\in\{0,1\}.
$$

候选池、condition、environment、mask、模型 checkpoint 和所有随机 source
必须写入固定 manifest。

### 4.2 按条件归一化

每个条件独立计算：

$$
\bar w_i
=
\frac{\exp(-V_i/\eta)}
{\sum_{j=1}^{M}\exp(-V_j/\eta)}.
$$

不得在整个数据集上统一排序或归一化，否则容易把训练质量集中到少数简单
条件，而不是改善每个条件内部的概率质量。

\(V_i\) 的定义必须在打开 held-out 结果前冻结。至少比较：

- 连续 \(V_{\max}/V_{\mathrm{int}}\) 能量；
- robust-feasible 优先、连续违反量次级的分层能量。

连续违反量下降不自动保证 Safe@1 上升，最终准入仍以 held-out Safe@1
为准。

### 4.3 相对熵或 ESS 预算

在有限候选池中，旧经验分布为均匀分布。记录：

$$
D_{\mathrm{KL}}
\left(
\bar w\,\middle\|\,\frac1M
\right)
=
\sum_i\bar w_i\log(M\bar w_i)
$$

和：

$$
\mathrm{ESS}
=
\frac{1}{\sum_i\bar w_i^2}.
$$

通过二分求解 \(\eta\)，使经验 KL 不超过固定预算，或使
\(\mathrm{ESS}/M\) 不低于固定比例。温度、容差、二分次数和退化处理必须
写入 manifest。

必须明确：

> 候选权重相对均匀分布的经验 KL 只约束有限池上的目标重加权，不等于
> 已经控制拟合后 \(q_{\theta_{k+1}}\) 与 \(q_k\) 的真实分布 KL。

有限池偏差、函数逼近误差和优化误差仍需用部署采样与分布指标实测。

### 4.4 使用重新采样的 source

加权 Path MeanFlow 目标为：

$$
\mathcal L_D
=
\sum_{i=1}^{M}
\bar w_i\,
\mathcal L_{\mathrm{PathMF}}
\left(\theta;c,z_i',y_i\right),
$$

其中：

$$
z_i'\sim\mathcal N(0,I)
$$

必须独立于最初生成 \(y_i\) 的 source \(z_i\)。

若继续训练原配对
\(z_i\rightarrow G_{\theta_k}(c,z_i)\)，模型主要是在复现自身函数，
不能把高质量路径可靠地扩展到新的 source 质量。使用独立 source 或按
\(\bar w\) 重采样目标，拟合的才是重加权后的条件目标边缘分布。

当前 `meanflow_transport_loss` 支持显式 source–target 配对；D 必须新建
source，而不能复用候选 lineage source。

### 4.5 Stage 1 锚定

D 同时混入固定比例的 Stage 1 原始目标或函数锚定，以降低有限池过拟合。
锚定数据、比例、损失权重和更新预算必须与其他对照共用选择协议。

## 5. B：局部可行支持扩充

冻结第 \(k\) 轮模型并生成：

$$
y=G_{\theta_k}(c,z).
$$

构造停止梯度的局部目标：

$$
\begin{aligned}
y^+ &= y+\Delta y,\\
\Delta y
&=
\operatorname{Proj}_{\lVert\Delta\rVert_G\le\delta}
\left(-\eta_y G^{-1}\nabla_yJ(y)\right).
\end{aligned}
$$

只有同时满足以下条件的目标才可加入支持池：

$$
V(y^+)<V(y),\qquad
d_G(y^+,y)\le\delta,
$$

并且：

- 不改变主要绕行侧或同伦模式；
- 修正前后保持明确的一一对应；
- 不把多个不同模式合并到同一解。

B 的职责不是为所有 source 提供标签，也不要求单步 strict-valid。它只向
当前条件的目标池加入少量安全或近安全支持。

## 6. 固定修正示范：可行域分布预热

固定修正示范的正式定位是：

$$
\boxed{\text{feasibility-oriented distribution warm start}}
$$

它不是正式 on-policy Stage 2。对每条示范
\(y_{\mathrm{demo}}\) 独立构造
\(y_{\mathrm{demo}}^{\mathrm{ref}}\)，并记录：

- \(d_{\mathrm{path}}(y_{\mathrm{demo}}^{\mathrm{ref}},
  y_{\mathrm{demo}})\)；
- \(V_{\max}\)、\(V_{\mathrm{int}}\) 的下降；
- 模式保持率；
- 修正前后的示范分布变化。

若多数示范需要大跨度或跳模式修正，应重新审视 Stage 1 的监督分布。
通过准入的固定修正示范可以加入 B+D 的支持池，但不能替代 Stage 1 锚定。

## 7. B+D：当前首要候选

每个条件的目标池为：

$$
\mathcal Y_c
=
\left\{
\text{Stage 1/当前候选},
\text{通过准入的修正示范},
\text{通过准入的局部修正候选}
\right\}.
$$

先扩充支持，再在整个条件目标池内按相同的 KL/ESS 预算重加权：

$$
w_i\propto\exp(-V_i/\eta).
$$

其中：

- B/固定修正示范负责创造或扩充可行支持；
- D 负责把概率质量逐步迁移到可行支持；
- KL/ESS 预算限制有限池上的过度集中；
- Stage 1 锚定保护未被有限池充分表达的原分布；
- 推理仍然只有一次 Path MeanFlow 前向。

B+D 不要求每个原 source 都对应一个远距离专家目标，也不使用候选选择器。

## 8. 其他固定对照

### A. 直接约束反向传播

$$
\mathcal L_A
=
\mathcal L_{\mathrm{anchor}}
+
\lambda_J
\mathbb E_{c,z}
\left[J\!\left(G_\theta(c,z)\right)\right].
$$

A 验证约束梯度能否直接改善 Safe@1，以及 Stage 1 锚定能否抑制条件分布
收缩。它是必要基线，不是默认最终方案。

当前证据状态固定为：

> **A0 direct-cost single-pair controllability passed**

该结论只表示：对一个经端点可行性过滤、且由正式路径 expert 确认为可修正的
固定 `(condition, mask, source)`，直接 privileged cost 参数反传能够把
Stage 1 输出从 strict-invalid 推到 strict-valid。它不构成分布训练通过、
held-out Safe@1 改善或正式 Stage 2 准入。

在提升结论前必须依次通过：

1. `grad_optimizer.py` 与 `train_flow.py` 的 cost、梯度、单步更新及多步输出
   等价性回归；
2. privileged cost 与 strict-valid 的阈值一致性及有限差分梯度审计；
3. 无 expert 筛选的多 pair 联合可控性，分别报告已训练 pair、同 condition
   新 source、新 condition；
4. 冻结 manifest 下的 A0-Cost、A0-Stage1、A0-Mixed 短程对照；正式准入只
   接受 held-out environment Safe@1 环境 bootstrap 增益下界大于零。

截至当前冻结实验，前两项已通过；第 3 项在 manifest
`9859a3291262f36a89efe25b6fb59f2477e880ddae50b09e076a4ddebfcff6c4`
上未通过。已训练 pair 和同 condition 新 source 通过，但新 condition 的
Safe@1 从 `12.5%` 降至 `0.0%`，平均 privileged cost 上升 `17.8%`。
因此不得启动或解释第 4 项三臂正式准入。

### C. 完整安全目标蒸馏

$$
y^{\mathrm{ref}}=T_{\mathrm{full}}(y).
$$

C 用于确认远距离目标跨度和修正映射不连续是否是主要失败原因。已有审计
已经发现明显不连续性，因此 C 只保留小规模对照。

若 C 使用缩小后的 manifest，其他参与该比较的方案也必须在同一 manifest
和预算上重跑；不得把小规模 C 与大规模 A/B/D/B+D 直接比较。

### E. 可行性条件生成

后续可训练：

$$
q_\theta(y\mid c,s),
$$

其中 \(s\) 为 robust-feasible、near-feasible 或 invalid 等质量等级。
推理时指定期望等级仍是单次生成，不是选择器。

但 robust-feasible 样本当前较少，且存在条件忽略和分布边缘外推风险，因此
E 只作为后续对照，不进入近期主线。

## 9. 各方案能力边界

| 方案 | cost 梯度进入生成器 | 逐路径专家目标 | 分布保护 | 可扩展支持 |
|---|---:|---:|---:|---:|
| A：直接 cost | 是 | 否 | 弱/依赖锚定 | 是 |
| C：完整修正蒸馏 | 否 | 是，且可能远距离 | 依赖 lineage | 是 |
| B：有界局部修正 | 否 | 少量局部目标 | 中等 | 是 |
| D：可行性重加权 | 否 | 否 | 经验 KL/ESS 可控 | 否 |
| B+D：支持扩充后重加权 | 否 | 仅少量局部目标 | 当前最强 | 是 |
| E：可行性条件生成 | 否 | 否 | 取决于数据 | 取决于数据 |

D 不能凭空创造当前分布中不存在的可行模式。若某条件所有候选均严重不可行，
D 只能提高“相对较低违反”路径的权重。必须记录：

- 无 robust-feasible 候选的条件比例；
- 无 near-feasible 候选的条件比例；
- 权重退化到单候选的比例；
- 低支持条件对总体训练 loss 的贡献。

这些是 B+D 需要支持扩充的直接依据。

## 10. 公平比较与双预算

### 10.1 学生训练预算

A/B/D/B+D 必须保持：

- 相同初始 Stage 1 checkpoint；
- 相同 condition、environment、mask 和基础 source manifest；
- 相同 batch 数和参数更新次数；
- 相同优化器及学习率选择协议；
- 相同 Stage 1 锚定数据和锚定权重选择协议。

D/B+D 的新训练 source 使用统一但独立的固定随机种子。

### 10.2 特权计算预算

必须分别统计：

- cost 前向次数；
- cost 梯度计算次数；
- 路径修正优化步数；
- 候选池大小；
- 总特权计算时间。

D 主要使用 cost 前向，A 每个学生更新可能调用 cost 及梯度，B/C 在目标
收集时调用梯度或优化器。只对齐学生 update 不构成公平比较。

最终同时报告：

$$
\text{performance vs. student updates}
$$

和：

$$
\text{performance vs. privileged cost evaluations}.
$$

还应单独报告 wall-clock，而不能把 cost 前向和 cost 梯度简单视为等价操作。

## 11. 指标及优先级

### 11.1 主性能

- Safe@1：唯一部署主指标；
- Oracle Safe@8：覆盖和模式诊断，不是系统性能。

### 11.2 分布保护

按以下优先级判断有害坍缩：

1. 路径模式覆盖率；
2. 不同模式的保留率和占比变化；
3. 条件内稠密路径两两距离；
4. 条件协方差有效秩；
5. Oracle Safe@8 与 Safe@1 的差距；
6. source Jacobian 能量。

Jacobian 只作辅助信号。只要不同有效模式仍被保留，同一模式内部无意义扰动
的减少不应被判定为坍缩。

原始总模式覆盖仍是预注册红线，不能在看到结果后更改。同时额外报告
viable-mode coverage 作为解释性指标。一个模式只有至少包含以下之一时才
计为 viable：

- strict-safe 路径；
- 按冻结阈值定义的 near-safe 路径；
- 可由 B 在冻结移动上界内修正的路径。

viable-mode coverage 不替代总模式覆盖，只用于判断被削弱的是有价值模式
还是持续远离可行域的失败模式。

还必须记录：

- Stage 1 完整地图和非阻断任务回退；
- 路径长度、曲率及基础几何质量；
- 每种模式内的安全概率质量变化；
- D/B+D 的权重熵、经验 KL、ESS 和每条件权重最大值。

## 12. D 与 B+D 的准入判据

### 当前 D0 证据状态

2026-07-30 的第一轮 D0 在程序上未通过：训练分区选出的
`strict_lexicographic + KL=0.3` 将 validation 经验 Safe@1 从
7.71% 提高到 14.01%，但总模式覆盖保留率为 89.9%，低于预注册 90%
红线。该结果不能事后追认为通过。

这次失败不否定 D 的核心假设。随后使用同时更换 condition、lineage source
和 mask 随机性的独立 manifest，固定 `strict_lexicographic + KL=0.2`
完成 D0-confirm：

- 经验 Safe@1：15.21% → 22.87%；
- 提升：+7.66 个百分点；
- environment bootstrap 95% 下界：+4.37 个百分点；
- ESS/M 最小值：0.6395；
- 总模式覆盖保留率：92.0%。

D0-confirm 已通过。因此当前定性更新为：

$$
\boxed{\text{D 的非参数分布重加权假设已通过独立确认。}}
$$

### 12.0 D0：无训练的非参数准入

在启动任何 D 学生训练前，先在冻结候选池上直接计算：

$$
\hat p_{\mathrm{safe}}^{\,w}(c)
=
\sum_i w_i\mathbf 1[y_i\in\mathcal F].
$$

D0 必须同时报告均匀经验分布的安全质量、重加权安全质量、经验 KL、ESS、
最大单样本权重、模式覆盖、各模式概率质量和模式熵：

$$
H_{\mathrm{mode}}=-\sum_m p_m\log p_m.
$$

候选索引上的 ESS 不能单独作为模式保护证据。即使多个高权重样本属于同一
几何绕行模式，总 ESS 仍可能很高。

条件必须分为：

$$
\mathcal C_{\mathrm{has\text{-}safe}}
\quad\text{和}\quad
\mathcal C_{\mathrm{observed\text{-}zero\text{-}safe}}.
$$

对前者报告预算内可达到的安全质量提升；对后者报告条件比例以及池内最小
\(V_{\max}\) 和最小 \(V_{\mathrm{int}}\) 的分布。有限池内观测到零安全
候选只说明该经验池的加权安全质量恒为零，不能声称真实
\(q_{\mathrm{Stage1}}(y\mid c)\) 没有安全支持。若每条件独立采样 \(M\)
次仍观测为零，二项分布的粗略 95% 上界为：

$$
p_{\mathrm{safe}}(c)\lesssim\frac{3}{M}.
$$

违反能量至少比较：

$$
E_{\mathrm{cont}}(y)
=
\lambda_{\max}V_{\max}(y)
+
\lambda_{\mathrm{int}}V_{\mathrm{int}}(y)
$$

和 strict-feasible 优先、连续违反量次级的分层能量。打开验证结果前应先
验证能量排序确实增加 strict-safe mass，而不只是降低平均 cost。

D0 的配置选择只使用训练环境；选定的能量和预算再在 environment-disjoint
validation 环境上进行一次准入判断。若所有满足 KL/ESS 和模式保护预算的
配置均不能显著提高经验安全质量，停止 D，不训练学生。

### 12.0.1 D0-confirm：保守预算的独立确认

第一轮 D0 不通过后，不修改原结果。新的确认性假设固定为：

$$
\boxed{\text{strict-lexicographic energy},\qquad
D_{\mathrm{KL}}(w\|1/M)=0.2.}
$$

D0-confirm 必须在打开结果前写入新 manifest，并同时更换：

- condition；
- 候选 lineage source；
- mask 随机性（若适用）。

新 condition 必须显式排除第一轮 D0 的 condition，而不只依赖不同随机
seed 降低重合概率。final-test environment 继续完全关闭。D0-confirm
只验证上述单一配置，不再在确认集上选择温度或能量。

### 12.0.2 D1：加权分布摊销准入

仅当 D0-confirm 通过后，才使用与候选 lineage 独立的新 source：

$$
z'_j\sim\mathcal N(0,I),\qquad
y_j^{\mathrm{target}}\sim\sum_iw_i\delta_{y_i},
$$

训练 Path MeanFlow 拟合加权目标边缘分布。D1 必须包含同预算对照：

$$
\begin{aligned}
\mathrm{D1\text{-}U}:&\quad w_i=1/M,\\
\mathrm{D1\text{-}W}:&\quad
w_i\propto\exp[-E(y_i)/\eta],\quad D_{\mathrm{KL}}=0.2.
\end{aligned}
$$

D1-U 与 D1-W 使用完全相同的候选池、独立 source、Stage 1 锚定、参数
更新次数、学习率协议和训练预算。只有 D1-W 在 held-out environment 上
稳定优于 D1-U，改善才可归因于可行性重加权，而不是有限池自蒸馏本身。

D1 分别报告：

- D0 经验池加权 Safe@1；
- 学生部署采样 Safe@1；
- 两者的 amortization gap；
- violation 分位数；
- 模式概率质量；
- Stage 1 完整地图和非阻断任务回退。

定义非参数收益摊销恢复率：

$$
R_{\mathrm{amort}}
=
\frac{
\mathrm{Safe@1}_{\mathrm{D1\text{-}W}}
-
\mathrm{Safe@1}_{\mathrm{Stage1}}
}{
\hat p_{\mathrm{safe}}^{\,w}
-
\mathrm{Safe@1}_{\mathrm{Stage1}}
}.
$$

若分母非正则不报告该比率。还需同时报告 D1-W 相对 D1-U 的增量，不能用
\(R_{\mathrm{amort}}\) 替代自蒸馏对照。

D0 通过但 D1 失败时，结论是 Path MeanFlow 对加权目标分布的摊销失败，
不能归因于非参数重加权原则本身。

### 当前 D1 证据状态

2026-07-30 的固定 100-update D1-U/D1-W 对照在程序上未通过，但给出了
明确的归因证据：

- Stage 1 Safe@1：15.31%；
- D1-U Safe@1：14.69%；
- D1-W Safe@1：16.98%；
- D1-W − D1-U：+2.29 个百分点，environment bootstrap 95% 下界
  +0.63 个百分点；
- D1-W − Stage 1：+1.67 个百分点，但 bootstrap 95% 下界为
  −1.46 个百分点；
- D0 加权目标经验安全质量：22.87%；
- \(R_{\mathrm{amort}}=22.1\%\)，amortization gap 为 5.89 个百分点；
- 主 regime 总模式覆盖保留率为 90.8%，完整地图和非阻断任务回退在预算内。

支持分层进一步显示：在 D0-confirm 的 21 个 has-safe 条件上，
D1-W 相对 D1-U 提高 6.55 个百分点；在 39 个 observed-zero-safe 条件上
二者差值为 0。前者按 environment 聚合的 bootstrap 95% 下界为
+2.03 个百分点。这与 D“只在已有支持内迁移概率质量”的能力边界一致。

因此，D1-W 相对 D1-U 的正增益可以归因于重加权，而不是普通有限池
自蒸馏；但 Path MeanFlow 尚未稳定恢复足够的非参数收益，D1 不能准入正式
Stage 2。当前主要失败属于：

$$
\boxed{\text{加权目标分布的摊销不足，而不是非参数重加权原则失败。}}
$$

### 12.1 主要性能

在 held-out environment 上：

$$
\Delta\mathrm{Safe@1}
=
\mathrm{Safe@1}_{\mathrm{candidate}}
-
\mathrm{Safe@1}_{\mathrm{Stage1}}
>0,
$$

且 environment-level bootstrap 置信区间下界必须为正。

### 12.2 分布保护

以下指标不能显著恶化：

- 有效模式覆盖；
- 条件内路径多样性；
- Oracle Safe@8；
- Stage 1 完整地图和非阻断任务表现；
- 路径长度及基础几何质量。

### 12.3 D 的目标分布质量

固定候选池上必须证明：

- 权重随违反量单调；
- 经验 KL/ESS 满足预设预算；
- 不由极少数条件或单条路径主导；
- 重加权目标的预计安全概率高于均匀经验分布；
- 使用新 source 的学生能够恢复这一改善。

加权训练 loss 下降不能单独作为准入证据。

### 12.4 B+D 的支持质量

加入支持池的局部目标中，大多数必须满足：

$$
V(y^+)<V(y),
$$

且移动受明确上界控制，不频繁发生模式跳转。必须分别报告“原支持内重加权”
和“新增支持”对 Safe@1 改善的贡献。

### 12.4.1 零观测支持复核与 B0

正式把条件交给 B 前，从 observed-zero-safe 条件中冻结子集，增加候选池
规模或使用多批独立 source，分成：

1. 稀有安全支持：扩大采样后出现 strict-safe，D 仍适用；
2. 确认的无安全支持：扩大采样后仍为零；
3. 近安全支持：无 strict-safe，但最低违反量低于冻结阈值；
4. 远离可行域：最低违反量仍高，未必适合有界局部修正。

B0 只针对确认无支持或极弱支持的条件。B0 输出必须区分：

$$
\begin{aligned}
\text{支持创建：}\quad&
y^+\in\mathcal F,\quad d_{\mathrm{path}}(y^+,y)\le\delta;\\
\text{局部进展：}\quad&
V(y^+)<V(y),\quad y^+\notin\mathcal F.
\end{aligned}
$$

只有支持创建可立即作为 D 的可行支持。局部进展不能在单轮实验中记为安全
支持；若要使用，必须另行预注册多轮 B+D 课程。B0 还应按主要失败类型分层：
`forbidden`、`stability`、`curvature` 和 `multiple violations`。

### 12.5 可摊销性

固定加权目标池上，学生必须实际恢复：

- violation 分布下降；
- Safe@1 改善；
- 重加权后的模式占比；
- 使用独立新 source 后的目标分布；
- Stage 1 锚定任务。

### 12.6 判据冻结

在打开 held-out 结果前，必须在统一 manifest 中冻结：

- 候选池大小 \(M\)；
- 违反能量定义；
- KL 预算或 ESS 下界；
- 温度求解器及退化条件处理；
- 局部步长 \(\eta_y\) 与移动上界 \(\delta\)；
- “大多数目标改善”所需的比例阈值；
- 模式跳转和分布恶化的容忍阈值；
- bootstrap 单元、重复次数、随机种子和置信水平；
- early-stopping 与 checkpoint 选择规则。

这些阈值不得根据同一次 held-out 结果事后调整。

## 13. 正式执行顺序

当前不直接启动原 A/B/C：

$$
\boxed{
\begin{aligned}
\text{冻结 Stage 1 与统一 manifest}
&\rightarrow \text{D0-confirm：固定 KL=0.2 的独立确认}\\
&\rightarrow \text{D1-U/D1-W：均匀与加权自蒸馏对照}\\
&\rightarrow \text{observed-zero-safe 条件的大候选池复核}\\
&\rightarrow \text{B0：针对确认无支持/极弱支持条件}\\
&\rightarrow \text{B+D 固定目标池摊销验证}\\
&\rightarrow \text{A/B/D/B+D 同预算小规模比较}\\
&\rightarrow \text{C 的缩小同条件对照}\\
&\rightarrow \text{仅一轮正式 Stage 2}\\
&\rightarrow \text{按 Safe@1 与分布保护决定是否继续}.
\end{aligned}}
$$

在前一步未通过时停止，不自动进入下一步，也不启动多轮 on-policy 数据聚合。

当前执行已停在 D1：D0-confirm 通过，D1-U/D1-W 未通过绝对 Safe@1
稳定改进门槛。因此不启动 B+D；下一轮研究假设应直接针对有限池自蒸馏损失
和低 \(R_{\mathrm{amort}}\) 的摊销机制。observed-zero-safe 的扩大采样与
B0 数据审计可并行准备，但不得混入新的 D1 训练。

B 不作为 D1 失败后的普遍补丁。B 只针对无安全支持或安全支持极弱的条件
扩充局部低违反/可行支持；B+D 的固定顺序是先扩充支持，再用与 D 相同的
保守预算重加权。

B0 的数据审计可以与 D1 准备并行，但 B0 目标不得提前混入 D1-U/D1-W。
若 Path MeanFlow 不能摊销已有支持内的加权分布，加入 B 产生的复杂新目标
不能视为对该问题的修复。

## 14. 当前研究主线

$$
\boxed{\text{Stage 2 不追求 Best-of-}K\text{ 覆盖。}}
$$

$$
\boxed{
\begin{gathered}
\text{近期先验证 D：当前安全/近安全支持能否通过受控重加权被摊销；}\\
\text{随后验证 B+D：局部扩充支持后，能否在分布锚定下迁移概率质量。}
\end{gathered}}
$$

这一表述是当前正式研究路线，不是已经验证的方法结论。

## 15. 方法来源与适用范围

本路线受以下原始工作启发：

- [Relative Entropy Policy Search (REPS), AAAI 2010](https://doi.org/10.1609/aaai.v24i1.7727)：
  用相对熵约束限制策略更新中的信息损失；
- [Maximum a Posteriori Policy Optimisation (MPO), 2018](https://arxiv.org/abs/1806.06920)：
  相对熵目标下的分布改进与参数化拟合；
- [Is Conditional Generative Modeling all you need for Decision-Making?,
  2022](https://arxiv.org/abs/2211.15657)：回报、约束和技能条件生成的后续
  对照依据。

这些工作提供设计动机，不证明本文 Path MeanFlow 重加权在当前几何约束
问题上的有效性。所有结论仍以本协议定义的 held-out Safe@1 和分布保护
实验为准。
