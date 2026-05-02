明白。现在我们把问题重新整理为：

[
\boxed{
\text{用可微 cost 函数定义一个新的目标分布，并把 pMF 迁移到这个新分布。}
}
]

你的 Score 和 Cost 互为相反数：

[
S(x,c)=-C(x,c)
]

所以不再写 Score，而统一用 cost：

[
C(x,c)
]

其中 (x) 是网络输出的轨迹，例如 B-spline 控制点、路径点序列或轨迹参数；(c) 是条件输入，例如地形图、起点、终点、机器人参数等。

---

# 1. 目标分布应该怎么定义？

如果直接最小化 cost：

[
\min_\theta C(X_\theta(\epsilon,c),c)
]

那么最优结果就是所有噪声 (\epsilon) 都映射到同一个最低 cost 轨迹：

[
X_\theta(\epsilon,c)\rightarrow x^*(c)
]

其中：

[
x^*(c)=\arg\min_x C(x,c)
]

于是分布坍塌为：

[
p_\theta(x\mid c)\rightarrow \delta(x-x^*)
]

所以我们不能只定义“最小 cost”，而要定义一个**由 cost 决定的新分布**：

[
\boxed{
q_\beta(x\mid c)
================

\frac{1}{Z_\beta(c)}
\exp\left[-\beta C(x,c)\right]
}
]

其中：

* (q_\beta(x\mid c))：第二阶段希望迁移到的新分布；
* (C(x,c))：你的 cost；
* (\beta>0)：inverse temperature，逆温度；
* (Z_\beta(c))：归一化常数，不需要显式计算。

也可以写成温度形式：

[
q_T(x\mid c)
============

\frac{1}{Z_T(c)}
\exp\left[-\frac{C(x,c)}{T}\right]
]

两者关系是：

[
\beta=\frac{1}{T}
]

---

# 2. 为什么这个分布不会天然坍塌？

只要：

[
T>0
]

或者：

[
\beta<\infty
]

目标分布就不是单点，而是一个有宽度的 Boltzmann 分布。

低 cost 轨迹概率更大：

[
C(x_1)<C(x_2)
\Rightarrow
q_\beta(x_1)>q_\beta(x_2)
]

但不是只有最低 cost 轨迹才有概率。

当：

[
\beta\rightarrow\infty
]

也就是：

[
T\rightarrow 0
]

目标分布才会退化成狄拉克分布：

[
q_\beta(x\mid c)\rightarrow \delta(x-x^*)
]

所以防止坍塌的核心不是第一阶段损失，而是：

[
\boxed{
\beta \text{ 不能过大，或者 } T \text{ 不能过小。}
}
]

---

# 3. Cost 未归一化时怎么办？

你的 cost 未归一化没有问题。Langevin 方法本身不需要知道：

[
Z_\beta(c)
]

也不要求 cost 是归一化概率。

但是有一个关键点：

[
\boxed{
\text{cost 的数值尺度会直接影响 } \beta \text{ 的有效强度。}
}
]

因为目标分布是：

[
q_\beta(x\mid c)\propto \exp[-\beta C(x,c)]
]

如果你的 cost 数值很大，比如：

[
C\sim 1000
]

那么即使：

[
\beta=1
]

也会让：

[
\exp[-\beta C]
]

极端尖锐，容易近似坍塌。

如果你的 cost 数值很小，比如：

[
C\sim 0.01
]

那么：

[
\beta=1
]

几乎没有选择压力，分布迁移很弱。

所以不归一化 cost 时，(\beta) 就承担了“尺度校准”的作用。

---

# 4. Langevin 分布迁移的核心公式

目标分布：

[
q_\beta(x\mid c)\propto \exp[-\beta C(x,c)]
]

对应的 overdamped Langevin dynamics 为：

[
dx_s
====

-\beta \nabla_x C(x_s,c),ds
+
\sqrt{2},dW_s
]

离散化后得到：

[
\boxed{
x^{m+1}
=======

## x^m

\eta\beta\nabla_x C(x^m,c)
+
\sqrt{2\eta}\xi^m
}
]

其中：

* (m)：Langevin 内循环步数；
* (\eta)：Langevin 步长；
* (\xi^m\sim\mathcal{N}(0,I))：高斯噪声；
* (-\nabla_x C)：把样本推向低 cost 区域；
* (\sqrt{2\eta}\xi)：保持分布扩散，防止所有样本塌到同一点；
* (\beta)：控制 cost 引导强度。

等价地，用温度 (T) 写：

[
\boxed{
x^{m+1}
=======

## x^m

\eta\nabla_x C(x^m,c)
+
\sqrt{2T\eta}\xi^m
}
]

这两个版本等价。
如果你的 cost 未归一化，我更建议用 (\beta) 版本：

[
x^{m+1}
=======

## x^m

\eta\beta\nabla_x C(x^m,c)
+
\sqrt{2\eta}\xi^m
]

因为 (\beta) 可以直观控制 cost 的实际作用强度。

---

# 5. 你的第二阶段训练流程

## Step 1：从当前 pMF 生成粒子

用当前网络采样一批轨迹：

[
\epsilon_i\sim\mathcal{N}(0,I)
]

[
x_i^0=X_\theta(\epsilon_i,c)
]

其中：

[
i=1,\ldots,K
]

(K) 是同一个条件 (c) 下的粒子数量。建议：

[
K=16\sim 128
]

---

## Step 2：在轨迹空间做 Langevin 迁移

对每个粒子 (x_i^0)，做 (M) 步 Langevin：

[
x_i^{m+1}
=========

## x_i^m

\eta\beta\nabla_x C(x_i^m,c)
+
\sqrt{2\eta}\xi_i^m
]

得到迁移后的粒子：

[
\tilde x_i=x_i^M
]

这一步的含义是：

[
p_\theta(x\mid c)
\quad
\longrightarrow
\quad
q_\beta(x\mid c)\propto \exp[-\beta C(x,c)]
]

注意：

[
\boxed{
\text{这一步只更新粒子 } x_i,\text{ 不直接更新网络参数 } \theta。
}
]

---

## Step 3：用迁移后的粒子重新训练 pMF

把：

[
\tilde x_i
]

当作新的轨迹样本，用原来的 pMF 训练方式训练网络。

例如按你的实现，如果网络输出就是终点轨迹 (x)，那么第二阶段的 pMF 训练数据就是：

[
\tilde x_i
]

构造中间状态：

[
z_t=(1-t)\tilde x_i+t\epsilon
]

然后计算你的原始 pMF 损失：

[
\boxed{
\mathcal{L}_{stage2}
====================

\mathbb{E}*{\tilde x_i,c,r,t,\epsilon}
\left[
\ell*{\mathrm{pMF}}(\theta;\tilde x_i,c,r,t,\epsilon)
\right]
}
]

这里没有：

[
L_{\mathrm{score}}
]

也没有：

[
L_{\mathrm{cost}}
]

直接作用到网络输出上。

Cost 只用于 Langevin 粒子迁移。

---

# 6. 为什么不要让 cost 直接反传到网络？

不要做：

[
\mathcal{L}
===========

C(X_\theta(\epsilon,c),c)
]

因为这会让所有 (\epsilon) 都追向同一个低 cost 点。

正确做法是：

[
X_\theta(\epsilon,c)
\rightarrow x_i^0
]

[
x_i^0
\overset{\text{Langevin}}{\longrightarrow}
\tilde x_i
]

[
X_\theta \text{ 学习 } \tilde x_i \text{ 的分布}
]

也就是说：

[
\boxed{
\text{cost 负责迁移样本分布，pMF 负责拟合迁移后的分布。}
}
]

训练时应当：

[
\tilde x_i=\operatorname{stopgrad}(\tilde x_i)
]

不要把 Langevin 过程反传回 (\theta)。

---

# 7. 完整算法写法

对每个训练 iteration：

1. 采样条件 (c)；
2. 采样噪声 (\epsilon_i)；
3. 当前模型生成：

[
x_i^0=X_\theta(\epsilon_i,c)
]

4. 对每个 (x_i^0) 执行 (M) 步 Langevin：

[
x_i^{m+1}
=========

## x_i^m

\eta\beta\nabla_x C(x_i^m,c)
+
\sqrt{2\eta}\xi_i^m
]

5. 得到：

[
\tilde x_i=x_i^M
]

6. 停止梯度：

[
\tilde x_i\leftarrow \operatorname{sg}(\tilde x_i)
]

7. 用 (\tilde x_i) 计算 pMF 训练损失：

[
\mathcal{L}_{stage2}
====================

\mathbb{E}
[
\ell_{\mathrm{pMF}}(\theta;\tilde x_i,c)
]
]

8. 更新网络参数：

[
\theta\leftarrow\theta-\gamma\nabla_\theta \mathcal{L}_{stage2}
]

---

# 8. (\beta)、(\eta)、(M) 怎么选？

因为你的 cost 未归一化，最重要的是调：

[
\beta
]

而不是盲目设：

[
\beta=1
]

---

## 8.1 (\beta)：控制目标分布尖锐程度

目标分布：

[
q_\beta(x\mid c)\propto \exp[-\beta C(x,c)]
]

如果 (\beta) 太小：

[
q_\beta \text{ 很平}
]

迁移弱，cost 下降慢。

如果 (\beta) 太大：

[
q_\beta \text{ 很尖}
]

容易接近最低 cost 点，出现坍塌趋势。

建议从小 (\beta) 开始，例如：

[
\beta=10^{-3},10^{-2},10^{-1}
]

具体取决于你的 cost 数值量级。

更稳的选择是根据初始粒子 cost 差来定：

[
\Delta C
========

\operatorname{P90}(C_i)-\operatorname{P10}(C_i)
]

让：

[
\beta \Delta C
\approx 1\sim 5
]

也就是：

[
\beta
\approx
\frac{1\sim 5}{\Delta C}
]

这个不是强行归一化 cost，而是让 Boltzmann 分布不要过平，也不要过尖。

---

## 8.2 (\eta)：控制每一步粒子移动幅度

Langevin 更新：

[
\Delta x
========

-\eta\beta\nabla C
+
\sqrt{2\eta}\xi
]

其中确定性位移大小约为：

[
|\eta\beta\nabla C|
]

噪声位移每维标准差是：

[
\sqrt{2\eta}
]

建议让每一步位移不要太大：

[
\frac{
\operatorname{median}\left(|\eta\beta\nabla C|\right)
}{
\operatorname{median}\left(|x_i-x_j|\right)
}
\approx
0.01\sim 0.1
]

如果一步就把轨迹推得很远，说明：

[
\eta\beta
]

过大。

---

## 8.3 (M)：Langevin 内循环步数

(M) 太小：

[
\tilde x_i
]

迁移不够。

(M) 太大：

粒子可能过度集中在低 cost 模式附近。

建议初始：

[
M=5\sim 20
]

后续可以逐渐增加到：

[
M=20\sim 50
]

但不建议一开始就很大。

---

# 9. 建议使用退火策略

第二阶段不要直接用很强的 (\beta)。

推荐：

[
\beta_1<\beta_2<\cdots<\beta_{\max}
]

也就是逐渐降低温度：

[
T_1>T_2>\cdots>T_{\min}
]

例如：

[
\beta: 0.001\rightarrow 0.003\rightarrow 0.01\rightarrow 0.03
]

或者根据你的 cost 尺度：

[
\beta\Delta C: 1\rightarrow 2\rightarrow 3\rightarrow 5
]

这样模型会从原分布平滑迁移到更低 cost 的新分布，而不是突然被拉到某个单点模式。

---

# 10. 如果轨迹有固定起点和终点

如果 (x) 是轨迹控制点：

[
x=[p_0,p_1,\ldots,p_N]
]

其中起点和终点固定：

[
p_0=p_{\mathrm{start}}
]

[
p_N=p_{\mathrm{goal}}
]

那么 Langevin 更新不能改变它们。

可以定义 mask：

[
M_x
]

对固定维度置零，对可优化维度置一。

更新改为：

[
x^{m+1}
=======

## x^m

\eta\beta M_x\nabla_x C(x^m,c)
+
\sqrt{2\eta}M_x\xi^m
]

然后每一步后强制投影：

[
p_0\leftarrow p_{\mathrm{start}}
]

[
p_N\leftarrow p_{\mathrm{goal}}
]

---

# 11. 如果轨迹变量不同维度尺度不同

例如：

* (x,y) 坐标尺度是米；
* yaw 是弧度；
* B-spline 控制点不同维度敏感性不同。

可以使用预条件 Langevin：

[
\boxed{
x^{m+1}
=======

## x^m

\eta\beta P\nabla_x C(x^m,c)
+
\sqrt{2\eta P}\xi^m
}
]

其中 (P) 是正定矩阵，常用对角矩阵即可。

例如：

[
P=\operatorname{diag}(s_1^2,s_2^2,\ldots,s_d^2)
]

这样不同维度的更新尺度更合理。

如果用 mask 和 preconditioner，则：

[
x^{m+1}
=======

## x^m

\eta\beta M_xP\nabla_x C
+
\sqrt{2\eta M_xP}\xi
]

---

# 12. 如果 cost 有硬约束或可行域

如果轨迹必须满足某些边界，例如速度限制、控制点范围、地图边界，可以在每一步 Langevin 后做投影：

[
x^{m+1}
\leftarrow
\Pi_{\mathcal{X}}(x^{m+1})
]

其中：

[
\mathcal{X}
]

是可行轨迹参数空间。

完整更新：

[
x^{m+1}
=======

\Pi_{\mathcal{X}}
\left[
x^m
---

\eta\beta\nabla_x C(x^m,c)
+
\sqrt{2\eta}\xi^m
\right]
]

不过要注意：投影会改变严格的目标分布，但工程上通常是必要的。

---

# 13. 推荐加入梯度裁剪

因为 cost 未归一化，(\nabla C) 可能很大。

建议对 cost 梯度做裁剪：

[
g=\nabla_x C(x,c)
]

[
g\leftarrow
g\cdot
\min
\left(
1,
\frac{g_{\max}}{|g|}
\right)
]

然后：

[
x^{m+1}
=======

## x^m

\eta\beta g
+
\sqrt{2\eta}\xi
]

这能防止某些粒子被一次性推飞。

---

# 14. 最终推荐的第二阶段损失

第二阶段的网络训练损失只有：

[
\boxed{
\mathcal{L}_{stage2}
====================

\mathbb{E}*{\tilde x,c,r,t,\epsilon}
\left[
\ell*{\mathrm{pMF}}(\theta;\tilde x,c,r,t,\epsilon)
\right]
}
]

其中：

[
\tilde x
]

来自：

[
x^0=X_\theta(\epsilon,c)
]

[
x^{m+1}
=======

## x^m

\eta\beta\nabla_x C(x^m,c)
+
\sqrt{2\eta}\xi^m
]

[
\tilde x=x^M
]

不要再写成：

[
\lambda L_{\mathrm{pmf-old}}+C(X_\theta)
]

也不要写成：

[
L_{\mathrm{pmf}}+L_{\mathrm{cost}}
]

因为那会重新回到“旧分布约束”和“单点最优”之间的拉扯。

---

# 15. 一版简洁伪代码

```python
# theta: 当前 pMF 模型
# C(x, c): 可微 cost
# beta: inverse temperature
# eta: Langevin step size
# M: Langevin steps
# K: particles per condition

for batch in loader:
    c = batch["condition"]

    # 1. 当前模型采样
    eps0 = torch.randn(K, *x_shape)
    with torch.no_grad():
        x0 = model.sample(eps0, c)   # x0 = X_theta(eps0, c)

    # 2. Langevin 粒子迁移
    x = x0.detach().clone().requires_grad_(True)

    for m in range(M):
        cost = C(x, c).sum()
        grad = torch.autograd.grad(cost, x)[0]

        grad = clip_grad_norm_like(grad, max_norm=g_max)

        noise = torch.randn_like(x)

        with torch.no_grad():
            x = x - eta * beta * grad + (2 * eta) ** 0.5 * noise

            # 可选：固定起点终点、投影到可行域
            x = project_constraints(x, c)

        x.requires_grad_(True)

    x_tilde = x.detach()

    # 3. 用迁移后的轨迹作为新数据训练 pMF
    loss = pmf_loss(model, x_tilde, c)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

关键点是：

[
x_{\text{tilde}}
]

必须 detach。

---

# 16. 判断是否坍塌的指标

训练时不要只看 cost 是否下降，还要看分布是否保留宽度。

建议监控：

### 1. 平均 cost

[
\mathbb{E}[C(x,c)]
]

应该下降。

### 2. Top-k cost

[
\operatorname{mean}(\operatorname{TopKLow}(C))
]

应该下降。

### 3. 样本两两距离

[
D_{\mathrm{pair}}
=================

\frac{1}{K(K-1)}
\sum_{i\neq j}
|x_i-x_j|
]

不能快速接近 0。

### 4. 协方差迹

[
\operatorname{Tr}(\operatorname{Cov}(x_1,\ldots,x_K))
]

不能塌到接近 0。

如果 cost 下降但多样性接近 0，说明：

[
\beta \text{ 太大}
]

或者：

[
M \text{ 太大}
]

或者：

[
\eta\beta \text{ 太大}
]

---

# 17. 最终总结

你现在应该把第二阶段理解成：

[
\boxed{
p_{\theta_0}(x\mid c)
\longrightarrow
q_\beta(x\mid c)
\propto
\exp[-\beta C(x,c)]
}
]

其中第一阶段模型只是初始化，不是约束目标。

Langevin 方法做的事情是：

[
x^{m+1}
=======

## x^m

\eta\beta\nabla_x C(x^m,c)
+
\sqrt{2\eta}\xi^m
]

它用：

[
-\nabla C
]

降低 cost，用：

[
\sqrt{2\eta}\xi
]

保持分布扩散，避免狄拉克坍塌。

然后 pMF 只负责学习迁移后的样本：

[
\mathcal{L}_{stage2}
====================

\mathbb{E}
[
\ell_{\mathrm{pMF}}(\theta;\tilde x,c)
]
]

一句话概括：

> **不要让 cost 直接训练网络；让 cost 通过 Langevin 把样本迁移到 (q_\beta(x\mid c)\propto \exp[-\beta C(x,c)])，再让 pMF 蒸馏这个新分布。**
