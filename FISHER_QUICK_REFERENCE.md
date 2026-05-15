# Fisher 感知安全梯度微调 - 快速参考卡

## 核心公式一览

### 定义
$$F_j = \mathbb{E}\left[\left(\frac{\partial \ell_{PMF}}{\partial\theta_j}\right)^2\right]$$
Fisher 对角：参数重要性指标

### 更新规则
$$u_j = \frac{u_j^{Adam}}{(F_j + \epsilon)^1}$$
在 Adam update 后应用 Fisher 缩放（$\alpha=1$）

### 结果
- 高 Fisher 参数：少更新（保护第一阶段先验）
- 低 Fisher 参数：多更新（自由优化）

---

## 代码快速参考

### 创建优化器
```python
# 1. 估计 Fisher
fisher_diag = estimate_diag_fisher_from_main_loss(
    model, train_loader, compute_main_loss_fn, 
    device, max_batches=100
)

# 2. 创建 FisherAdamW
optimizer = FisherAdamW(
    named_params=list(model.named_parameters()),
    fisher_diag=fisher_diag,
    lr=1e-4, fisher_alpha=1.0
)
```

### 训练循环
```python
for batch in train_loader:
    optimizer.zero_grad()
    loss = compute_safe_loss(model, batch)
    loss.backward()
    optimizer.step()  # Fisher 感知缩放在此应用
```

---

## 关键参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `fisher_alpha` | 1.0 | 缩放指数 |
| `fisher_eps` | 1e-8 | 数值稳定化 |
| `f_min` | 0.1 | Fisher 下界 |
| `f_max` | 10.0 | Fisher 上界 |
| `max_batches` | 100 | 估计 batch 数 |

---

## 改动 5 处

1. **添加 FisherAdamW 类** (第 37-155 行)
2. **添加 Fisher 估计函数** (第 156-255 行)
3. **更新 Fisher state** (第 512-521 行)：去掉 EMA
4. **更新 diffusion_loss** (第 1200-1260 行)：移除 EMA 和 Fisher penalty
5. **阶段 2 初始化** (第 1993-2060 行)：创建 FisherAdamW

---

## 与原方法区别

| 方面 | 原方法 | 新方法 |
|------|--------|--------|
| Fisher 作用位置 | loss 正则项 | 优化器梯度 |
| EMA 平滑 | 有（$\beta=0.95$） | 无 |
| 超参数 | $\lambda$ | 无 |
| 第一步效果 | 无 | 有 |

---

## 数学核心

**最优化问题**：
$$\min_s \frac{1}{2}s^\top Fs \quad \text{s.t.} \quad g^\top s = -\delta$$

**闭式解**：
$$s^\star \propto -F^{-1}g$$

**实现**（对角+$\alpha=1$）：
$$s_j = -\eta \frac{g_j}{F_j + \epsilon}$$

---

## 监控指标

```
Fisher Diag Mean: E[F_j]     # 应在 0.1~10
Param Drift Mean: E[Δθ²]     # 应缓慢增长
```

---

## 文件位置

| 文件 | 说明 |
|------|------|
| `train_fisher.py` | 主实现 |
| `FISHER_AWARE_GRADIENT_UPDATE.md` | 完整理论 |
| `FISHER_ADAMW_PYTORCH_GUIDE.md` | PyTorch 指南 |

---

## 一句话总结

**用第一阶段 PMF 损失梯度的平方估计参数重要性，在第二阶段 safe_loss 优化时，梯度以此重要性反比缩放，保护高重要参数，防止破坏第一阶段先验。**

---

## 常见问题

**Q: 为什么 alpha=1？**
A: 对应完整的 Fisher 逆预条件，理论上最优。

**Q: 为什么无 EMA？**
A: 简化实现，Fisher 直接来自梯度，更透明，无需调 ema_beta。

**Q: 是否改变目标函数？**
A: 不改变，只改变优化几何（有效 Hessian）。

**Q: 收敛点会变吗？**
A: 理论上不变（都是 safe_loss 的驻点），但实际可能不同（非凸优化）。

---

## 推荐用法

```python
# 第一阶段：标准训练
optimizer = Adam(model.parameters())
train(model, optimizer, main_loss)  # → 学到 θ^(0)

# 第二阶段切换
fisher = estimate_from(train_data, main_loss)
optimizer = FisherAdamW(model, fisher_diag=fisher)
train(model, optimizer, safe_loss)  # → 学到 θ_final
```

保证：
✅ 保护第一阶段先验
✅ 有效优化 safe_loss
✅ 无额外超参数调优
✅ 理论上有保证
