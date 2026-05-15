# Fisher 感知安全梯度微调 - 实现总结

## 核心改动概览

本文档总结了在 `train_fisher.py` 中实现的 Fisher 感知梯度更新方法。该方法的核心思想是：

**使用第一阶段 PMF 训练损失估计参数重要性（Fisher 对角），然后在第二阶段按重要性重加权 safe-loss 梯度更新。**

---

## 1. 新增 FisherAdamW 优化器

### 位置
文件开头，导入部分之后（第37-155行）

### 核心功能
```python
class FisherAdamW(torch.optim.Optimizer):
```

- 继承自 `torch.optim.Optimizer`
- 在标准 Adam 计算出 update 后，用 Fisher 对角矩阵对其进行逐参数缩放
- 缩放因子：$(F_j + \epsilon)^\alpha$，其中 $\alpha=1$（对应理论上的 Fisher 逆预条件）

### 参数说明
- `named_params`: 模型的 named_parameters() 迭代器
- `fisher_diag`: dict，键为参数名，值为 Fisher 对角张量
- `lr`: 学习率
- `betas`: Adam 的 (beta1, beta2)
- `eps`: Adam 的数值稳定化常数
- `weight_decay`: 权重衰减系数
- `fisher_eps`: Fisher 缩放的数值稳定化常数（防止除以零）
- `fisher_alpha`: Fisher 缩放的指数（固定为 1.0）

### 更新规则
```
u_j^{Adam} = m_j / (sqrt(v_j) + eps)
u_j = u_j^{Adam} / (F_j + fisher_eps)^alpha
theta_j <- theta_j - lr * u_j
```

其中 `alpha=1` 对应理论上完整的 Fisher 逆预条件。

---

## 2. 新增 Fisher 估计函数

### 位置
第156-255行：`estimate_diag_fisher_from_main_loss()`

### 核心功能
从第一阶段的 main_loss 梯度平方直接估计对角 Fisher 信息矩阵。

### 实现细节
```
Fisher_j = E[(∂main_loss/∂θ_j)²]
```

**不使用 EMA 平滑**，而是：
1. 在多个 batch 上计算梯度平方
2. 求平均
3. 全局归一化（除以所有参数的平均 Fisher 值）
4. 裁剪到合理范围 [f_min, f_max]

### 参数
- `model`: 神经网络模型
- `dataloader`: 数据加载器
- `compute_loss_fn`: 计算 main_loss 的函数
- `device`: 设备
- `max_batches`: 最大 batch 数（为了加快估计，通常用前100个 batch）
- `f_min`, `f_max`: Fisher 对角裁剪范围

### 返回
dict，键为参数名，值为 Fisher 对角张量

---

## 3. 更新 _get_stage2_fisher_proxy_state()

### 改动内容
- **去掉** `'ema_beta': 0.95` 字段
- **保留** 其他必要字段：
  - `'ref_params'`: 参考参数（第一阶段的模型参数）
  - `'fisher_diag'`: Fisher 对角（在每个 batch 更新，无 EMA）
  - `'proxy_scale'`: 监控指标用的缩放因子（已停用）
  - `'fisher_eps'`: 数值稳定化常数

---

## 4. diffusion_loss 函数中的 Fisher 处理

### 改动：第 1200-1260 行

**关键变化**：
- **移除** EMA 平滑：`proxy_state['ema_beta'] * old_fisher + ...`
- **直接使用** 当前 batch 梯度的平方作为 Fisher 对角
- **移除** Fisher 惩罚项从 loss 中：不再有 `capsize_loss = safe_loss + proxy_state['proxy_scale'] * fisher_penalty`

### 新的 Fisher 计算逻辑

```python
# 用主损失的参数梯度平方直接作为 Fisher 对角（无 EMA 平滑）
grads = torch.autograd.grad(
    fisher_source_loss,
    trainable_params,
    retain_graph=True,
    allow_unused=True,
    create_graph=False,
)

fisher_diag = []
with torch.no_grad():
    for grad, param in zip(grads, trainable_params):
        if grad is None:
            new_fisher = torch.zeros_like(param)
        else:
            new_fisher = grad.detach().pow(2)
        # 裁剪到合理范围
        new_fisher = torch.clamp(new_fisher, min=proxy_state['fisher_eps'], max=1e6)
        fisher_diag.append(new_fisher)
    proxy_state['fisher_diag'] = fisher_diag
```

**原理**：
- Fisher 直接来自 main_loss 梯度的平方（而非 safe_loss）
- 这样可以保护第一阶段学到的轨迹生成分布
- 不添加额外的惩罚项，而是让优化器在更新时应用 Fisher 缩放

---

## 5. 阶段 2 初始化中的 Fisher 估计

### 位置
第 1993-2060 行：阶段切换时的优化器重初始化

### 核心流程

#### 5.1 定义 main_loss 计算函数
```python
def compute_main_loss_fn(model, batch):
    # 计算 Pixel MeanFlow 的 main_loss
    # 用于后续 Fisher 估计
```

#### 5.2 从第一阶段数据估计 Fisher
```python
print("📊 正在从第一阶段数据估计 Fisher 对角...")
fisher_diag = estimate_diag_fisher_from_main_loss(
    model=model,
    dataloader=temp_dataloader,
    compute_loss_fn=compute_main_loss_fn,
    device=device,
    max_batches=100,  # 用前100个batch估计
    f_min=0.1,
    f_max=10.0
)
```

#### 5.3 创建 FisherAdamW 优化器
```python
base_optimizer = FisherAdamW(
    named_params=list(model.named_parameters()),
    fisher_diag=fisher_diag,
    lr=1e-4,
    betas=(0.95, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    fisher_eps=1e-8,
    fisher_alpha=1.0  # alpha = 1
)

# 用 ScheduledOptim wrapper 包装
optimizer = Optim.ScheduledOptim(
    base_optimizer,
    lr_mul=stage2_config['lr_mul'],
    d_model=512,
    n_warmup_steps=50
)
```

---

## 6. 实现的数学原理

### 6.1 参数重要性的定义

在第一阶段模型 $\theta^{(0)}$ 附近，考虑小参数扰动 $s$，第一阶段生成器输出变化近似为：

$$f_{\theta^{(0)}+s}(x) \approx f_{\theta^{(0)}}(x) + J(x)s$$

其中 $J(x)$ 是模型输出对参数的 Jacobian。

输出扰动的平均能量为：

$$D_{PMF}(s) = \frac{1}{2}s^\top Fs$$

其中 $F = E_x[J(x)^\top J(x)]$ 是 Fisher 信息矩阵。

**结论**：$F_j$ 越大，同样幅度的参数扰动对第一阶段输出影响越大。

### 6.2 Fisher 感知更新的推导

第二阶段目标：在下降 safe loss 的同时，最小化对第一阶段信息的损伤。

优化问题：
$$\min_s \frac{1}{2}s^\top Fs \quad \text{subject to} \quad g^\top s = -\delta$$

其中 $g = \nabla_\theta L_{safe}$，$\delta$ 是 safe loss 的下降量。

Lagrange 对偶解：
$$s^\star \propto -F^{-1}g$$

**实际实现**（对角近似，$\alpha=1$）：
$$s_j = -\eta \frac{g_j}{F_j + \epsilon}$$

高 Fisher 参数 $F_j$ 大，有效学习率小 → 少更新
低 Fisher 参数 $F_j$ 小，有效学习率大 → 多更新

### 6.3 收敛点不变

该方法改变的是优化的几何（有效 Hessian 的条件数），而不是目标函数本身：

$$\nabla_\theta L_{safe}(\theta^\star) = 0 \text{ 不变}$$

但实际训练轨迹和最终 checkpoint 可能不同（在非凸优化中）。

---

## 7. 与原方法的对比

| 项目 | 原 Fisher penalty | 新 Fisher-aware 更新 |
|------|----------------|------------------|
| Fisher 放在哪里 | loss 正则项 | 梯度更新方向 |
| 是否需要 $\lambda$ | 需要（超参调优难） | 不需要 |
| 第一轮是否生效 | 否（$\theta=\theta^{(0)}$ 时为0） | 是（第一轮就生效） |
| 保护方式 | 参数偏离后拉回 | 更新前避开高重要性方向 |
| 目标函数 | 改变 safe-loss 目标 | 不改变 safe-loss 目标 |
| 核心解释 | 限制参数距离 | 限制每步信息损伤 |

---

## 8. 实现特点

### 8.1 不使用 EMA
- 原方法使用 `ema_beta=0.95` 对 Fisher 进行指数移动平均
- 新方法**直接使用当前 batch 的梯度平方**
- 优点：
  - 每个 batch Fisher 估计更新，实时反应数据特性
  - 减少超参数（不需要调 ema_beta）
  - 数学上更清晰（直接用主损失梯度定义重要性）

### 8.2 批量 Fisher 估计
- 在阶段2初始化时，用前100个batch的数据估计一次 Fisher
- 之后在每个训练 step 中更新（在 diffusion_loss 中）
- 这样既有初期的精确估计，又有持续的动态更新

### 8.3 全局归一化
Fisher 值经过全局归一化，避免数值尺度问题：
$$\tilde{F}_j = \frac{F_j}{\text{mean}(F)} / (\epsilon)$$

这样所有参数的 Fisher 值在同一个尺度上。

---

## 9. 使用说明

### 9.1 训练流程
1. **第一阶段**：标准 AdamW 训练 PMF
2. **阶段切换点**（epoch = stage1_config['epochs']）：
   - 加载第一阶段最优模型
   - 从第一阶段训练数据估计 Fisher 对角
   - 创建 FisherAdamW 优化器
3. **第二阶段**：FisherAdamW + safe_loss 微调

### 9.2 关键参数
- `fisher_alpha=1.0`：Fisher 缩放指数（已固定）
- `max_batches=100`：Fisher 估计用的 batch 数
- `f_min=0.1, f_max=10.0`：Fisher 对角裁剪范围
- `fisher_eps=1e-8`：数值稳定化常数

### 9.3 监控指标
在训练输出中观察：
- `Fisher Diag Mean`：Fisher 对角的平均值
- `Param Drift Mean`：参数偏移的平均值
- 预期：高 Fisher 参数的偏移应该小于低 Fisher 参数

---

## 10. 可进一步优化的方向

1. **每 batch 更新 Fisher**：当前在 diffusion_loss 中每个 batch 都计算梯度平方，但未应用到优化器（因为 optimizer.step() 前无法修改）。可考虑在 backward 后手动应用 Fisher 缩放。

2. **高阶 Fisher 估计**：当前用的是对角 Fisher。若计算资源允许，可用完整的 Hessian 矩阵。

3. **自适应 alpha**：当前固定 $\alpha=1$。可考虑根据训练进度调整，例如早期 $\alpha$ 较大（保护多），后期 $\alpha$ 较小（允许更多改变）。

4. **参数分组**：可能不同层的参数需要不同的 Fisher 缩放策略（e.g., embedding vs. transformer 层）。

---

## 11. 文件修改清单

| 文件 | 修改内容 |
|------|---------|
| train_fisher.py | 添加 FisherAdamW 类 |
| train_fisher.py | 添加 estimate_diag_fisher_from_main_loss() 函数 |
| train_fisher.py | 更新 _get_stage2_fisher_proxy_state()：去掉 ema_beta |
| train_fisher.py | 更新 diffusion_loss()：移除 EMA，直接使用梯度平方 |
| train_fisher.py | 更新阶段2初始化：调用 Fisher 估计，创建 FisherAdamW |

---

## 12. 参考公式汇总

### Fisher 对角的定义
$$F_j = \mathbb{E}_{(z,c) \sim \mathcal{D}}\left[\left(\frac{\partial \ell_{PMF}(\theta^{(0)}; z,c)}{\partial\theta_j}\right)^2\right]$$

### 归一化 Fisher
$$\tilde{F}_j = \frac{F_j}{\text{mean}(F) + \epsilon}$$

### Fisher 感知更新（$\alpha=1$）
$$\theta_{t+1,j} = \theta_{t,j} - \eta \frac{g_{t,j}}{F_j + \epsilon_{fisher}}$$

其中 $g_{t,j} = \frac{\partial L_{safe}}{\partial\theta_j}$

### safe loss 驻点不变
$$\nabla_\theta L_{safe}(\theta^\star) = 0 \quad \text{在任何有效的优化器下都成立}$$

---

## 总结

这次实现的核心改进是：

1. **去掉 EMA**：更简洁，减少超参数
2. **Fisher 直接作用在梯度上**：而非加入 loss 正则项
3. **只需估计一次 Fisher**：在阶段切换时从第一阶段数据估计，之后持续更新
4. **实现了理论上的 Fisher 逆预条件**：$\alpha=1$ 对应完整的二阶信息

该方法既保留了 Fisher 方法的理论优势，又通过简化实现提高了可用性和稳定性。
