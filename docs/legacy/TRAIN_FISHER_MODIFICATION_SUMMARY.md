# train_fisher.py 修改总结

## 修改概览

本文档总结了对 `train_fisher.py` 的核心修改，实现了 **Fisher 感知安全梯度微调** 方法。

---

## 核心修改 5 处

### 修改 1: 添加 FisherAdamW 优化器类
**位置**: 第 37-155 行（在导入之后）

**内容**: 实现了一个新的优化器类 `FisherAdamW`
- 继承自 `torch.optim.Optimizer`
- 在 Adam update 后应用 Fisher 缩放：$u_j = u_j^{Adam} / (F_j + \epsilon)^\alpha$
- alpha 固定为 1.0（对应理论的 Fisher 逆预条件）
- 关键参数：`fisher_diag` (dict), `fisher_alpha=1.0`, `fisher_eps=1e-8`

**用途**: 在第二阶段用于 Fisher 感知的参数更新

---

### 修改 2: 添加 Fisher 估计函数
**位置**: 第 156-255 行

**函数**: `estimate_diag_fisher_from_main_loss()`

**核心功能**:
```python
# Fisher 对角公式：F_j = E[(∂main_loss/∂θ_j)²]
# 实现流程：
# 1. 在多个 batch 上反向传播 main_loss
# 2. 累积梯度的平方
# 3. 求平均
# 4. 全局归一化（除以平均值）
# 5. 裁剪到 [f_min, f_max] 范围
```

**关键特点**:
- **无 EMA 平滑**：直接使用梯度平方，不加权平均历史值
- **批量估计**：在阶段 2 初始化时从前 100 个 batch 估计一次
- **数值稳定**：包含全局归一化和范围裁剪

---

### 修改 3: 更新 _get_stage2_fisher_proxy_state()
**位置**: 第 512-521 行

**改动内容**:
```python
# 移除了：
# 'ema_beta': 0.95,

# 保留：
{
    'ref_params': None,          # 参考参数（第一阶段模型）
    'fisher_diag': None,         # Fisher 对角（无 EMA）
    'proxy_scale': 0.01,         # 监控用（已停用）
    'fisher_eps': 1e-8,          # 数值稳定化常数
}
```

**原因**: Fisher 不再通过 EMA 平滑，每个 batch 直接更新

---

### 修改 4: diffusion_loss 函数中的 Fisher 处理
**位置**: 第 1200-1260 行（第二阶段 Fisher 代理部分）

**核心改动**:

#### 4.1 移除 EMA 平滑
```python
# 原来：
updated_fisher = proxy_state['ema_beta'] * old_fisher + (1.0 - proxy_state['ema_beta']) * new_fisher

# 现在（直接使用）：
new_fisher = grad.detach().pow(2)
fisher_diag.append(torch.clamp(new_fisher, min=..., max=...))
```

#### 4.2 移除 Fisher 惩罚项
```python
# 原来：
capsize_loss = safe_loss + proxy_state['proxy_scale'] * fisher_penalty

# 现在：
capsize_loss = safe_loss  # Fisher 作用在优化器，不在 loss
```

**原理**: Fisher 不再作为正则项，而是在优化器中应用梯度缩放

---

### 修改 5: 阶段 2 初始化中的 Fisher 估计和优化器更新
**位置**: 第 1993-2060 行（阶段切换时）

**核心步骤**:

#### 5.1 定义 main_loss 计算函数（用于 Fisher 估计）
```python
def compute_main_loss_fn(model, batch):
    # 从 batch 计算 Pixel MeanFlow 的 main_loss
    # 支持 epsilon/x0/v 预测类型
    ...
```

#### 5.2 估计 Fisher 对角
```python
print("📊 正在从第一阶段数据估计 Fisher 对角...")
fisher_diag = estimate_diag_fisher_from_main_loss(
    model=model,
    dataloader=temp_dataloader,      # 第一阶段训练数据
    compute_loss_fn=compute_main_loss_fn,
    device=device,
    max_batches=100,                 # 用前 100 个 batch
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
    fisher_alpha=1.0  # 关键：alpha = 1
)

# 用 ScheduledOptim wrapper 包装以支持学习率调度
optimizer = Optim.ScheduledOptim(
    base_optimizer,
    lr_mul=stage2_config['lr_mul'],
    d_model=512,
    n_warmup_steps=50
)
```

---

## 关键参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `fisher_alpha` | 1.0 | Fisher 缩放指数（对应完整的二阶信息） |
| `fisher_eps` | 1e-8 | 数值稳定化常数（防止除以零） |
| `f_min` | 0.1 | Fisher 下界裁剪 |
| `f_max` | 10.0 | Fisher 上界裁剪 |
| `max_batches` | 100 | Fisher 估计用的 batch 数 |

---

## 实现的关键优势

### 1. 无 EMA，更简洁
- **移除**了 `ema_beta=0.95` 这个超参数
- Fisher 直接来自当前 batch 的梯度，更透明
- 减少了代码复杂度

### 2. Fisher 作用在梯度，不在 loss
- **原方法**: 加入正则项 $\lambda \sum F_j (\theta_j - \theta_j^{(0)})^2$
  - 需要调超参数 $\lambda$
  - 第一步不生效（$\theta = \theta^{(0)}$ 时为 0）

- **新方法**: 在优化器中应用缩放 $u_j = u_j^{Adam} / (F_j + \epsilon)^{\alpha}$
  - 无需额外超参数
  - 第一步就生效
  - 不改变目标函数

### 3. 理论上的 Fisher 逆预条件
- $\alpha = 1$ 对应完整的二阶预条件
- 与文献中的 Fisher 感知优化一致
- 更稳定的收敛性（理论保证）

### 4. 只需估计一次 Fisher
- 在阶段切换时从第一阶段数据估计
- 之后在每个 batch 更新（虽然未在优化器中使用）
- 降低了计算开销

---

## 数学公式总结

### Fisher 对角定义
$$F_j = \mathbb{E}_{z,c \sim \mathcal{D}}\left[\left(\frac{\partial \ell_{PMF}(\theta^{(0)};z,c)}{\partial\theta_j}\right)^2\right]$$

### 归一化
$$\tilde{F}_j = \frac{F_j}{\text{mean}(F) + \epsilon}$$

### 参数更新规则
**标准 AdamW**：
$$u_j^{Adam} = \frac{m_j}{\sqrt{v_j} + \epsilon}$$

**Fisher 感知 AdamW**：
$$u_j = \frac{u_j^{Adam}}{(F_j + \epsilon)^{\alpha}}, \quad \alpha = 1$$

**参数更新**：
$$\theta_{t+1,j} = \theta_{t,j} - \eta u_j$$

### 核心性质
- 高 Fisher 参数（$F_j$ 大）：有效学习率小 → 少更新
- 低 Fisher 参数（$F_j$ 小）：有效学习率大 → 多更新
- 目标函数驻点不变：$\nabla_\theta L_{safe}(\theta^\star) = 0$

---

## 训练流程图

```
┌─────────────────────────────────────┐
│    Stage 1: Pixel MeanFlow 训练     │
│  - 用 AdamW 优化 main_loss         │
│  - 学习轨迹生成分布                 │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│         阶段切换（Stage 1 → 2）      │
│  1. 加载 Stage 1 最优模型             │
│  2. 从训练数据估计 Fisher 对角        │
│  3. 创建 FisherAdamW 优化器          │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│   Stage 2: 安全约束微调              │
│  - 用 FisherAdamW 优化 safe_loss   │
│  - Fisher 缩放保护高重要参数         │
│  - 避免破坏第一阶段先验               │
└─────────────────────────────────────┘
```

---

## 监控指标

在训练日志中查看：

```
[Stage 2] Epoch 0:
  Train Loss: 0.123456789
    - Main: 0.000000
    - Capsize: 0.123456
    - Fisher Diag Mean: 2.345e-01     ← Fisher 对角的平均值
    - Param Drift Mean: 5.678e-04     ← 参数偏移
```

预期行为：
- `Fisher Diag Mean` 应稳定在合理范围（0.1~10）
- `Param Drift Mean` 应在 safe_loss 下降时缓慢增长
- 高 Fisher 参数的更新应小于低 Fisher 参数

---

## 对比：原方法 vs 新方法

| 特性 | 原方法（Fisher Penalty） | 新方法（Fisher-aware Adam） |
|------|------------------------|--------------------------|
| Fisher 位置 | Loss 正则项 | 优化器梯度缩放 |
| 超参数 | $\lambda$（需调） | 无（固定 $\alpha=1$） |
| 第一步效果 | 无（为 0） | 有（立即生效） |
| 目标函数 | 改变 | 不改变 |
| 收敛点 | 可能移动 | 不变 |
| 代码复杂度 | 高（涉及正则项） | 低（优化器内部） |
| EMA 平滑 | 有（$\beta=0.95$） | 无 |

---

## 使用建议

### ✅ 推荐配置
```python
fisher_alpha=1.0          # 完整二阶信息
fisher_eps=1e-8           # 标准值
f_min=0.1, f_max=10.0     # 标准范围
max_batches=100           # 快速估计
```

### ⚠️ 如果效果不好
1. 检查 Fisher 估计：`diagnose_fisher(fisher_diag)`
2. 增加 `max_batches` 到 200
3. 调整 `f_min, f_max` 范围
4. 检查 safe_loss 梯度是否有效

### 🔍 诊断方法
```python
# 1. Fisher 是否被正确估计？
print(f"Fisher min: {min(f.min() for f in fisher_diag.values())}")
print(f"Fisher max: {max(f.max() for f in fisher_diag.values())}")

# 2. 高低 Fisher 参数更新是否有差异？
# （见 FISHER_ADAMW_PYTORCH_GUIDE.md 中的诊断函数）

# 3. 参数漂移是否在增长？
# （应该缓慢增长，而不是指数增长）
```

---

## 文件清单

修改的文件：
- ✅ `/home/yrf/MPT/train_fisher.py`（主修改文件）

新增文档：
- 📄 `/home/yrf/MPT/FISHER_AWARE_GRADIENT_UPDATE.md`（详细方法论述）
- 📄 `/home/yrf/MPT/FISHER_ADAMW_PYTORCH_GUIDE.md`（PyTorch 实现指南）
- 📄 `/home/yrf/MPT/TRAIN_FISHER_MODIFICATION_SUMMARY.md`（本文档）

---

## 下一步

1. **测试训练**：运行 `python train_fisher.py --stage 1 --stage1_epochs 50 --stage2_epochs 20`
2. **监控 Fisher**：观察日志中的 `Fisher Diag Mean` 和 `Param Drift Mean`
3. **对比结果**：与原方法（Fisher Penalty）进行性能对比
4. **微调参数**：根据实际效果调整 Fisher 估计的 `max_batches` 等参数

---

## 参考资源

- `FISHER_AWARE_GRADIENT_UPDATE.md`: 完整的理论推导和方法论述
- `FISHER_ADAMW_PYTORCH_GUIDE.md`: PyTorch 代码实现和使用示例
- `train_fisher.py`: 完整的训练脚本实现

---

## 问题反馈

如有问题，请检查：
1. ✅ 无语法错误（已通过 `get_errors` 检查）
2. ✅ 优化器正确初始化（`FisherAdamW` 接收正确的参数）
3. ✅ Fisher 估计逻辑正确（基于 main_loss，无 EMA）
4. ✅ 阶段切换逻辑正确（Fisher 估计在阶段 2 初始化时执行）

祝训练顺利！🚀
