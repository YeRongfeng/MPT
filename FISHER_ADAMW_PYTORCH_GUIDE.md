# Fisher 感知安全梯度微调 - PyTorch 实现快速参考

## 核心流程代码片段

### 第一步：定义 FisherAdamW 优化器

```python
import math
import torch
from torch.optim import Optimizer

class FisherAdamW(Optimizer):
    """Fisher 感知的 AdamW 优化器"""
    
    def __init__(
        self,
        named_params,
        fisher_diag,
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        fisher_eps=1e-8,
        fisher_alpha=1.0,
    ):
        named_params = [(n, p) for n, p in named_params if p.requires_grad]
        params = [p for _, p in named_params]
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fisher_eps=fisher_eps,
            fisher_alpha=fisher_alpha,
        )
        
        super().__init__(params, defaults)
        
        # 保存 Fisher 对角（参数id -> 张量）
        self.fisher_by_id = {}
        for name, p in named_params:
            self.fisher_by_id[id(p)] = fisher_diag[name].detach()
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            fisher_eps = group["fisher_eps"]
            fisher_alpha = group["fisher_alpha"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.detach()
                
                if grad.is_sparse:
                    raise RuntimeError("FisherAdamW does not support sparse gradients.")
                
                state = self.state[p]
                
                # 初始化
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                
                state["step"] += 1
                step = state["step"]
                
                # Adam 矩更新
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # 偏置修正
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                
                # Adam 梯度
                denom = exp_avg_sq.sqrt()
                denom.div_(math.sqrt(bias_correction2))
                denom.add_(eps)
                
                adam_update = exp_avg / bias_correction1 / denom
                
                # Fisher 缩放
                fisher = self.fisher_by_id[id(p)].to(device=p.device, dtype=p.dtype)
                fisher_scale = (fisher + fisher_eps).pow(fisher_alpha)
                
                update = adam_update / fisher_scale
                
                # 权重衰减（decoupled）
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                
                # 参数更新
                p.add_(update, alpha=-lr)
        
        return loss
```

### 第二步：估计 Fisher 对角

```python
def estimate_diag_fisher_from_main_loss(
    model,
    dataloader,
    compute_loss_fn,
    device,
    max_batches=None,
    f_min=0.1,
    f_max=10.0,
):
    """
    从 main_loss 梯度的平方估计对角 Fisher
    
    使用公式：F_j = E[(∂L_main/∂θ_j)²]
    """
    model.eval()
    
    fisher = {
        name: torch.zeros_like(p, device=device)
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    
    num_batches = 0
    
    with torch.enable_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            model.zero_grad(set_to_none=True)
            
            # 移动数据到设备
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            # 计算 main_loss
            main_loss = compute_loss_fn(model, batch)
            
            if main_loss.dim() > 0:
                main_loss = main_loss.mean()
            
            # 反向传播
            main_loss.backward()
            
            # 累积梯度平方
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[name] += p.grad.detach().pow(2)
            
            num_batches += 1
    
    # 平均化
    for name in fisher:
        fisher[name] /= max(num_batches, 1)
    
    # 全局归一化
    all_fishers = list(fisher.values())
    if len(all_fishers) > 0:
        mean_f = torch.stack([v.mean() for v in all_fishers]).mean()
        
        for name in fisher:
            fisher[name] = fisher[name] / (mean_f + 1e-8)
            # 裁剪到合理范围
            fisher[name] = torch.clamp(fisher[name], f_min, f_max)
            fisher[name] = fisher[name].detach()
    
    return fisher
```

### 第三步：在阶段 2 切换时使用

```python
# 当从阶段 1 切换到阶段 2 时：

# 1. 估计 Fisher
fisher_diag = estimate_diag_fisher_from_main_loss(
    model=model,
    dataloader=train_dataloader,  # 第一阶段的训练数据
    compute_loss_fn=compute_main_loss_fn,
    device=device,
    max_batches=100,  # 用前 100 个 batch 估计
    f_min=0.1,
    f_max=10.0
)

# 2. 创建 FisherAdamW 优化器
optimizer = FisherAdamW(
    named_params=list(model.named_parameters()),
    fisher_diag=fisher_diag,
    lr=1e-4,  # 基础学习率
    betas=(0.95, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    fisher_eps=1e-8,
    fisher_alpha=1.0  # 重要：alpha=1
)

# 3. 阶段 2 训练循环（正常 PyTorch 训练）
model.train()

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        
        # 前向传播计算 safe_loss
        traj = model(batch)
        safe_loss = compute_safe_loss(traj, batch)
        
        # 反向传播
        safe_loss.backward()
        
        # 可选：梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步骤（Fisher 感知缩放在这里应用）
        optimizer.step()
        
        print(f"Loss: {safe_loss.item()}")
```

---

## 数学原理对应代码

### 原理 1：Fisher 对角定义

$$F_j = \mathbb{E}\left[\left(\frac{\partial \ell_{PMF}}{\partial\theta_j}\right)^2\right]$$

对应代码：
```python
# 在 estimate_diag_fisher_from_main_loss 中：
main_loss.backward()  # 计算 ∂L/∂θ
fisher[name] += p.grad.detach().pow(2)  # 累积梯度平方
```

### 原理 2：归一化

$$\tilde{F}_j = \frac{F_j}{\text{mean}(F) + \epsilon}$$

对应代码：
```python
mean_f = torch.stack([v.mean() for v in all_fishers]).mean()
for name in fisher:
    fisher[name] = fisher[name] / (mean_f + 1e-8)
```

### 原理 3：Fisher 感知更新

$$\theta_{t+1,j} = \theta_{t,j} - \eta \frac{u_j^{Adam}}{F_j + \epsilon}$$

对应代码（在 FisherAdamW.step() 中）：
```python
adam_update = exp_avg / bias_correction1 / denom  # u_j^{Adam}
fisher_scale = (fisher + fisher_eps).pow(fisher_alpha)  # (F_j + ε)^α
update = adam_update / fisher_scale  # 应用 Fisher 缩放
p.add_(update, alpha=-lr)  # 参数更新
```

---

## 关键配置参数

| 参数 | 说明 | 推荐值 | 范围 |
|------|------|--------|------|
| `fisher_alpha` | Fisher 缩放指数 | 1.0 | [0.5, 1.0, 2.0]（1.0最推荐） |
| `fisher_eps` | Fisher 数值稳定化 | 1e-8 | [1e-10, 1e-6] |
| `f_min` | Fisher 下界裁剪 | 0.1 | [0.01, 1.0] |
| `f_max` | Fisher 上界裁剪 | 10.0 | [1.0, 100.0] |
| `max_batches` | Fisher 估计用的 batch 数 | 100 | [50, 200] |
| `lr` | 基础学习率 | 1e-4 | [1e-5, 1e-3] |

---

## 诊断指标

在训练循环中，可以添加以下诊断代码：

```python
# 诊断 1：Fisher 的参数分布
def diagnose_fisher(fisher_diag):
    all_vals = torch.cat([v.flatten() for v in fisher_diag.values()])
    print(f"  Fisher min: {all_vals.min():.3e}")
    print(f"  Fisher max: {all_vals.max():.3e}")
    print(f"  Fisher mean: {all_vals.mean():.3e}")
    print(f"  Fisher median: {all_vals.median():.3e}")

# 诊断 2：高低 Fisher 参数的更新对比
def compare_updates_by_fisher(model, fisher_diag, threshold=0.5):
    all_fisher_vals = []
    for name, f in fisher_diag.items():
        all_fisher_vals.extend(f.flatten().tolist())
    
    median_fisher = sorted(all_fisher_vals)[len(all_fisher_vals)//2]
    high_fisher_threshold = median_fisher * (1 + threshold)
    low_fisher_threshold = median_fisher * (1 - threshold)
    
    high_updates_sq = []
    low_updates_sq = []
    
    for name, p in model.named_parameters():
        if p.grad is not None and name in fisher_diag:
            f = fisher_diag[name]
            update_sq = p.grad.detach().pow(2)
            
            high_mask = f > high_fisher_threshold
            low_mask = f < low_fisher_threshold
            
            if high_mask.any():
                high_updates_sq.append(update_sq[high_mask].mean())
            if low_mask.any():
                low_updates_sq.append(update_sq[low_mask].mean())
    
    if high_updates_sq:
        print(f"  High Fisher param updates: {torch.stack(high_updates_sq).mean():.3e}")
    if low_updates_sq:
        print(f"  Low Fisher param updates: {torch.stack(low_updates_sq).mean():.3e}")

# 诊断 3：参数漂移
def diagnose_param_drift(model, ref_params):
    total_drift_sq = 0
    for (name, p), ref_p in zip(model.named_parameters(), ref_params):
        if p.requires_grad:
            drift = p - ref_p
            total_drift_sq += (drift.pow(2).sum().item())
    
    return total_drift_sq ** 0.5

# 在训练循环中使用
print("=== Fisher Diagnostics ===")
diagnose_fisher(fisher_diag)
compare_updates_by_fisher(model, fisher_diag)
param_drift = diagnose_param_drift(model, ref_params)
print(f"Total param drift (L2): {param_drift:.3e}")
```

---

## 常见问题和解决方案

### Q1: Fisher 值全为零？
A: 
- 检查 main_loss 梯度是否为零（可能 main_loss weight 设为 0）
- 确保在估计 Fisher 时 main_loss 是可微的
- 检查是否有 `torch.no_grad()` 意外包装了代码

### Q2: 优化器更新幅度过小？
A:
- 检查 `fisher_alpha` 是否设得太大（推荐 1.0）
- 检查 `fisher_eps` 是否设得太大
- 尝试增加 `max_batches` 以获得更稳定的 Fisher 估计

### Q3: 第二阶段 loss 不下降？
A:
- Fisher 可能估计不准确，尝试 `max_batches=200`
- 检查 safe_loss 计算是否正确
- 尝试调整 `f_min`, `f_max` 裁剪范围

### Q4: 内存溢出？
A:
- 在 `estimate_diag_fisher_from_main_loss` 中减少 `max_batches`
- 减小 batch size
- 在 Fisher 估计时用 `model.eval()` + 禁用梯度累积

---

## 与其他方法的比较

### vs. 标准 AdamW（无 Fisher）
- 优点：保护高重要性参数，防止第一阶段先验破坏
- 缺点：多了一次 Fisher 估计开销

### vs. Fisher penalty（加入 loss 正则项）
```python
# Fisher penalty 版本
loss = safe_loss + lambda * sum(F_j * (θ_j - θ_j^(0))^2)

# Fisher-aware 版本（本实现）
# 更新中应用缩放：u_j = u_j^Adam / (F_j + ε)^α
```
- Fisher-aware 版本优点：
  - 不改变目标函数
  - 第一步就生效
  - 不需要调 lambda

---

## 参考文献和引用

该方法基于 Fisher 信息矩阵的性质：

1. **Fisher Information Matrix**: 用于衡量参数对模型输出的影响
2. **Fisher-aware optimization**: Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks" (EWC)
3. **Preconditioned Gradient Descent**: 使用二阶信息预条件梯度

本实现简化了全 Fisher 矩阵到对角 Fisher，从而大幅降低计算复杂度。

---

## 完整使用示例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 假设有以下对象：
# - model: nn.Module
# - train_loader, val_loader: DataLoader
# - compute_main_loss_fn: 计算 main_loss 的函数
# - compute_safe_loss: 计算 safe_loss 的函数

# ============ 第一阶段训练（常规） ============
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(stage1_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        main_loss = compute_main_loss_fn(model, batch)
        main_loss.backward()
        optimizer.step()

# ============ 阶段切换 ============
print("Switching to Stage 2...")

# 估计 Fisher
fisher_diag = estimate_diag_fisher_from_main_loss(
    model=model,
    dataloader=train_loader,
    compute_loss_fn=compute_main_loss_fn,
    device=device,
    max_batches=100
)

# 创建 Fisher 感知优化器
optimizer = FisherAdamW(
    named_params=list(model.named_parameters()),
    fisher_diag=fisher_diag,
    lr=1e-4,
    fisher_alpha=1.0
)

# ============ 第二阶段训练（Fisher 感知） ============
for epoch in range(stage2_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        
        # 计算 safe_loss
        traj = model(batch)
        safe_loss = compute_safe_loss(traj, batch)
        
        # 反向传播
        safe_loss.backward()
        
        # Fisher 感知更新（在这一步自动应用）
        optimizer.step()
        
    # 验证
    val_loss = 0
    for batch in val_loader:
        traj = model(batch)
        val_loss += compute_safe_loss(traj, batch).item()
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch}: train_loss={safe_loss:.4f}, val_loss={val_loss:.4f}")
```

这就是完整的 Fisher 感知安全梯度微调的 PyTorch 实现！
