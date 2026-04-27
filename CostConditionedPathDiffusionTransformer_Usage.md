# CostConditionedPathDiffusionTransformer 使用指南

## 概述

`CostConditionedPathDiffusionTransformer` 是基于 `PathDiffusionTransformer` 扩展的新型神经网络模型，支持将成本（cost）作为额外的条件输入，用于生成条件化的路径规划结果。

## 核心特性

### 1. **成本条件化输入**
- 新增标量成本输入 `cost: (B,)`
- 成本可代表多种度量：路径长度、曲率、碰撞风险、能量消耗等
- 建议将成本归一化到 `[0, 1]` 范围以获得最佳效果

### 2. **正弦位置编码式嵌入**
```python
# 使用与时间步(timestep)完全相同的嵌入策略
self.cost_embedder = TimestepEmbedder(d_model)
```
- 成本通过正弦位置编码转换为 `d_model` 维的嵌入向量
- 保证不同成本值获得不同的特征表示

### 3. **条件融合**
成本条件与时间步、起点、终点一同融合：
```
原始:  [time_emb, time_r_emb, start_emb, goal_emb] → cond_mlp → condition
新增:  [time_emb, time_r_emb, start_emb, goal_emb, cost_emb] → cond_mlp → condition
```

## 使用方法

### 基础使用

```python
import torch
from dit.Models import CostConditionedPathDiffusionTransformer

# 1. 模型初始化
model = CostConditionedPathDiffusionTransformer(
    n_layers=6,
    n_heads=8,
    d_k=64,
    d_v=64,
    d_model=512,
    d_inner=2048,
    pad_idx=0,
    dropout=0.1,
    n_position=1000,
    train_shape=(100, 100),
    n_path_steps=24,
    diffusion_steps=50,
    prediction_type='x0',
    loss_type='x0'
)

# 2. 准备输入
batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

map_input = torch.randn(batch_size, 3, 100, 100, device=device)      # 地图
noisy_path = torch.randn(batch_size, 24, 2, device=device)           # 加噪路径
timestep = torch.rand(batch_size, device=device)                     # 时间步
timestep_r = 1.0 - timestep                                          # 反向时间步
start_pose = torch.randn(batch_size, 4, device=device)               # 起点
goal_pose = torch.randn(batch_size, 4, device=device)                # 终点

# 【新增】成本条件
cost = torch.rand(batch_size, device=device)  # [0, 1] 范围的成本标量

# 3. 前向传播
output = model(
    map_input=map_input,
    noisy_path=noisy_path,
    timestep=timestep,
    timestep_r=timestep_r,
    start_pose=start_pose,
    goal_pose=goal_pose,
    cost=cost  # 新增参数
)

print(output.shape)  # torch.Size([4, 24, 2])
```

### 训练示例

```python
import torch.nn as nn
import torch.optim as optim

model = CostConditionedPathDiffusionTransformer(...)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# 训练循环
for epoch in range(num_epochs):
    model.train()
    
    for batch_idx, (map_input, noisy_path, timestep, start_pose, goal_pose, cost, target) in enumerate(dataloader):
        # 前向传播
        output = model(
            map_input=map_input,
            noisy_path=noisy_path,
            timestep=timestep,
            timestep_r=1.0 - timestep,
            start_pose=start_pose,
            goal_pose=goal_pose,
            cost=cost  # 成本条件
        )
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 采样使用（获取速度场）

```python
model.eval()

# 方式1：基础速度场获取
z_in = torch.randn(batch_size, 24, 2, device=device)
t_val = 0.5
cost_val = 0.3

with torch.no_grad():
    v_field = model.get_velocity(
        z_in=z_in,
        t_scalar=t_val,
        r_scalar=1.0 - t_val,
        cost_scalar=cost_val,  # 新增参数
        map_input=map_input,
        start_pose=start_pose,
        goal_pose=goal_pose
    )

# 方式2：可微分的速度场获取（用于梯度优化）
t_tensor = torch.tensor(0.5, device=device)
cost_tensor = torch.tensor(0.3, device=device)

with torch.enable_grad():
    v_field = model.get_velocity_tensor(
        z_in=z_in,
        t_scalar=t_tensor,
        r_scalar=1.0 - t_tensor,
        cost_scalar=cost_tensor,  # 新增参数
        map_input=map_input,
        start_pose=start_pose,
        goal_pose=goal_pose
    )
```

## 架构细节

### 数据流图

```
┌─────────────────────────────────────────────────────────────┐
│ 输入                                                         │
├──────────────┬──────────────┬──────────────┬───────────────┤
│ 地图         │ 路径         │ 条件信息      │ 成本（新增）   │
│ (B,3,H,W)    │ (B,24,2)     │ t,start,goal │ (B,)          │
└──────┬───────┴──────┬───────┴───────┬──────┴────────┬───────┘
       │              │               │               │
       ▼              ▼               ▼               ▼
   ┌────────┐    ┌──────────┐  ┌────────────┐  ┌──────────────┐
   │CNN特征│    │路径编码  │  │时间步编码  │  │成本编码      │
   │提取   │    │         │  │          │  │(正弦位置编码)│
   └───┬────┘    └────┬─────┘  └─────┬──────┘  └───────┬──────┘
       │              │             │                 │
       │              │      ┌──────▼────────────────▼────────┐
       │              │      │ 条件融合 MLP                    │
       │              │      │ Input: 5×d_model (新增cost)    │
       │              │      └───────────┬─────────────────────┘
       │              │                  │
       │         ┌────▼────┐         ┌──▼───┐
       │         │tokens拼接│         │cond  │
       │         │[S,P,G]  │         │(d_m) │
       │         └────┬────┘         └──┬───┘
       │              │                 │
       └──────┬───────┴────────────────┬┘
              │                        │
              ▼                        ▼
          ┌─────────────────────────────────┐
          │ DiT Blocks (带 Cross-Attention) │
          │ - Self-Attention (路径平滑性)   │
          │ - Cross-Attention (地图融合)    │
          │ - AdaLN (条件调制，含成本)     │
          │ - FFN                           │
          └─────────────┬───────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ 预测头           │
              │ [24,2]输出       │
              └──────────────────┘
```

### 成本嵌入机制

```python
# 核心实现
class CostConditionedPathDiffusionTransformer:
    def __init__(self, ...):
        # 成本使用TimestepEmbedder进行嵌入
        self.cost_embedder = TimestepEmbedder(d_model)
    
    def forward(self, ..., cost):
        # 1. 成本标量 → 正弦位置编码
        if cost.dim() == 0:
            cost_batch = cost.unsqueeze(0).expand(B)
        else:
            cost_batch = cost
        
        # 2. 嵌入成本 (B,) → (B, d_model)
        cost_emb = self.cost_embedder(cost_batch)
        
        # 3. 与其他条件融合
        cond = self.cond_mlp(
            torch.cat([t_emb, r_emb, s_emb, g_emb, cost_emb], dim=-1)
        )
        
        # 4. 通过AdaLN调制DiT层
        for block in self.dit_blocks:
            x = block(x, map_tokens, cond)
```

## 重要参数说明

| 参数 | 类型 | 说明 | 范围 |
|------|------|------|------|
| `cost` | `torch.Tensor` | 成本标量输入 | 建议 [0, 1] |
| `prediction_type` | `str` | 预测类型 | 'x0', 'epsilon', 'v' |
| `loss_type` | `str` | 损失类型 | 'x0', 'epsilon', 'v' |
| `n_path_steps` | `int` | 路径控制点数 | 默认24 |
| `diffusion_steps` | `int` | 扩散采样步数 | 默认50 |

## 与原始PathDiffusionTransformer的区别

| 特性 | PathDiffusionTransformer | CostConditionedPathDiffusionTransformer |
|------|-------------------------|----------------------------------------|
| 成本输入 | ❌ 无 | ✅ 有 `(B,)` 标量 |
| 成本嵌入 | ❌ 无 | ✅ 正弦位置编码 |
| 条件融合 | 4个条件 | 5个条件（新增成本） |
| cond_mlp输入 | 4×d_model | 5×d_model |
| 前向签名 | forward(...) | forward(..., cost) |
| 速度计算 | get_velocity(...) | get_velocity(..., cost_scalar) |
| 参数量 | 54.0M | 54.3M（稍增） |

## 最佳实践

### 1. **成本归一化**
```python
# 推荐做法：将成本归一化到 [0, 1]
cost_raw = compute_path_cost(paths)  # 原始成本，可能范围很大
cost_min, cost_max = cost_raw.min(), cost_raw.max()
cost_normalized = (cost_raw - cost_min) / (cost_max - cost_min + 1e-8)
```

### 2. **成本的语义含义**
- `cost ≈ 0`: 低成本路径（短、光滑、安全）
- `cost ≈ 1`: 高成本路径（长、曲折、危险）
- 可以是多维成本的加权综合

### 3. **批量成本与单一成本**
```python
# 方式1：批量成本（每个样本不同的成本）
cost = torch.tensor([0.2, 0.5, 0.8, 0.3], device=device)  # (B,)
output = model(..., cost=cost)

# 方式2：单一成本（所有样本相同的成本）
cost = torch.tensor(0.5, device=device)  # 标量
output = model(..., cost=cost)  # 内部自动扩展
```

### 4. **梯度流**
```python
# 对成本进行梯度优化
cost = torch.tensor(0.5, device=device, requires_grad=True)
output = model(..., cost=cost)
loss = criterion(output, target)
loss.backward()  # 成本会获得梯度
cost_grad = cost.grad  # 梯度信息
```

## 常见问题

**Q: 成本应该如何定义？**  
A: 成本可以是任何标量度量：路径长度、曲率积分、碰撞概率、能量消耗等。建议归一化到 [0, 1]。

**Q: 成本输入是必须的吗？**  
A: 是的，`forward()` 方法现在需要 `cost` 参数。如果没有实际的成本信息，可以传入 0.5（中等成本）。

**Q: 能否同时控制多个条件（时间步、起终点、成本）？**  
A: 可以。模型已经支持同时调整时间步、起终点位置和成本，实现多条件联合控制。

**Q: 成本对网络输出的影响有多大？**  
A: 取决于训练数据中成本变化与路径输出的相关性。如果训练数据中成本信息很重要，模型会学习到强相关性。

## 与现有训练框架的集成

在现有的 `train_dit.py` 或类似训练脚本中使用新类：

```python
# 旧代码
model = PathDiffusionTransformer(...)
output = model(map_input, noisy_path, timestep, timestep_r, start_pose, goal_pose)

# 新代码
model = CostConditionedPathDiffusionTransformer(...)
# 需要从数据加载器获取成本信息
output = model(map_input, noisy_path, timestep, timestep_r, start_pose, goal_pose, cost)
```

## 参考文献

- DiT: Scalable Diffusion Models with Transformers (Peebles & Xie, 2022)
- Attention Is All You Need (Vaswani et al., 2017)
- 原始 PathDiffusionTransformer 设计
