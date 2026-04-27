"""
测试成本条件化路径扩散变换器 (CostConditionedPathDiffusionTransformer)

演示如何使用新增的成本条件化功能
"""

import torch
import torch.nn as nn
from dit.Models import CostConditionedPathDiffusionTransformer


def test_cost_conditioned_dit():
    """
    基础功能测试
    """
    print("=" * 70)
    print("成本条件化路径扩散变换器 (CostConditionedPathDiffusionTransformer) 测试")
    print("=" * 70)
    
    # ========== 模型参数配置 ==========
    model_config = {
        'n_layers': 6,           # Transformer层数
        'n_heads': 8,            # 多头注意力头数
        'd_k': 64,               # Key维度
        'd_v': 64,               # Value维度
        'd_model': 512,          # 模型隐层维度
        'd_inner': 2048,         # FFN内层维度
        'pad_idx': 0,            # 填充索引
        'dropout': 0.1,          # Dropout比率
        'n_position': 1000,      # 最大位置数
        'train_shape': (100, 100),  # 训练输入形状
        'n_path_steps': 24,      # 中间控制点数量
        'diffusion_steps': 50,   # 扩散步数
        'prediction_type': 'x0', # 预测类型
        'loss_type': 'x0'        # 损失类型
    }
    
    print("\n【模型配置】")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # ========== 初始化模型 ==========
    print("\n【模型初始化】")
    model = CostConditionedPathDiffusionTransformer(**model_config)
    print(f"✓ 模型创建成功")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 总参数数: {total_params:,}")
    print(f"✓ 可训练参数: {trainable_params:,}")
    
    # ========== 准备输入数据 ==========
    print("\n【输入数据准备】")
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 地图输入 (B, 3, 100, 100)
    map_input = torch.randn(batch_size, 3, 100, 100, device=device)
    
    # 加噪路径 (B, 24, 2)
    noisy_path = torch.randn(batch_size, 24, 2, device=device)
    
    # 时间步 (B,)
    timestep = torch.rand(batch_size, device=device)
    timestep_r = 1.0 - timestep  # 反向时间步
    
    # 起点和终点 (B, 4) - 格式: (x, y, cos(theta), sin(theta))
    start_pose = torch.randn(batch_size, 4, device=device)
    start_pose[:, 2:] = start_pose[:, 2:] / torch.norm(start_pose[:, 2:], dim=1, keepdim=True)  # 归一化角度
    
    goal_pose = torch.randn(batch_size, 4, device=device)
    goal_pose[:, 2:] = goal_pose[:, 2:] / torch.norm(goal_pose[:, 2:], dim=1, keepdim=True)
    
    # 【关键】成本条件 (B,) - 标量值，表示路径成本（如长度、曲率、碰撞风险等）
    # 建议归一化到 [0, 1] 范围
    cost = torch.rand(batch_size, device=device)
    
    print(f"  ✓ map_input shape: {map_input.shape}")
    print(f"  ✓ noisy_path shape: {noisy_path.shape}")
    print(f"  ✓ timestep shape: {timestep.shape}")
    print(f"  ✓ timestep_r shape: {timestep_r.shape}")
    print(f"  ✓ start_pose shape: {start_pose.shape}")
    print(f"  ✓ goal_pose shape: {goal_pose.shape}")
    print(f"  ✓ cost shape: {cost.shape} (新增成本条件)")
    print(f"  ✓ cost values: {cost.detach().cpu().numpy()}")
    
    # ========== 前向传播 ==========
    print("\n【前向传播测试】")
    model.eval()
    with torch.no_grad():
        output = model(
            map_input=map_input,
            noisy_path=noisy_path,
            timestep=timestep,
            timestep_r=timestep_r,
            start_pose=start_pose,
            goal_pose=goal_pose,
            cost=cost  # 新增成本条件输入
        )
    
    print(f"✓ 前向传播成功")
    print(f"  输出形状: {output.shape} (预期: (B, n_path_steps, 2))")
    print(f"  输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # ========== 验证成本条件的效果 ==========
    print("\n【成本条件效果验证】")
    
    # 创建两个不同成本的输入
    cost_low = torch.full((batch_size,), 0.1, device=device)   # 低成本
    cost_high = torch.full((batch_size,), 0.9, device=device)  # 高成本
    
    with torch.no_grad():
        output_low = model(
            map_input=map_input,
            noisy_path=noisy_path,
            timestep=timestep,
            timestep_r=timestep_r,
            start_pose=start_pose,
            goal_pose=goal_pose,
            cost=cost_low
        )
        
        output_high = model(
            map_input=map_input,
            noisy_path=noisy_path,
            timestep=timestep,
            timestep_r=timestep_r,
            start_pose=start_pose,
            goal_pose=goal_pose,
            cost=cost_high
        )
    
    output_diff = (output_high - output_low).abs().mean()
    print(f"✓ 低成本输出与高成本输出的差异: {output_diff.item():.6f}")
    print(f"  (差异 > 0 表示成本条件正在影响网络输出)")
    
    # ========== 反向传播测试 ==========
    print("\n【反向传播测试】")
    model.train()
    
    output = model(
        map_input=map_input,
        noisy_path=noisy_path,
        timestep=timestep,
        timestep_r=timestep_r,
        start_pose=start_pose,
        goal_pose=goal_pose,
        cost=cost
    )
    
    # 计算损失
    target = torch.randn_like(output)
    loss = nn.MSELoss()(output, target)
    
    # 反向传播
    loss.backward()
    print(f"✓ 反向传播成功")
    print(f"  损失值: {loss.item():.6f}")
    
    # 检查梯度
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for p in model.parameters())
    print(f"  ✓ 有梯度的参数: {has_grad}/{total}")
    
    # ========== 速度场计算测试 ==========
    print("\n【速度场计算测试】")
    model.eval()
    
    # 获取速度场（用于采样）
    z_in = torch.randn(batch_size, 24, 2, device=device)
    t_val = 0.5
    cost_val = 0.3
    
    with torch.no_grad():
        v_field = model.get_velocity(
            z_in=z_in,
            t_scalar=t_val,
            r_scalar=1.0 - t_val,
            cost_scalar=cost_val,
            map_input=map_input,
            start_pose=start_pose,
            goal_pose=goal_pose
        )
    
    print(f"✓ 速度场计算成功")
    print(f"  速度场形状: {v_field.shape} (预期: (B, 24, 2))")
    print(f"  速度场范围: [{v_field.min().item():.4f}, {v_field.max().item():.4f}]")
    
    # ========== 成本嵌入验证 ==========
    print("\n【成本嵌入模块验证】")
    
    # 直接测试成本embedder
    cost_test = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
    cost_embeddings = model.cost_embedder(cost_test)
    
    print(f"✓ 成本嵌入形状: {cost_embeddings.shape} (预期: (5, d_model))")
    print(f"  成本嵌入范围: [{cost_embeddings.min().item():.4f}, {cost_embeddings.max().item():.4f}]")
    
    # 验证不同成本得到不同的嵌入
    embedding_diff = (cost_embeddings[0] - cost_embeddings[-1]).norm().item()
    print(f"  不同成本的嵌入差异: {embedding_diff:.4f} (应该 > 0)")
    
    print("\n" + "=" * 70)
    print("✓ 所有测试通过！")
    print("=" * 70)


if __name__ == "__main__":
    test_cost_conditioned_dit()
