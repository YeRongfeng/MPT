"""
CostConditionedPathDiffusionTransformer 实际使用示例

演示如何在实际的路径规划项目中集成成本条件化功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dit.Models import CostConditionedPathDiffusionTransformer


class PathDataset(Dataset):
    """
    示例数据集类
    在实际使用中需要根据你的数据格式进行调整
    """
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        返回格式：
        - map_input: (3, H, W) 地图
        - noisy_path: (24, 2) 加噪路径
        - timestep: 标量时间步
        - start_pose: (4,) 起点位姿
        - goal_pose: (4,) 目标位姿
        - cost: 标量成本（新增）
        - target: (24, 2) 目标输出
        """
        map_input = torch.randn(3, 100, 100)
        noisy_path = torch.randn(24, 2)
        timestep = torch.rand(1).item()
        start_pose = torch.randn(4)
        start_pose[2:] /= torch.norm(start_pose[2:])
        goal_pose = torch.randn(4)
        goal_pose[2:] /= torch.norm(goal_pose[2:])
        cost = torch.rand(1).item()  # 成本标量 [0, 1]
        target = torch.randn(24, 2)
        
        return {
            'map_input': map_input,
            'noisy_path': noisy_path,
            'timestep': timestep,
            'start_pose': start_pose,
            'goal_pose': goal_pose,
            'cost': cost,
            'target': target
        }


def collate_fn(batch):
    """
    自定义collate函数处理批量数据
    """
    map_inputs = torch.stack([item['map_input'] for item in batch])
    noisy_paths = torch.stack([item['noisy_path'] for item in batch])
    timesteps = torch.tensor([item['timestep'] for item in batch])
    start_poses = torch.stack([item['start_pose'] for item in batch])
    goal_poses = torch.stack([item['goal_pose'] for item in batch])
    costs = torch.tensor([item['cost'] for item in batch])  # 成本批量化
    targets = torch.stack([item['target'] for item in batch])
    
    return {
        'map_input': map_inputs,
        'noisy_path': noisy_paths,
        'timestep': timesteps,
        'start_pose': start_poses,
        'goal_pose': goal_poses,
        'cost': costs,
        'target': targets
    }


def train_example():
    """
    训练循环示例
    """
    print("=" * 70)
    print("CostConditionedPathDiffusionTransformer 训练示例")
    print("=" * 70)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 模型配置
    model = CostConditionedPathDiffusionTransformer(
        n_layers=4,              # 使用较少的层加快演示
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
    ).to(device)
    
    print(f"✓ 模型初始化完成")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 数据加载
    print("\n【数据加载】")
    dataset = PathDataset(num_samples=20)  # 小数据集用于演示
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    print(f"✓ 数据集加载完成 (20个样本)")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 简短的训练循环（演示用）
    print("\n【训练】")
    num_epochs = 2
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # 将数据移到设备
            map_input = batch['map_input'].to(device)
            noisy_path = batch['noisy_path'].to(device)
            timestep = batch['timestep'].to(device)
            start_pose = batch['start_pose'].to(device)
            goal_pose = batch['goal_pose'].to(device)
            cost = batch['cost'].to(device)  # 成本数据
            target = batch['target'].to(device)
            
            # 【关键】前向传播包含成本条件
            output = model(
                map_input=map_input,
                noisy_path=noisy_path,
                timestep=timestep,
                timestep_r=1.0 - timestep,
                start_pose=start_pose,
                goal_pose=goal_pose,
                cost=cost  # 新增参数
            )
            
            # 计算损失
            loss = criterion(output, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"✓ Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    
    return model


def inference_example(model):
    """
    推理示例
    """
    print("\n" + "=" * 70)
    print("推理示例 - 使用不同成本条件生成路径")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # 准备输入数据
    batch_size = 1
    map_input = torch.randn(batch_size, 3, 100, 100, device=device)
    noisy_path = torch.randn(batch_size, 24, 2, device=device)
    timestep = torch.tensor([0.5], device=device)
    start_pose = torch.tensor([[0.2, 0.3, 1.0, 0.0]], device=device)
    goal_pose = torch.tensor([[0.8, 0.7, 0.0, 1.0]], device=device)
    
    # 测试不同的成本条件
    cost_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print("\n使用不同成本条件生成路径:")
    print("-" * 70)
    
    with torch.no_grad():
        for cost_val in cost_values:
            cost = torch.tensor([cost_val], device=device)
            
            output = model(
                map_input=map_input,
                noisy_path=noisy_path,
                timestep=timestep,
                timestep_r=1.0 - timestep,
                start_pose=start_pose,
                goal_pose=goal_pose,
                cost=cost
            )
            
            # 统计输出信息
            output_norm = torch.norm(output)
            output_std = output.std()
            
            print(f"  成本 = {cost_val:.1f} | 输出范数: {output_norm:.4f} | 标准差: {output_std:.4f}")
    
    print("-" * 70)
    print("✓ 推理完成 (不同成本条件下的路径差异)")
    
    # 速度场计算示例
    print("\n【速度场计算】(用于采样过程)")
    print("-" * 70)
    
    z_in = torch.randn(batch_size, 24, 2, device=device)
    t_val = 0.3
    cost_val = 0.5
    
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
    
    print(f"  t = {t_val}, cost = {cost_val}")
    print(f"  速度场范数: {v_field.norm():.4f}")
    print(f"  速度场范围: [{v_field.min():.4f}, {v_field.max():.4f}]")


def cost_analysis_example(model):
    """
    成本分析示例 - 研究成本如何影响网络输出
    """
    print("\n" + "=" * 70)
    print("成本影响分析 - 衡量成本条件的有效性")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # 固定其他条件
    batch_size = 1
    map_input = torch.randn(batch_size, 3, 100, 100, device=device)
    noisy_path = torch.randn(batch_size, 24, 2, device=device)
    timestep = torch.tensor([0.5], device=device)
    start_pose = torch.tensor([[0.2, 0.3, 1.0, 0.0]], device=device)
    goal_pose = torch.tensor([[0.8, 0.7, 0.0, 1.0]], device=device)
    
    # 遍历成本值并记录输出
    cost_values = torch.linspace(0, 1, 10, device=device)
    outputs = []
    
    print("\n遍历成本值 [0, 1]:")
    print("-" * 70)
    
    with torch.no_grad():
        for cost_val in cost_values:
            cost = cost_val.unsqueeze(0)
            
            output = model(
                map_input=map_input,
                noisy_path=noisy_path,
                timestep=timestep,
                timestep_r=1.0 - timestep,
                start_pose=start_pose,
                goal_pose=goal_pose,
                cost=cost
            )
            
            outputs.append(output)
    
    outputs = torch.stack(outputs)  # (10, 1, 24, 2)
    
    # 计算相邻成本输出的差异
    diffs = torch.norm(outputs[1:] - outputs[:-1], dim=(1, 2))
    avg_diff = diffs.mean().item()
    max_diff = diffs.max().item()
    
    print(f"  相邻成本步长的平均差异: {avg_diff:.6f}")
    print(f"  相邻成本步长的最大差异: {max_diff:.6f}")
    print(f"  总体变化范围: {avg_diff * 9:.6f}")
    
    if avg_diff > 1e-6:
        print(f"  ✓ 成本条件正在有效影响网络输出")
    else:
        print(f"  ⚠ 成本条件影响较小（可能需要训练优化）")
    
    print("-" * 70)


if __name__ == "__main__":
    # 1. 训练示例
    model = train_example()
    
    # 2. 推理示例
    inference_example(model)
    
    # 3. 成本影响分析
    cost_analysis_example(model)
    
    print("\n✓ 所有示例完成！")
