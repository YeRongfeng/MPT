"""
评估Guided Sampling在多个测试样本上的表现
使用最优参数：guidance_scale=0.15, start_step=0.7
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import json
from dit.Models import PathDiffusionTransformer
from dataLoader_dit import UnevenPathDataLoader
from grad_optimizer import TrajectoryOptimizerSE2
from dataLoader_uneven import compute_map_yaw_bins
from tqdm import tqdm

def evaluate_trajectory(trajectory, cost_map, map_info, yaw_stability, device):
    """评估轨迹的cost和unsafe ratio"""
    optimizer = TrajectoryOptimizerSE2(
        trajectory.detach(), cost_map, map_info, device=device
    )
    with torch.no_grad():
        cost = optimizer.cost_on_poses(trajectory)
    
    traj_np = trajectory.cpu().numpy()
    unsafe_count = 0
    for i in range(len(traj_np)):
        x, y, yaw = traj_np[i]
        x_idx = int((x + 20) / 0.4)
        y_idx = int((y + 20) / 0.4)
        yaw_idx = int((yaw + np.pi) / (2 * np.pi / 36)) % 36
        
        if 0 <= y_idx < yaw_stability.shape[0] and 0 <= x_idx < yaw_stability.shape[1]:
            stability_val = yaw_stability[y_idx, x_idx, yaw_idx]
            if torch.is_tensor(stability_val):
                is_unsafe = (stability_val == 0).item()
            else:
                is_unsafe = (stability_val == 0)
            if is_unsafe:
                unsafe_count += 1
    
    return cost.item(), unsafe_count / len(traj_np)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载数据
    print("Loading dataset...")
    env_list = ['env000008']
    dataFolder = '/home/yrf/MPT/data/sim_dataset/val'
    dataset = UnevenPathDataLoader(env_list, dataFolder, compute_stability_map=True)
    print(f"✓ Loaded {len(dataset)} samples")
    
    # 加载模型
    print("Loading model...")
    modelFolder = 'data/sim'
    modelFile = osp.join(modelFolder, 'model_params.json')
    model_param = json.load(open(modelFile))
    
    model = PathDiffusionTransformer(**model_param['model_args'])
    model = model.to(device)
    checkpoint = torch.load(osp.join(modelFolder, 'stage1_best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded")
    
    map_info = {
        'resolution': 0.4,
        'origin': (-20.0, -20.0, -np.pi),
        'size': (100, 100, 36)
    }
    
    # 测试多个样本
    num_samples = min(10, len(dataset))
    print(f"\nEvaluating on {num_samples} samples...")
    
    results_standard = []
    results_guided = []
    
    for idx in tqdm(range(num_samples)):
        sample = dataset[idx]
        if sample is None:
            continue
        
        # 准备输入
        nx = sample['map'][0, :, :].to(device)
        ny = sample['map'][1, :, :].to(device)
        nz = sample['map'][2, :, :].to(device)
        nz = torch.abs(nz)
        
        cost_map_full = sample['cost_map'].to(device).unsqueeze(0)
        trajectory_gt = sample['trajectory'].to(device)
        start_pose, goal_pose = trajectory_gt[0], trajectory_gt[-1]
        
        # 归一化
        start_normalized = torch.zeros(1, 4, device=device)
        start_normalized[0, :2] = start_pose[:2] / 20.0
        start_normalized[0, 2] = torch.sin(start_pose[2])
        start_normalized[0, 3] = torch.cos(start_pose[2])
        
        goal_normalized = torch.zeros(1, 4, device=device)
        goal_normalized[0, :2] = goal_pose[:2] / 20.0
        goal_normalized[0, 2] = torch.sin(goal_pose[2])
        goal_normalized[0, 3] = torch.cos(goal_pose[2])
        
        map_input = torch.stack([nx, ny, nz], dim=0).unsqueeze(0)
        yaw_stability = compute_map_yaw_bins(nx, ny, nz, yaw_bins=36)
        cost_map = cost_map_full[0].permute(2, 0, 1)
        
        # Standard sampling
        with torch.no_grad():
            traj_std = model.sample(
                map_input, start_normalized, goal_normalized,
                num_samples=1, ddim_steps=50
            )[0]
        
        traj_std_denorm = torch.zeros(20, 3, device=device)
        traj_std_denorm[:, :2] = traj_std[:, :2] * 20.0
        traj_std_denorm[:, 2] = torch.atan2(traj_std[:, 2], traj_std[:, 3])
        
        traj_std_full = torch.cat([
            start_pose.unsqueeze(0),
            traj_std_denorm,
            goal_pose.unsqueeze(0)
        ], dim=0)
        
        # Guided sampling
        traj_guided = model.guided_sample(
            map_input, start_normalized, goal_normalized,
            cost_map_full, map_info,
            num_samples=1, ddim_steps=50,
            guidance_scale=0.15,
            guidance_start_step=0.7
        )[0]
        
        traj_guided_denorm = torch.zeros(20, 3, device=device)
        traj_guided_denorm[:, :2] = traj_guided[:, :2] * 20.0
        traj_guided_denorm[:, 2] = torch.atan2(traj_guided[:, 2], traj_guided[:, 3])
        
        traj_guided_full = torch.cat([
            start_pose.unsqueeze(0),
            traj_guided_denorm,
            goal_pose.unsqueeze(0)
        ], dim=0)
        
        # 评估
        cost_std, unsafe_std = evaluate_trajectory(
            traj_std_full, cost_map, map_info, yaw_stability, device
        )
        cost_guided, unsafe_guided = evaluate_trajectory(
            traj_guided_full, cost_map, map_info, yaw_stability, device
        )
        
        results_standard.append({'cost': cost_std, 'unsafe': unsafe_std})
        results_guided.append({'cost': cost_guided, 'unsafe': unsafe_guided})
    
    # 统计结果
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    
    costs_std = [r['cost'] for r in results_standard]
    costs_guided = [r['cost'] for r in results_guided]
    unsafe_std = [r['unsafe'] for r in results_standard]
    unsafe_guided = [r['unsafe'] for r in results_guided]
    
    print(f"\nStandard Sampling:")
    print(f"  Cost: mean={np.mean(costs_std):.4e}, std={np.std(costs_std):.4e}")
    print(f"  Unsafe: mean={np.mean(unsafe_std)*100:.2f}%, std={np.std(unsafe_std)*100:.2f}%")
    
    print(f"\nGuided Sampling (scale=0.15, start=0.7):")
    print(f"  Cost: mean={np.mean(costs_guided):.4e}, std={np.std(costs_guided):.4e}")
    print(f"  Unsafe: mean={np.mean(unsafe_guided)*100:.2f}%, std={np.std(unsafe_guided)*100:.2f}%")
    
    cost_improvement = (np.mean(costs_std) - np.mean(costs_guided)) / np.mean(costs_std) * 100
    unsafe_improvement = (np.mean(unsafe_std) - np.mean(unsafe_guided)) / (np.mean(unsafe_std) + 1e-9) * 100
    
    print(f"\nImprovement:")
    print(f"  Cost: {cost_improvement:+.2f}%")
    print(f"  Unsafe: {unsafe_improvement:+.2f}%")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(costs_std, costs_guided, alpha=0.6, s=100)
    axes[0].plot([min(costs_std), max(costs_std)], 
                 [min(costs_std), max(costs_std)], 'r--', label='y=x')
    axes[0].set_xlabel('Standard Cost')
    axes[0].set_ylabel('Guided Cost')
    axes[0].set_title('Cost Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(np.array(unsafe_std)*100, np.array(unsafe_guided)*100, alpha=0.6, s=100)
    axes[1].plot([0, max(unsafe_std)*100], [0, max(unsafe_std)*100], 'r--', label='y=x')
    axes[1].set_xlabel('Standard Unsafe (%)')
    axes[1].set_ylabel('Guided Unsafe (%)')
    axes[1].set_title('Unsafe Point Ratio Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('guided_sampling_eval.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization to guided_sampling_eval.png")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
