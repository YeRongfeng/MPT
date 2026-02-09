"""
测试真实轨迹数据的B样条拟合
支持不定长度的密集轨迹，模仿vis_dit.py的数据读取方式
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
from os import path as osp
import sys

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用支持的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from bspline_utils import (
    fit_bspline_least_squares,
    reconstruct_from_control_points,
    reconstruct_from_basis_matrix,
    compute_reconstruction_errors,
    compute_trajectory_angle
)


def resample_by_arc_length(trajectory, num_samples):
    """
    按弧长重采样轨迹，使得采样点在弧长上均匀分布
    
    Args:
        trajectory: (N, D) 轨迹数组
        num_samples: 重采样后的点数
        
    Returns:
        resampled_trajectory: (num_samples, D) 重采样后的轨迹
    """
    if len(trajectory) < 2:
        return trajectory
    
    # 1. 计算累积弧长
    diffs = np.diff(trajectory[:, :2], axis=0)  # 只用xy计算弧长
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.zeros(len(trajectory))
    cum_lengths[1:] = np.cumsum(seg_lengths)
    
    # 2. 归一化到[0, 1]
    total_length = cum_lengths[-1]
    if total_length < 1e-8:
        # 退化情况，所有点在同一位置
        return np.tile(trajectory[0], (num_samples, 1))
    
    normalized_cum = cum_lengths / total_length
    
    # 3. 均匀采样目标弧长
    target_arc_lengths = np.linspace(0, 1, num_samples)
    
    # 4. 插值到目标弧长位置
    resampled = np.zeros((num_samples, trajectory.shape[1]))
    for d in range(trajectory.shape[1]):
        if d == 2:  # 角度需要特殊处理
            # 展开角度以处理周期性
            angles_unwrapped = np.unwrap(trajectory[:, d])
            angles_interp = np.interp(target_arc_lengths, normalized_cum, angles_unwrapped)
            # 重新wrap到[-pi, pi]
            resampled[:, d] = np.arctan2(np.sin(angles_interp), np.cos(angles_interp))
        else:
            resampled[:, d] = np.interp(target_arc_lengths, normalized_cum, trajectory[:, d])
    
    return resampled


# ==========================================
# 数据加载接口（模仿vis_dit.py）
# ==========================================

def load_trajectory_from_pickle(path_file):
    """
    从pickle文件加载轨迹数据（与vis_dit.py相同的格式）
    
    Args:
        path_file: pickle文件路径
        
    Returns:
        trajectory: (N, 3) 轨迹数组 [x, y, theta]
        path_data: 完整的pickle数据字典
    """
    with open(path_file, 'rb') as f:
        path_data = pickle.load(f)
        trajectory = path_data['path']  # [N, 3]
    
    return trajectory, path_data


def load_trajectory_dataset(dataset_path, env_type, path_indices=None, max_paths=None):
    """
    批量加载轨迹数据集
    
    Args:
        dataset_path: 数据集根路径，例如 'data/sim_dataset/val'
        env_type: 环境类型，例如 'env000008'
        path_indices: 要加载的路径索引列表，如果为None则加载所有
        max_paths: 最多加载的路径数量
        
    Returns:
        trajectories: 轨迹列表
        metadata: 元数据列表
    """
    env_folder = osp.join(dataset_path, env_type)
    
    if not osp.exists(env_folder):
        raise ValueError(f"Environment folder not found: {env_folder}")
    
    # 如果没有指定路径索引，扫描所有path_*.p文件
    if path_indices is None:
        path_files = [f for f in os.listdir(env_folder) if f.startswith('path_') and f.endswith('.p')]
        path_indices = [int(f.split('_')[1].split('.')[0]) for f in path_files]
        path_indices.sort()
    
    if max_paths is not None:
        path_indices = path_indices[:max_paths]
    
    trajectories = []
    metadata = []
    
    for idx in path_indices:
        path_file = osp.join(env_folder, f'path_{idx}.p')
        if not osp.exists(path_file):
            print(f"Warning: Path file not found: {path_file}")
            continue
        
        try:
            trajectory, path_data = load_trajectory_from_pickle(path_file)
            trajectories.append(trajectory)
            metadata.append({
                'path_idx': idx,
                'env_type': env_type,
                'path_file': path_file,
                'length': len(trajectory),
                'path_data': path_data
            })
        except Exception as e:
            print(f"Error loading {path_file}: {e}")
            continue
    
    print(f"Loaded {len(trajectories)} trajectories from {env_type}")
    return trajectories, metadata


# ==========================================
# 轨迹拟合和重建测试
# ==========================================

def test_trajectory_fitting(trajectory, num_control_points=20, target_output_points=None):
    """
    测试单条轨迹的拟合和重建
    
    Args:
        trajectory: (N, 3) 原始轨迹 [x, y, theta]
        num_control_points: 控制点数量
        target_output_points: 目标输出点数量，如果为None则与输入相同
        
    Returns:
        result: 包含拟合结果和误差的字典
    """
    N = len(trajectory)
    if target_output_points is None:
        target_output_points = N
    
    # 1. 拟合控制点（只使用位置信息 x, y）
    control_points_xy, basis_matrix_fit, knot_vector = fit_bspline_least_squares(
        trajectory[:, :2],  # 只拟合位置
        num_control_points=num_control_points,
        degree=3
    )
    
    # 2. 重建到目标点数（只重建位置）
    if target_output_points == N:
        # 使用预计算的基函数矩阵（更高效）
        traj_recon_xy = reconstruct_from_basis_matrix(control_points_xy, basis_matrix_fit)
    else:
        # 重新生成不同点数的轨迹
        traj_recon_xy = reconstruct_from_control_points(
            control_points_xy, 
            num_output_points=target_output_points,
            degree=3
        )
    
    # 3. 通过差分计算重建轨迹的角度
    traj_recon_angles = compute_trajectory_angle(traj_recon_xy)
    
    # 4. 组合位置和角度
    traj_recon = np.column_stack([traj_recon_xy, traj_recon_angles])
    
    # 5. 关键修正：使用弧长重采样对齐两条轨迹
    # 这样相同索引的点才对应相同的弧长位置
    print(f"      [弧长重采样] 对齐原始轨迹和重建轨迹...")
    
    # 先统一采样点数
    if target_output_points != N:
        # 使用弧长重采样原始轨迹
        trajectory_resampled = resample_by_arc_length(trajectory, target_output_points)
    else:
        trajectory_resampled = trajectory.copy()
    
    # 对两条轨迹都进行弧长重采样到相同点数
    # 这确保了第n个点在两条轨迹上都对应相同的弧长比例
    num_arc_samples = target_output_points
    traj_orig_arc = resample_by_arc_length(trajectory_resampled, num_arc_samples)
    traj_recon_arc = resample_by_arc_length(traj_recon, num_arc_samples)
    
    # 用于可视化和对比的轨迹
    trajectory_for_comparison = trajectory_resampled
    
    # 6. 计算位置误差（使用弧长对齐后的轨迹）
    errors = compute_reconstruction_errors(
        traj_orig_arc[:, :2],  # 位置
        traj_recon_arc[:, :2]
    )
    
    # 7. 计算角度误差（使用弧长对齐后的轨迹）
    angle_orig = traj_orig_arc[:, 2]
    angle_recon = traj_recon_arc[:, 2]
    angle_diff = np.arctan2(np.sin(angle_recon - angle_orig), 
                           np.cos(angle_recon - angle_orig))
    angle_error_deg = np.degrees(np.abs(angle_diff))
    
    # 添加更详细的角度误差统计
    errors['angle_error_from_bspline_mean'] = np.mean(angle_error_deg)
    errors['angle_error_from_bspline_max'] = np.max(angle_error_deg)
    errors['angle_error_from_bspline_median'] = np.median(angle_error_deg)
    errors['angle_error_from_bspline_std'] = np.std(angle_error_deg)
    errors['angle_error_from_bspline_diffs'] = angle_error_deg
    errors['pct_over_1deg'] = 100 * np.sum(angle_error_deg > 1) / len(angle_error_deg)
    errors['pct_over_5deg'] = 100 * np.sum(angle_error_deg > 5) / len(angle_error_deg)
    errors['pct_over_10deg'] = 100 * np.sum(angle_error_deg > 10) / len(angle_error_deg)
    
    # 验证：均值必须小于等于最大值
    assert errors['angle_error_from_bspline_mean'] <= errors['angle_error_from_bspline_max'], \
        f"BUG: 均值({errors['angle_error_from_bspline_mean']:.4f}) > 最大值({errors['angle_error_from_bspline_max']:.4f})"
    
    return {
        'trajectory_orig': trajectory,
        'trajectory_recon': traj_recon,
        'trajectory_for_comparison': trajectory_for_comparison,
        'traj_orig_arc': traj_orig_arc,  # 弧长对齐后的原始轨迹
        'traj_recon_arc': traj_recon_arc,  # 弧长对齐后的重建轨迹
        'control_points': control_points_xy,
        'num_control_points': num_control_points,
        'input_length': N,
        'output_length': target_output_points,
        'errors': errors
    }


def batch_test_trajectories(trajectories, metadata, num_control_points_list=[15, 20, 30]):
    """
    批量测试多条轨迹
    
    Args:
        trajectories: 轨迹列表
        metadata: 元数据列表
        num_control_points_list: 要测试的控制点数量列表
        
    Returns:
        results: 测试结果列表
    """
    results = []
    
    for traj, meta in zip(trajectories, metadata):
        print(f"\n{'='*70}")
        print(f"Testing trajectory: {meta['env_type']}/path_{meta['path_idx']}")
        print(f"Length: {meta['length']} points")
        print(f"{'='*70}")
        
        traj_results = {
            'metadata': meta,
            'tests': {}
        }
        
        for num_cp in num_control_points_list:
            print(f"\n  Testing with {num_cp} control points...")
            
            result = test_trajectory_fitting(
                traj,
                num_control_points=num_cp,
                target_output_points=None  # 重建到相同长度
            )
            
            err = result['errors']
            print(f"    位置误差(均值): {err['pos_error_mean']:.6f}")
            print(f"    位置误差(最大): {err['pos_error_max']:.6f}")
            print(f"    角度误差(均值): {err['angle_error_from_bspline_mean']:.4f}° (中位数: {err['angle_error_from_bspline_median']:.4f}°)")
            print(f"    角度误差(最大): {err['angle_error_from_bspline_max']:.4f}° (标准差: {err['angle_error_from_bspline_std']:.4f}°)")
            print(f"    >1°点比例: {err['pct_over_1deg']:.1f}%")
            print(f"    >5°点比例: {err['pct_over_5deg']:.1f}%")
            print(f"    >10°点比例: {err['pct_over_10deg']:.1f}%")
            
            traj_results['tests'][num_cp] = result
        
        results.append(traj_results)
    
    return results


# ==========================================
# 可视化
# ==========================================

def visualize_fitting_result(result, title='', save_path=None):
    """
    可视化单条轨迹的拟合结果（使用弧长对齐后的轨迹）
    """
    traj_orig = result['traj_orig_arc']  # 使用弧长对齐后的
    traj_recon = result['traj_recon_arc']  # 使用弧长对齐后的
    control_points = result['control_points']
    errors = result['errors']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 轨迹对比 (XY平面)
    ax = axes[0, 0]
    ax.plot(traj_orig[:, 0], traj_orig[:, 1], 'b-', linewidth=2, 
           label='Original', alpha=0.7)
    ax.plot(traj_recon[:, 0], traj_recon[:, 1], 'r--', linewidth=2, 
           label='Reconstructed', alpha=0.7)
    ax.scatter(control_points[:, 0], control_points[:, 1], 
              c='green', s=80, marker='x', linewidths=2.5, 
              label=f'Control Points (n={len(control_points)})', zorder=5)
    ax.scatter(traj_orig[0, 0], traj_orig[0, 1], c='purple', s=150, 
              marker='o', label='Start', zorder=6)
    ax.scatter(traj_orig[-1, 0], traj_orig[-1, 1], c='red', s=150, 
              marker='s', label='Goal', zorder=6)
    ax.set_title('Trajectory Comparison (XY Plane)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 2. 位置误差分布
    ax = axes[0, 1]
    sc = ax.scatter(traj_orig[:, 0], traj_orig[:, 1], 
                   c=errors['pos_errors'], cmap='hot', s=30, vmin=0)
    plt.colorbar(sc, ax=ax, label='Position Error')
    ax.set_title(f'Position Error Distribution (mean={errors["pos_error_mean"]:.4f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 3. 角度对比
    ax = axes[0, 2]
    indices = np.arange(len(traj_orig))
    ax.plot(indices, np.degrees(traj_orig[:, 2]), 'b-', 
           linewidth=2, label='Original Angle', alpha=0.7)
    ax.plot(indices, np.degrees(traj_recon[:, 2]), 'r--', 
           linewidth=2, label='Reconstructed Angle', alpha=0.7)
    ax.set_title('Angle Comparison')
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Angle (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 位置误差曲线
    ax = axes[1, 0]
    ax.plot(errors['pos_errors'], 'b-', linewidth=2)
    ax.axhline(y=errors['pos_error_mean'], color='r', linestyle='--', 
              label=f'Mean={errors["pos_error_mean"]:.4f}')
    ax.set_title('Position Error Curve')
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Position Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 角度误差曲线
    ax = axes[1, 1]
    angle_errors = errors['angle_error_from_bspline_diffs']
    ax.plot(angle_errors, 'r-', linewidth=2)
    ax.axhline(y=errors['angle_error_from_bspline_mean'], color='b', 
              linestyle='--', label=f'Mean={errors["angle_error_from_bspline_mean"]:.2f}°')
    ax.axhline(y=5.0, color='orange', linestyle=':', label='5° threshold')
    ax.axhline(y=10.0, color='red', linestyle=':', label='10° threshold')
    ax.set_title('Angle Error Curve')
    ax.set_xlabel('Point Index')
    ax.set_ylabel('Angle Error (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 角度误差直方图
    ax = axes[1, 2]
    ax.hist(angle_errors, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=5.0, color='orange', linestyle='--', label='5°')
    ax.axvline(x=10.0, color='red', linestyle='--', label='10°')
    ax.axvline(x=errors['angle_error_from_bspline_mean'], color='blue', 
              linestyle='--', label=f'Mean={errors["angle_error_from_bspline_mean"]:.2f}°')
    ax.set_xlabel('Angle Error (deg)')
    ax.set_ylabel('Frequency')
    ax.set_title('Angle Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def print_summary_table(results):
    """
    打印汇总表格
    """
    print("\n\n" + "="*130)
    print("Summary Table")
    print("="*130)
    print(f"{'Env/Path':<20} {'Traj Len':<10} {'Ctrl Pts':<10} {'Pos Err(mean)':<15} "
          f"{'Ang Err(mean)':<15} {'Ang Err(median)':<16} {'Ang Err(max)':<16} {'>1deg %':<10} {'>5deg %':<10} {'>10deg %':<10}")
    print("-"*130)
    
    for res in results:
        meta = res['metadata']
        env_path = f"{meta['env_type']}/path_{meta['path_idx']}"
        traj_len = meta['length']
        
        for num_cp, test_result in res['tests'].items():
            err = test_result['errors']
            print(f"{env_path:<20} {traj_len:<10} {num_cp:<10} "
                  f"{err['pos_error_mean']:<15.6f} "
                  f"{err['angle_error_from_bspline_mean']:<15.4f} "
                  f"{err['angle_error_from_bspline_median']:<16.4f} "
                  f"{err['angle_error_from_bspline_max']:<16.4f} "
                  f"{err['pct_over_1deg']:<10.1f} "
                  f"{err['pct_over_5deg']:<10.1f} "
                  f"{err['pct_over_10deg']:<10.1f}")


# ==========================================
# 主测试函数
# ==========================================

def main():
    """
    主测试函数
    
    使用方法：
    1. 默认：自动从 data/sim_dataset/val 读取数据
    2. 或者：修改下面的参数指定自己的数据路径
    """
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*10 + "B-Spline Fitting Test for Real Trajectories" + " "*16 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # ==================== 配置参数 ====================
    # 数据集路径（与vis_dit.py相同的格式）
    dataset_path = 'data/sim_dataset/train'
    
    # 环境类型
    env_type = 'env000010'
    
    # 路径索引（None表示加载所有）
    # path_indices = None  # 加载所有路径
    path_indices = [0, 1, 2, 3, 4, 5]  # 或指定特定路径
    
    # 最多加载的路径数量
    max_paths = 6
    
    # 控制点数量列表
    num_control_points_list = [8+2, 16+2, 24+2, 32+2]
    
    # 是否保存可视化图表
    save_visualizations = True
    save_dir = 'bspline_fitting_results'
    # =================================================
    
    # 检查数据集路径是否存在
    if not osp.exists(dataset_path):
        print(f"\n⚠️  Dataset path not found: {dataset_path}")
        print("\nPlease provide real trajectory data!")
        print("Data format should be pickle file containing dictionary:")
        print("  {'path': numpy.array([N, 3])}  # [x, y, theta]")
        print("\nExample directory structure:")
        print("  data/sim_dataset/val/")
        print("    env000008/")
        print("      path_0.p")
        print("      path_1.p")
        print("      ...")
        return
    
    # 1. 加载轨迹数据
    print(f"\n{'='*70}")
    print("Loading trajectory data...")
    print(f"{'='*70}")
    trajectories, metadata = load_trajectory_dataset(
        dataset_path, env_type, path_indices, max_paths
    )
    
    if len(trajectories) == 0:
        print("❌ No trajectory data found!")
        return
    
    # 打印轨迹统计信息
    lengths = [len(t) for t in trajectories]
    print(f"\nTrajectory Statistics:")
    print(f"  Total: {len(trajectories)}")
    print(f"  Length range: {min(lengths)} - {max(lengths)} points")
    print(f"  Average length: {np.mean(lengths):.1f} points")
    
    # 2. 批量测试
    print(f"\n{'='*70}")
    print("Starting batch testing...")
    print(f"{'='*70}")
    results = batch_test_trajectories(
        trajectories, metadata, num_control_points_list
    )
    
    # 3. 打印汇总表格
    print_summary_table(results)
    
    # 4. 可视化（选择性保存）
    if save_visualizations:
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n{'='*70}")
        print("Generating visualizations...")
        print(f"{'='*70}")
        
        for res in results:
            meta = res['metadata']
            env_path = f"{meta['env_type']}_path_{meta['path_idx']}"
            
            # 为每个控制点数量生成图表
            for num_cp in [8+2, 16+2, 24+2, 32+2]:  # 只可视化指定控制点数量的情况
                if num_cp in res['tests']:
                    title = f"{env_path} - {num_cp} Control Points (Length={meta['length']})"
                    save_path = osp.join(save_dir, f"{env_path}_{num_cp}cp.png")
                    visualize_fitting_result(
                        res['tests'][num_cp],
                        title=title,
                        save_path=save_path
                    )
    
    print("\n" + "="*70)
    print("✓ Testing completed!")
    print("="*70)


if __name__ == '__main__':
    main()
