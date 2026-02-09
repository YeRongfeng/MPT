"""
B样条拟合测试脚本
测试从密集轨迹拟合控制点，再从控制点重建轨迹的完整流程
"""

import numpy as np
import matplotlib.pyplot as plt
from bspline_utils import (
    fit_bspline_least_squares,
    reconstruct_from_control_points,
    reconstruct_from_basis_matrix,
    compute_reconstruction_errors,
    generate_test_trajectory
)


def test_single_trajectory(traj_type, num_dense_points=100, num_control_points=20):
    """
    测试单条轨迹的拟合和重建
    
    Args:
        traj_type: 轨迹类型
        num_dense_points: 密集轨迹点数量
        num_control_points: 控制点数量
        
    Returns:
        results: 包含误差信息的字典
    """
    print(f"\n{'='*70}")
    print(f"测试轨迹: {traj_type}")
    print(f"密集点数: {num_dense_points}, 控制点数: {num_control_points}")
    print(f"{'='*70}")
    
    # 1. 生成密集轨迹
    traj_dense = generate_test_trajectory(traj_type, num_dense_points)
    print(f"✓ 生成密集轨迹: {traj_dense.shape}")
    
    # 2. 拟合控制点
    control_points, basis_matrix, knot_vector = fit_bspline_least_squares(
        traj_dense, 
        num_control_points=num_control_points, 
        degree=3
    )
    print(f"✓ 拟合控制点: {control_points.shape}")
    
    # 3. 从控制点重建密集轨迹（使用预计算的基函数矩阵）
    traj_recon = reconstruct_from_basis_matrix(control_points, basis_matrix)
    print(f"✓ 重建轨迹: {traj_recon.shape}")
    
    # 4. 计算误差
    errors = compute_reconstruction_errors(traj_dense, traj_recon)
    
    print(f"\n误差统计:")
    print(f"  位置误差 (平均): {errors['pos_error_mean']:.6f}")
    print(f"  位置误差 (最大): {errors['pos_error_max']:.6f}")
    print(f"  位置误差 (标准差): {errors['pos_error_std']:.6f}")
    print(f"  角度误差 (平均): {errors['angle_error_mean']:.4f}°")
    print(f"  角度误差 (最大): {errors['angle_error_max']:.4f}°")
    
    return {
        'traj_type': traj_type,
        'traj_dense': traj_dense,
        'control_points': control_points,
        'traj_recon': traj_recon,
        'errors': errors
    }


def test_multiple_trajectories():
    """
    测试多种轨迹类型
    """
    print("\n" + "="*70)
    print("多轨迹类型测试")
    print("="*70)
    
    # 测试配置
    traj_types = ['sine', 'spiral', 's_curve', 'complex', 'circle', 'zigzag']
    num_dense_points = 100
    num_control_points_list = [15, 20, 30]
    
    results = {}
    
    for num_cp in num_control_points_list:
        print(f"\n\n{'#'*70}")
        print(f"控制点数量: {num_cp}")
        print(f"{'#'*70}")
        
        results[num_cp] = {}
        
        for traj_type in traj_types:
            result = test_single_trajectory(traj_type, num_dense_points, num_cp)
            results[num_cp][traj_type] = result
    
    return results


def print_summary_table(results):
    """
    打印汇总表格
    """
    print("\n\n" + "="*90)
    print("汇总表格")
    print("="*90)
    print(f"{'控制点数':<10} {'轨迹类型':<12} {'位置误差(均值)':<15} {'位置误差(最大)':<15} {'角度误差(均值)':<15}")
    print("-"*90)
    
    for num_cp in sorted(results.keys()):
        for traj_type in results[num_cp].keys():
            err = results[num_cp][traj_type]['errors']
            print(f"{num_cp:<10} {traj_type:<12} {err['pos_error_mean']:<15.6f} "
                  f"{err['pos_error_max']:<15.6f} {err['angle_error_mean']:<15.4f}")


def visualize_results(results, num_cp=20):
    """
    可视化结果
    
    Args:
        results: 测试结果字典
        num_cp: 要可视化的控制点数量
    """
    print(f"\n生成可视化图表（控制点数={num_cp}）...")
    
    result_dict = results[num_cp]
    traj_types = list(result_dict.keys())
    n_trajs = len(traj_types)
    
    # 创建大图
    fig, axes = plt.subplots(2, n_trajs, figsize=(4*n_trajs, 8))
    
    if n_trajs == 1:
        axes = axes.reshape(-1, 1)
    
    for i, traj_type in enumerate(traj_types):
        result = result_dict[traj_type]
        traj_dense = result['traj_dense']
        traj_recon = result['traj_recon']
        control_points = result['control_points']
        errors = result['errors']
        
        # 上图：轨迹对比
        ax = axes[0, i]
        ax.plot(traj_dense[:, 0], traj_dense[:, 1], 'b-', 
               linewidth=2, label='原始轨迹', alpha=0.7)
        ax.plot(traj_recon[:, 0], traj_recon[:, 1], 'r--', 
               linewidth=2, label='重建轨迹', alpha=0.7)
        ax.scatter(control_points[:, 0], control_points[:, 1], 
                  c='green', s=80, marker='x', linewidths=2.5, 
                  label=f'控制点 (n={len(control_points)})', zorder=5)
        ax.set_title(f'{traj_type}\n位置误差: {errors["pos_error_mean"]:.4f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 下图：误差曲线
        ax = axes[1, i]
        ax.plot(errors['pos_errors'], 'b-', linewidth=2, label='位置误差')
        ax.axhline(y=errors['pos_error_mean'], color='r', linestyle='--', 
                  linewidth=1.5, label=f"均值: {errors['pos_error_mean']:.4f}")
        ax.set_title(f'位置误差分布', fontsize=10)
        ax.set_xlabel('点索引')
        ax.set_ylabel('误差')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'bspline_test_results_{num_cp}cp.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ 保存图表: {filename}")
    plt.close(fig)


def visualize_control_point_comparison():
    """
    对比不同控制点数量的效果
    """
    print(f"\n生成控制点数量对比图表...")
    
    traj_type = 'spiral'
    num_dense = 100
    cp_nums = [10, 15, 20, 30, 40]
    
    fig, axes = plt.subplots(1, len(cp_nums), figsize=(5*len(cp_nums), 5))
    
    for i, num_cp in enumerate(cp_nums):
        # 生成和拟合
        traj_dense = generate_test_trajectory(traj_type, num_dense)
        control_points, basis_matrix, _ = fit_bspline_least_squares(
            traj_dense, num_control_points=num_cp, degree=3
        )
        traj_recon = reconstruct_from_basis_matrix(control_points, basis_matrix)
        errors = compute_reconstruction_errors(traj_dense, traj_recon)
        
        # 绘图
        ax = axes[i]
        ax.plot(traj_dense[:, 0], traj_dense[:, 1], 'b-', 
               linewidth=2, label='原始', alpha=0.7)
        ax.plot(traj_recon[:, 0], traj_recon[:, 1], 'r--', 
               linewidth=2, label='重建', alpha=0.7)
        ax.scatter(control_points[:, 0], control_points[:, 1], 
                  c='green', s=80, marker='x', linewidths=2.5, 
                  label=f'{num_cp} 控制点', zorder=5)
        ax.set_title(f'{num_cp} 控制点\n误差: {errors["pos_error_mean"]:.4f}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    filename = 'bspline_control_point_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ 保存对比图: {filename}")
    plt.close(fig)


def main():
    """
    主测试函数
    """
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*20 + "B样条拟合测试程序" + " "*28 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # 测试1: 多轨迹测试
    results = test_multiple_trajectories()
    
    # 打印汇总表格
    print_summary_table(results)
    
    # 测试2: 可视化（选择20个控制点）
    if 20 in results:
        visualize_results(results, num_cp=20)
    
    # 测试3: 控制点数量对比
    visualize_control_point_comparison()
    
    print("\n" + "="*70)
    print("✓ 所有测试完成！")
    print("="*70)
    print("\n生成的文件:")
    print("  - bspline_test_results_20cp.png (多轨迹测试)")
    print("  - bspline_control_point_comparison.png (控制点数量对比)")
    print()


if __name__ == '__main__':
    main()
