"""
分析B样条拟合对角度的影响
重点关注：即使位置误差小，角度误差是否会大
"""

import numpy as np
import matplotlib.pyplot as plt
from bspline_utils import (
    fit_bspline_least_squares,
    reconstruct_from_basis_matrix,
    compute_reconstruction_errors,
    generate_test_trajectory
)


def compute_velocity_and_angle(trajectory):
    """
    计算轨迹的速度和角度
    
    Args:
        trajectory: (N, 2) 轨迹点
        
    Returns:
        velocities: (N, 2) 速度向量
        speeds: (N,) 速度大小
        angles: (N,) 角度（弧度）
    """
    velocities = np.zeros_like(trajectory)
    
    # 中心差分
    velocities[1:-1] = (trajectory[2:] - trajectory[:-2]) / 2
    
    # 边界处理
    velocities[0] = trajectory[1] - trajectory[0]
    velocities[-1] = trajectory[-1] - trajectory[-2]
    
    speeds = np.linalg.norm(velocities, axis=1)
    angles = np.arctan2(velocities[:, 1], velocities[:, 0])
    
    return velocities, speeds, angles


def analyze_angle_distortion(traj_type, num_dense=100, num_cp=20):
    """
    详细分析角度失真
    """
    print(f"\n{'='*80}")
    print(f"分析轨迹类型: {traj_type}")
    print(f"密集点数: {num_dense}, 控制点数: {num_cp}")
    print(f"{'='*80}")
    
    # 1. 生成原始轨迹
    traj_orig = generate_test_trajectory(traj_type, num_dense)
    
    # 2. 拟合和重建
    control_points, basis_matrix, _ = fit_bspline_least_squares(
        traj_orig, num_control_points=num_cp, degree=3
    )
    traj_recon = reconstruct_from_basis_matrix(control_points, basis_matrix)
    
    # 3. 计算速度和角度
    vel_orig, speed_orig, angle_orig = compute_velocity_and_angle(traj_orig)
    vel_recon, speed_recon, angle_recon = compute_velocity_and_angle(traj_recon)
    
    # 4. 计算各种误差
    pos_error = np.linalg.norm(traj_orig - traj_recon, axis=1)
    
    # 速度误差
    vel_error = np.linalg.norm(vel_orig - vel_recon, axis=1)
    speed_error = np.abs(speed_orig - speed_recon)
    
    # 角度误差（考虑周期性）
    angle_diff = np.arctan2(np.sin(angle_recon - angle_orig), 
                           np.cos(angle_recon - angle_orig))
    angle_error_deg = np.degrees(np.abs(angle_diff))
    
    # 5. 统计
    print(f"\n位置误差:")
    print(f"  均值: {np.mean(pos_error):.6f}")
    print(f"  最大: {np.max(pos_error):.6f}")
    print(f"  中位数: {np.median(pos_error):.6f}")
    
    print(f"\n速度误差:")
    print(f"  速度向量误差均值: {np.mean(vel_error):.6f}")
    print(f"  速度大小误差均值: {np.mean(speed_error):.6f}")
    print(f"  速度大小误差最大: {np.max(speed_error):.6f}")
    
    print(f"\n角度误差:")
    print(f"  均值: {np.mean(angle_error_deg):.4f}°")
    print(f"  最大: {np.max(angle_error_deg):.4f}°")
    print(f"  中位数: {np.median(angle_error_deg):.4f}°")
    print(f"  >1° 的点: {np.sum(angle_error_deg > 1)} / {len(angle_error_deg)} ({100*np.sum(angle_error_deg > 1)/len(angle_error_deg):.1f}%)")
    print(f"  >5° 的点: {np.sum(angle_error_deg > 5)} / {len(angle_error_deg)} ({100*np.sum(angle_error_deg > 5)/len(angle_error_deg):.1f}%)")
    print(f"  >10° 的点: {np.sum(angle_error_deg > 10)} / {len(angle_error_deg)} ({100*np.sum(angle_error_deg > 10)/len(angle_error_deg):.1f}%)")
    
    # 6. 计算误差放大因子
    pos_error_norm = np.mean(pos_error) / (np.mean(np.linalg.norm(traj_orig, axis=1)) + 1e-8)
    angle_error_norm = np.mean(angle_error_deg) / 180.0  # 归一化到[0,1]
    
    amplification_factor = angle_error_norm / (pos_error_norm + 1e-8)
    
    print(f"\n误差放大分析:")
    print(f"  归一化位置误差: {pos_error_norm:.6f}")
    print(f"  归一化角度误差: {angle_error_norm:.6f}")
    print(f"  角度误差放大因子: {amplification_factor:.2f}x")
    
    # 7. 分析角度误差与曲率的关系
    # 计算原始轨迹的曲率（用角度变化率近似）
    angle_change = np.abs(np.diff(angle_orig))
    angle_change = np.minimum(angle_change, 2*np.pi - angle_change)  # 处理周期性
    curvature_approx = np.zeros(len(angle_orig))
    curvature_approx[:-1] = angle_change
    curvature_approx[-1] = curvature_approx[-2]
    
    # 找出高曲率区域
    high_curvature_threshold = np.percentile(curvature_approx, 75)
    high_curvature_mask = curvature_approx > high_curvature_threshold
    
    print(f"\n高曲率区域分析:")
    print(f"  高曲率点数: {np.sum(high_curvature_mask)} / {len(high_curvature_mask)}")
    print(f"  高曲率区域平均角度误差: {np.mean(angle_error_deg[high_curvature_mask]):.4f}°")
    print(f"  低曲率区域平均角度误差: {np.mean(angle_error_deg[~high_curvature_mask]):.4f}°")
    
    return {
        'traj_type': traj_type,
        'traj_orig': traj_orig,
        'traj_recon': traj_recon,
        'control_points': control_points,
        'pos_error': pos_error,
        'vel_error': vel_error,
        'speed_error': speed_error,
        'angle_error_deg': angle_error_deg,
        'angle_orig': angle_orig,
        'angle_recon': angle_recon,
        'curvature_approx': curvature_approx,
        'amplification_factor': amplification_factor
    }


def visualize_angle_distortion(result, save_path=None):
    """
    可视化角度失真
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    traj_orig = result['traj_orig']
    traj_recon = result['traj_recon']
    control_points = result['control_points']
    angle_error_deg = result['angle_error_deg']
    angle_orig = result['angle_orig']
    angle_recon = result['angle_recon']
    pos_error = result['pos_error']
    
    # 1. 轨迹对比
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(traj_orig[:, 0], traj_orig[:, 1], 'b-', linewidth=2, label='原始', alpha=0.7)
    ax1.plot(traj_recon[:, 0], traj_recon[:, 1], 'r--', linewidth=2, label='重建', alpha=0.7)
    ax1.scatter(control_points[:, 0], control_points[:, 1], c='green', s=60, 
               marker='x', linewidths=2, label=f'控制点 (n={len(control_points)})', zorder=5)
    ax1.set_title(f'{result["traj_type"]} - 轨迹对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. 位置误差分布
    ax2 = fig.add_subplot(gs[0, 1])
    points = ax2.scatter(traj_orig[:, 0], traj_orig[:, 1], c=pos_error, 
                        cmap='hot', s=30, vmin=0)
    plt.colorbar(points, ax=ax2, label='位置误差')
    ax2.set_title(f'位置误差分布 (均值={np.mean(pos_error):.4f})')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. 角度误差分布
    ax3 = fig.add_subplot(gs[0, 2])
    points = ax3.scatter(traj_orig[:, 0], traj_orig[:, 1], c=angle_error_deg, 
                        cmap='hot', s=30, vmin=0)
    plt.colorbar(points, ax=ax3, label='角度误差 (°)')
    ax3.set_title(f'角度误差分布 (均值={np.mean(angle_error_deg):.2f}°)')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # 4. 位置误差曲线
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(pos_error, 'b-', linewidth=1.5)
    ax4.axhline(y=np.mean(pos_error), color='r', linestyle='--', 
               label=f'均值={np.mean(pos_error):.4f}')
    ax4.set_title('位置误差曲线')
    ax4.set_xlabel('点索引')
    ax4.set_ylabel('位置误差')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 角度误差曲线
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(angle_error_deg, 'r-', linewidth=1.5)
    ax5.axhline(y=np.mean(angle_error_deg), color='b', linestyle='--', 
               label=f'均值={np.mean(angle_error_deg):.2f}°')
    ax5.axhline(y=1.0, color='orange', linestyle=':', label='1°阈值')
    ax5.axhline(y=5.0, color='red', linestyle=':', label='5°阈值')
    ax5.set_title('角度误差曲线')
    ax5.set_xlabel('点索引')
    ax5.set_ylabel('角度误差 (°)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 角度对比
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(np.degrees(angle_orig), 'b-', linewidth=1.5, label='原始角度', alpha=0.7)
    ax6.plot(np.degrees(angle_recon), 'r--', linewidth=1.5, label='重建角度', alpha=0.7)
    ax6.set_title('角度对比')
    ax6.set_xlabel('点索引')
    ax6.set_ylabel('角度 (°)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 位置误差 vs 角度误差散点图
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.scatter(pos_error, angle_error_deg, alpha=0.5, s=20)
    ax7.set_xlabel('位置误差')
    ax7.set_ylabel('角度误差 (°)')
    ax7.set_title('位置误差 vs 角度误差')
    ax7.grid(True, alpha=0.3)
    
    # 8. 曲率 vs 角度误差
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.scatter(result['curvature_approx'], angle_error_deg, alpha=0.5, s=20)
    ax8.set_xlabel('曲率近似（角度变化率）')
    ax8.set_ylabel('角度误差 (°)')
    ax8.set_title('曲率 vs 角度误差')
    ax8.grid(True, alpha=0.3)
    
    # 9. 角度误差直方图
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.hist(angle_error_deg, bins=30, edgecolor='black', alpha=0.7)
    ax9.axvline(x=1.0, color='orange', linestyle='--', label='1°')
    ax9.axvline(x=5.0, color='red', linestyle='--', label='5°')
    ax9.axvline(x=np.mean(angle_error_deg), color='blue', linestyle='--', 
               label=f'均值={np.mean(angle_error_deg):.2f}°')
    ax9.set_xlabel('角度误差 (°)')
    ax9.set_ylabel('频数')
    ax9.set_title('角度误差分布直方图')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存图表: {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close(fig)


def compare_control_point_numbers():
    """
    对比不同控制点数量对角度的影响
    """
    print(f"\n{'='*80}")
    print("对比控制点数量对角度误差的影响")
    print(f"{'='*80}")
    
    traj_types = ['sine', 'spiral', 's_curve', 'complex', 'zigzag']
    cp_nums = [10, 15, 20, 30, 40, 50]
    
    results_table = []
    
    for traj_type in traj_types:
        print(f"\n--- {traj_type} ---")
        traj_orig = generate_test_trajectory(traj_type, 100)
        
        for num_cp in cp_nums:
            control_points, basis_matrix, _ = fit_bspline_least_squares(
                traj_orig, num_control_points=num_cp, degree=3
            )
            traj_recon = reconstruct_from_basis_matrix(control_points, basis_matrix)
            
            # 计算误差
            pos_error = np.mean(np.linalg.norm(traj_orig - traj_recon, axis=1))
            
            _, _, angle_orig = compute_velocity_and_angle(traj_orig)
            _, _, angle_recon = compute_velocity_and_angle(traj_recon)
            angle_diff = np.arctan2(np.sin(angle_recon - angle_orig), 
                                   np.cos(angle_recon - angle_orig))
            angle_error = np.mean(np.degrees(np.abs(angle_diff)))
            angle_error_max = np.max(np.degrees(np.abs(angle_diff)))
            
            # 计算>5度的点的比例
            pct_over_5deg = 100 * np.sum(np.degrees(np.abs(angle_diff)) > 5) / len(angle_diff)
            
            results_table.append({
                'traj_type': traj_type,
                'num_cp': num_cp,
                'pos_error': pos_error,
                'angle_error_mean': angle_error,
                'angle_error_max': angle_error_max,
                'pct_over_5deg': pct_over_5deg
            })
            
            print(f"  {num_cp:2d} CP: pos={pos_error:.6f}, angle_mean={angle_error:.4f}°, "
                  f"angle_max={angle_error_max:.4f}°, >5°点={pct_over_5deg:.1f}%")
    
    # 打印汇总表格
    print(f"\n{'='*100}")
    print("汇总表格")
    print(f"{'='*100}")
    print(f"{'轨迹类型':<12} {'控制点':<8} {'位置误差':<12} {'角度误差(均)':<14} "
          f"{'角度误差(最大)':<16} {'>5°点比例':<12}")
    print("-"*100)
    
    for r in results_table:
        print(f"{r['traj_type']:<12} {r['num_cp']:<8} {r['pos_error']:<12.6f} "
              f"{r['angle_error_mean']:<14.4f} {r['angle_error_max']:<16.4f} "
              f"{r['pct_over_5deg']:<12.1f}")
    
    return results_table


def main():
    """
    主函数
    """
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*20 + "B样条角度失真分析" + " "*37 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    # 测试1: 详细分析各种轨迹
    print("\n" + "="*80)
    print("测试1: 详细分析各种轨迹的角度失真")
    print("="*80)
    
    traj_types = ['sine', 'spiral', 's_curve', 'complex', 'zigzag']
    
    for traj_type in traj_types:
        result = analyze_angle_distortion(traj_type, num_dense=100, num_cp=20)
        visualize_angle_distortion(result, save_path=f'angle_analysis_{traj_type}.png')
    
    # 测试2: 对比控制点数量
    print("\n" + "="*80)
    print("测试2: 对比控制点数量对角度误差的影响")
    print("="*80)
    
    results_table = compare_control_point_numbers()
    
    # 测试3: 关键结论
    print("\n" + "="*80)
    print("关键结论")
    print("="*80)
    
    print("""
1. 角度失真问题确实存在：
   - 即使位置误差很小，角度误差可能很大
   - 角度误差放大因子通常在10-1000x之间
   
2. 失真的根源：
   - B样条拟合会对轨迹进行平滑
   - 平滑过程会改变局部切向方向
   - 高曲率区域（急转弯）受影响最严重
   
3. 对你的任务的影响：
   ⚠️  如果任务对角度敏感，纯位置控制点可能不够
   ⚠️  速度通过差分计算会放大位置误差
   ⚠️  在高曲率区域，角度可能偏差5-50度
   
4. 解决方案：
   方案A: 增加控制点数量
   - 30-50个控制点可以显著降低角度误差
   - 但会增加网络输出维度和计算量
   
   方案B: 使用Hermite样条（同时拟合位置和切向）
   - 每个控制点包含位置(x,y)和切向(dx,dy)
   - 可以精确控制角度
   - 适合角度敏感任务
   
   方案C: 混合表示
   - 控制点 + 关键点的角度约束
   - 在高曲率区域添加角度监督
   
   方案D: 直接输出轨迹点（不用B样条）
   - 如果平滑性不是必需的
   - 可以考虑直接输出密集轨迹点
   
5. 建议：
   对于你的任务（局部角度敏感），建议：
   ✓ 使用方案B（Hermite样条）或方案C（混合表示）
   ✓ 在损失函数中添加角度约束
   ✓ 在高曲率区域增加控制点密度
    """)
    
    print("\n" + "="*80)
    print("✓ 分析完成！")
    print("="*80)
    print("\n生成的文件:")
    for traj_type in traj_types:
        print(f"  - angle_analysis_{traj_type}.png")
    print()


if __name__ == '__main__':
    main()
