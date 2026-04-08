#!/usr/bin/env python3
"""
FM-RE Cost Map 查询功能测试
===========================

测试 safe_oracle_from_trajectory 的 cost_map 查询功能。
"""

import torch
import sys
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_cost_map_oracle():
    """测试基于 cost_map 的 oracle"""
    print("="*60)
    print("测试: 基于 Cost Map 的 Safe Oracle")
    print("="*60)
    
    try:
        from fm_re_integration import safe_oracle_from_trajectory
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建模拟的 cost_map (D, H, W)
        D, H, W = 36, 100, 100  # (yaw, y, x)
        cost_map = torch.zeros(D, H, W, device=device)
        
        # 设置不同的成本区域
        # 低成本区域 (安全): cost ~ 0.2
        cost_map[:, 20:50, 20:50] = 0.2
        
        # 高成本区域 (不安全): cost ~ 0.8
        cost_map[:, 60:80, 60:80] = 0.8
        
        # 中等成本区域 (边界): cost ~ 0.5
        cost_map[:, 50:60, 50:60] = 0.5
        
        # 定义地图信息
        map_info = {
            'resolution': 0.4,
            'origin': (-20.0, -20.0, -np.pi),
            'size': (W, H, D)  # (W, H, D)
        }
        
        # 创建测试轨迹
        B = 16
        
        # 轨迹1: 在低成本区域 (应该安全)
        # 物理坐标映射：网格 [30, 30] -> 物理 [30*0.4 - 20, 30*0.4 - 20] = [-8, -8]
        traj_safe = torch.ones(B, 10, 2, device=device) * -8.0  # (B, 10, 2)
        
        # 轨迹2: 在高成本区域 (应该不安全)
        # 物理坐标映射：网格 [70, 70] -> 物理 [70*0.4 - 20, 70*0.4 - 20] = [8, 8]
        traj_unsafe = torch.ones(B, 10, 2, device=device) * 8.0  # (B, 10, 2)
        
        # 测试 1: 简单规则
        print("\n【测试 1】简单规则 (use_simple_oracle=True)")
        is_safe_1, score_1 = safe_oracle_from_trajectory(
            traj_safe,
            cost_map=None,
            map_info=None,
            stability_threshold=0.1,
            device=device,
            use_simple_oracle=True
        )
        print(f"  安全轨迹: 安全率 = {is_safe_1.float().mean():.1%}")
        
        # 测试 2: 基于 cost_map 查询（低成本区域）
        print("\n【测试 2】Cost Map 查询 - 低成本区域")
        is_safe_2, score_2 = safe_oracle_from_trajectory(
            traj_safe,
            cost_map=cost_map,
            map_info=map_info,
            stability_threshold=0.5,
            device=device,
            use_simple_oracle=False
        )
        print(f"  低成本轨迹:")
        print(f"    - 安全率: {is_safe_2.float().mean():.1%}")
        print(f"    - 平均安全评分: {score_2.mean():.4f}")
        print(f"    - 成本值范围: [{score_2.min():.4f}, {score_2.max():.4f}]")
        
        # 测试 3: 基于 cost_map 查询（高成本区域）
        print("\n【测试 3】Cost Map 查询 - 高成本区域")
        is_safe_3, score_3 = safe_oracle_from_trajectory(
            traj_unsafe,
            cost_map=cost_map,
            map_info=map_info,
            stability_threshold=0.5,
            device=device,
            use_simple_oracle=False
        )
        print(f"  高成本轨迹:")
        print(f"    - 安全率: {is_safe_3.float().mean():.1%}")
        print(f"    - 平均安全评分: {score_3.mean():.4f}")
        print(f"    - 成本值范围: [{score_3.min():.4f}, {score_3.max():.4f}]")
        
        # 验证结论
        print("\n【验证结果】")
        if is_safe_2.float().mean() > is_safe_3.float().mean():
            print("✓ Cost Map 查询正确：低成本区域被判为更安全")
        else:
            print("✗ Cost Map 查询异常：低成本区域不如高成本区域安全")
            return False
        
        # 测试 4: 混合轨迹 (一部分低成本，一部分高成本)
        print("\n【测试 4】混合轨迹")
        traj_mixed = torch.cat([
            torch.ones(B // 2, 5, 2, device=device) * -8.0,  # 低成本
            torch.ones(B // 2, 5, 2, device=device) * 8.0     # 高成本
        ], dim=1)  # (B, 10, 2)
        
        is_safe_4, score_4 = safe_oracle_from_trajectory(
            traj_mixed,
            cost_map=cost_map,
            map_info=map_info,
            stability_threshold=0.5,
            device=device,
            use_simple_oracle=False
        )
        print(f"  混合轨迹: 安全率 = {is_safe_4.float().mean():.1%}")
        print(f"  前一半（低成本）安全率: {is_safe_4[:B//2].float().mean():.1%}")
        print(f"  后一半（高成本）安全率: {is_safe_4[B//2:].float().mean():.1%}")
        
        print("\n✓ Cost Map 查询测试完成")
        return True
    
    except Exception as e:
        print(f"✗ Cost Map 查询测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cost_map_edge_cases():
    """测试 cost_map 的边界情况"""
    print("\n" + "="*60)
    print("测试: Cost Map 边界情况")
    print("="*60)
    
    try:
        from fm_re_integration import safe_oracle_from_trajectory
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建 cost_map
        D, H, W = 36, 100, 100
        cost_map = torch.rand(D, H, W, device=device) * 0.5 + 0.25  # 值在 [0.25, 0.75]
        
        map_info = {
            'resolution': 0.4,
            'origin': (-20.0, -20.0, -np.pi),
            'size': (W, H, D)
        }
        
        # 边界情况 1: 轨迹在范围边界
        print("\n【边界情况 1】轨迹在范围边界")
        traj_boundary = torch.ones(4, 5, 2, device=device)
        traj_boundary[0] = -20.0  # 最小值
        traj_boundary[1] = 20.0   # 最大值
        traj_boundary[2] = -19.9  # 接近最小值
        traj_boundary[3] = 19.9   # 接近最大值
        
        is_safe, score = safe_oracle_from_trajectory(
            traj_boundary,
            cost_map=cost_map,
            map_info=map_info,
            stability_threshold=0.5,
            device=device,
            use_simple_oracle=False
        )
        print(f"  边界轨迹安全率: {is_safe.float().mean():.1%}")
        
        # 边界情况 2: 单点轨迹
        print("\n【边界情况 2】单点轨迹")
        traj_single = torch.randn(4, 1, 2, device=device) * 10
        is_safe, score = safe_oracle_from_trajectory(
            traj_single,
            cost_map=cost_map,
            map_info=map_info,
            stability_threshold=0.5,
            device=device,
            use_simple_oracle=False
        )
        print(f"  单点轨迹处理成功，安全率: {is_safe.float().mean():.1%}")
        
        # 边界情况 3: 没有 cost_map 时自动回退
        print("\n【边界情况 3】缺失 cost_map 自动回退")
        traj = torch.randn(4, 10, 2, device=device) * 5
        is_safe, score = safe_oracle_from_trajectory(
            traj,
            cost_map=None,  # 故意传 None
            map_info=map_info,  # 即使有 map_info 也应该回退
            stability_threshold=0.5,
            device=device,
            use_simple_oracle=False  # 尝试使用 cost_map，但会自动回退
        )
        print(f"  自动回退到简单规则成功，安全率: {is_safe.float().mean():.1%}")
        
        print("\n✓ 边界情况测试完成")
        return True
    
    except Exception as e:
        print(f"✗ 边界情况测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  FM-RE Cost Map 查询功能测试".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    
    results = [
        test_cost_map_oracle(),
        test_cost_map_edge_cases(),
    ]
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n通过: {passed}/{total} 个测试")
    
    if passed == total:
        print("\n✓ 所有 Cost Map 功能测试通过！")
        print("\n现在可以运行 FM-RE 训练：")
        print("  python train_dit.py --stage 2 --resume /logs/stage1_best_model.pth")
        return 0
    else:
        print("\n✗ 部分测试失败。")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
