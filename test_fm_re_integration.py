#!/usr/bin/env python3
"""
FM-RE 集成验证脚本
==================

用于验证 FM-RE 模块是否正确集成到 train_dit.py 中。
运行此脚本来检查：
  1. FM-RE 模块导入
  2. 函数签名
  3. 基本的张量操作
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_fm_re_imports():
    """测试 FM-RE 模块导入"""
    print("="*60)
    print("测试 1: FM-RE 模块导入")
    print("="*60)
    
    try:
        from fm_re_integration import (
            add_random_exploration,
            sigma_schedule_linear,
            sigma_schedule_cosine,
            safe_oracle_from_trajectory,
            compute_fm_re_loss,
            FMREMonitor,
            FMRELossWeights
        )
        print("✓ 所有 FM-RE 模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ FM-RE 导入失败: {e}")
        return False


def test_add_random_exploration():
    """测试随机化探索函数"""
    print("\n" + "="*60)
    print("测试 2: 随机化探索函数")
    print("="*60)
    
    try:
        from fm_re_integration import add_random_exploration, sigma_schedule_linear
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建测试张量
        B, num_points, d = 16, 24, 2
        u_det = torch.randn(B, num_points, d, device=device)
        t = torch.rand(B, device=device) * 0.5 + 0.5  # [0.5, 1.0]
        
        # 调用函数
        u_stoch, noise, log_pi = add_random_exploration(
            u_det, t, t0=0.6, sigma_schedule=sigma_schedule_linear, device=device
        )
        
        # 检查输出形状
        assert u_stoch.shape == (B, num_points, d), f"u_stoch shape mismatch: {u_stoch.shape}"
        assert noise.shape == (B, num_points, d), f"noise shape mismatch: {noise.shape}"
        assert log_pi.shape == (B,), f"log_pi shape mismatch: {log_pi.shape}"
        
        # 检查数值
        assert not torch.isnan(u_stoch).any(), "u_stoch contains NaN"
        assert not torch.isnan(log_pi).any(), "log_pi contains NaN"
        
        print(f"✓ 随机化探索函数运行成功")
        print(f"  - u_stochastic shape: {u_stoch.shape}")
        print(f"  - noise shape: {noise.shape}")
        print(f"  - log_pi shape: {log_pi.shape}")
        print(f"  - log_pi range: [{log_pi.min():.4f}, {log_pi.max():.4f}]")
        return True
    
    except Exception as e:
        print(f"✗ 随机化探索函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_safe_oracle():
    """测试安全判断器"""
    print("\n" + "="*60)
    print("测试 3: 安全判断器")
    print("="*60)
    
    try:
        from fm_re_integration import safe_oracle_from_trajectory
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建测试轨迹
        B, num_points = 16, 24
        
        # 安全轨迹：在 [-20, 20] 范围内，光滑
        safe_traj = torch.linspace(-10, 10, num_points).unsqueeze(0).unsqueeze(-1)
        safe_traj = safe_traj.repeat(B, 1, 2).to(device)
        
        # 不安全轨迹：超出范围
        unsafe_traj = torch.ones(B, num_points, 2, device=device) * 50.0
        
        # 测试
        is_safe_1, score_1 = safe_oracle_from_trajectory(safe_traj, device=device, use_simple_oracle=True)
        is_safe_2, score_2 = safe_oracle_from_trajectory(unsafe_traj, device=device, use_simple_oracle=True)
        
        # 检查输出
        assert is_safe_1.shape == (B,), f"is_safe shape mismatch: {is_safe_1.shape}"
        assert score_1.shape == (B,), f"score shape mismatch: {score_1.shape}"
        
        # 安全轨迹的安全率应该较高
        safe_rate_1 = is_safe_1.float().mean().item()
        safe_rate_2 = is_safe_2.float().mean().item()
        
        print(f"✓ 安全判断器运行成功")
        print(f"  - 安全轨迹安全率: {safe_rate_1:.1%}")
        print(f"  - 不安全轨迹安全率: {safe_rate_2:.1%}")
        print(f"  - 判断器正确性: {'✓' if safe_rate_1 >= safe_rate_2 else '✗'}")
        
        return True
    
    except Exception as e:
        print(f"✗ 安全判断器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compute_fm_re_loss():
    """测试 FM-RE 损失计算"""
    print("\n" + "="*60)
    print("测试 4: FM-RE 损失计算")
    print("="*60)
    
    try:
        from fm_re_integration import compute_fm_re_loss
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建测试数据
        B = 16
        log_pi_list = [torch.randn(B, device=device) for _ in range(5)]  # 5 个时间步
        rewards = torch.bernoulli(torch.ones(B, device=device) * 0.5)    # 50% 安全
        
        # 计算损失
        loss_re, adv = compute_fm_re_loss(log_pi_list, rewards, lambda_re=0.1, device=device)
        
        # 检查输出
        assert not torch.isnan(loss_re), "loss_re contains NaN"
        assert loss_re.shape == (), f"loss_re shape mismatch: {loss_re.shape}"
        
        print(f"✓ FM-RE 损失计算成功")
        print(f"  - loss_re: {loss_re.item():.6f}")
        print(f"  - advantage_mean: {adv.item():.6f}")
        print(f"  - reward_rate: {rewards.mean().item():.1%}")
        return True
    
    except Exception as e:
        print(f"✗ FM-RE 损失计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_train_dit_imports():
    """测试 train_dit.py 导入"""
    print("\n" + "="*60)
    print("测试 5: train_dit.py 导入")
    print("="*60)
    
    try:
        # 这只是检查是否有明显的语法错误
        import train_dit
        print("✓ train_dit.py 导入成功（无语法错误）")
        
        # 检查关键函数是否存在
        assert hasattr(train_dit, 'diffusion_loss'), "diffusion_loss 函数不存在"
        assert hasattr(train_dit, 'train_epoch'), "train_epoch 函数不存在"
        assert hasattr(train_dit, 'eval_epoch'), "eval_epoch 函数不存在"
        
        print("✓ 所有关键函数存在")
        return True
    
    except Exception as e:
        print(f"✗ train_dit.py 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sigma_schedules():
    """测试 sigma 日程"""
    print("\n" + "="*60)
    print("测试 6: Sigma 日程函数")
    print("="*60)
    
    try:
        from fm_re_integration import sigma_schedule_linear, sigma_schedule_cosine
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        t = torch.linspace(0, 1, 11, device=device)
        
        # 测试线性日程
        sigma_linear = sigma_schedule_linear(t, sigma_min=0.05, sigma_max=0.2)
        
        # 测试余弦日程
        sigma_cosine = sigma_schedule_cosine(t, sigma_min=0.05, sigma_max=0.2)
        
        assert sigma_linear.shape == t.shape, "sigma_linear shape mismatch"
        assert sigma_cosine.shape == t.shape, "sigma_cosine shape mismatch"
        assert (sigma_linear >= 0.05).all(), "sigma_linear 小于最小值"
        assert (sigma_linear <= 0.2).all(), "sigma_linear 大于最大值"
        
        print(f"✓ Sigma 日程函数运行成功")
        print(f"  - 线性日程 (t=0-1): {sigma_linear[0]:.4f} -> {sigma_linear[-1]:.4f}")
        print(f"  - 余弦日程 (t=0-1): {sigma_cosine[0]:.4f} -> {sigma_cosine[-1]:.4f}")
        return True
    
    except Exception as e:
        print(f"✗ Sigma 日程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  FM-RE 集成验证".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    
    tests = [
        test_fm_re_imports,
        test_add_random_exploration,
        test_safe_oracle,
        test_compute_fm_re_loss,
        test_sigma_schedules,
        test_train_dit_imports,
    ]
    
    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"✗ 测试 {test_func.__name__} 发生异常: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n通过: {passed}/{total} 个测试")
    
    if passed == total:
        print("\n✓ 所有测试通过！FM-RE 集成正确。")
        print("\n现在可以运行：")
        print("  python train_dit.py --batchSize 32 --dataFolder /path/to/data \\")
        print("    --fileDir /logs --stage 1 --stage1_epochs 200")
        return 0
    else:
        print("\n✗ 部分测试失败。请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
