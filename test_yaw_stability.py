#!/usr/bin/env python3
"""
测试yaw_stability功能的脚本
"""

import numpy as np
import torch
from dataLoader_uneven import UnevenPathDataLoader, compute_map_yaw_bins
import matplotlib.pyplot as plt
import os

def test_yaw_stability_computation():
    """测试yaw稳定性计算功能"""
    print("测试yaw稳定性计算功能...")
    
    # 从数据加载器获取真实地图数据
    data_folder = '/home/yrf/MPT/data/terrain_test/val'
    if not os.path.exists(data_folder):
        print("数据文件夹不存在，跳过测试")
        return False
    
    # 查找可用的环境
    env_list = []
    for env_name in ['env000001']:
        env_path = os.path.join(data_folder, env_name)
        if os.path.exists(env_path):
            env_list.append(env_name)
    
    if not env_list:
        print("没有找到可用的数据环境，跳过测试")
        return False
    
    try:
        # 创建数据加载器并获取真实地图数据
        dataset = UnevenPathDataLoader(env_list, data_folder)
        if len(dataset) == 0:
            print("数据集为空，跳过测试")
            return False
        
        # 获取第一个样本的地图数据
        sample = dataset[0]
        map_input = sample['map']  # [3, H, W] - [normal_x, normal_y, normal_z]
        
        # 提取法向量分量
        normal_x_torch = map_input[0]  # [H, W]
        normal_y_torch = map_input[1]  # [H, W]
        normal_z_torch = map_input[2]  # [H, W]
        
        H, W = normal_x_torch.shape
        print(f"从数据加载器获取的真实地图数据，大小: {H}x{W}")
        
        # 测试性能
        import time
    
        
        # PyTorch版本计时
        start_time = time.time()
        yaw_stability = compute_map_yaw_bins(normal_x_torch, normal_y_torch, normal_z_torch, yaw_bins=36)
        torch_time = time.time() - start_time
        
        print(f"yaw_stability形状: {yaw_stability.shape}")
        print(f"稳定角度比例范围: {yaw_stability.min():.3f} - {yaw_stability.max():.3f}")
        print(f"PyTorch版本计算时间: {torch_time:.4f}秒")
        
        # 计算每个点的稳定角度数量
        stable_counts = torch.sum(yaw_stability, dim=2)
        print(f"每个点稳定角度数量范围: {stable_counts.min()} - {stable_counts.max()}")
        
        # 转换回numpy用于可视化
        normal_x_np = normal_x_torch.cpu().numpy()
        normal_y_np = normal_y_torch.cpu().numpy()
        normal_z_np = normal_z_torch.cpu().numpy()
        yaw_stability_np = yaw_stability.cpu().numpy()
        stable_counts_np = stable_counts.cpu().numpy()
        
        # 可视化结果
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 法向量可视化
        im1 = axes[0, 0].imshow(normal_x_np, cmap='RdBu', vmin=-1, vmax=1, origin='lower')
        axes[0, 0].set_title('Normal X')
        plt.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(normal_y_np, cmap='RdBu', vmin=-1, vmax=1, origin='lower')
        axes[0, 1].set_title('Normal Y')
        plt.colorbar(im2, ax=axes[0, 1])

        im3 = axes[0, 2].imshow(normal_z_np, cmap='Blues', vmin=0, vmax=1, origin='lower')
        axes[0, 2].set_title('Normal Z')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # 稳定性可视化
        im4 = axes[1, 0].imshow(stable_counts_np, cmap='viridis', origin='lower')
        axes[1, 0].set_title('Stable Yaw Count')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # 显示特定角度的稳定性（0度和90度）
        im5 = axes[1, 1].imshow(yaw_stability_np[:, :, 0], cmap='RdYlGn', origin='lower')
        axes[1, 1].set_title('Stability at 0°')
        plt.colorbar(im5, ax=axes[1, 1])
        
        im6 = axes[1, 2].imshow(yaw_stability_np[:, :, 4], cmap='RdYlGn', origin='lower')  # 约90度
        axes[1, 2].set_title('Stability at ~90°')
        plt.colorbar(im6, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('/home/yrf/MPT/yaw_stability_test.png')
        plt.close()
        
        print("可视化结果已保存到 yaw_stability_test.png")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader_integration():
    """测试数据加载器集成"""
    print("\n测试数据加载器集成...")
    
    # 检查数据目录是否存在
    data_folder = '/home/yrf/MPT/data/terrain_test/val'
    if not os.path.exists(data_folder):
        print("数据文件夹不存在，跳过数据加载器测试")
        return False
    
    # 查找可用的环境
    env_list = []
    for env_name in ['env000001']:
        env_path = os.path.join(data_folder, env_name)
        if os.path.exists(env_path):
            env_list.append(env_name)
    
    if not env_list:
        print("没有找到可用的数据环境，跳过数据加载器测试")
        return False
    
    print(f"找到可用环境: {env_list}")
    
    try:
        # 创建数据加载器
        dataset = UnevenPathDataLoader(env_list, data_folder)
        
        if len(dataset) == 0:
            print("数据集为空，跳过测试")
            return False
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试单个样本加载
        sample = dataset[0]
        
        print("样本数据结构:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)}")
        
        # 特别检查yaw_stability
        yaw_stability = sample['yaw_stability']
        print(f"\nyaw_stability详细信息:")
        print(f"  形状: {yaw_stability.shape}")
        print(f"  数值范围: {yaw_stability.min()} - {yaw_stability.max()}")
        print(f"  平均稳定角度比例: {yaw_stability.mean():.3f}")
        
        # 检查每个点的稳定角度数量
        stable_counts = torch.sum(yaw_stability, dim=2)
        print(f"  每个点稳定角度数量范围: {stable_counts.min()} - {stable_counts.max()}")
        
        return True
        
    except Exception as e:
        print(f"数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试yaw稳定性功能...\n")
    
    # 测试1: 基础计算功能
    success1 = test_yaw_stability_computation()
    
    # 测试2: 数据加载器集成
    success2 = test_dataloader_integration()
    
    # 总结
    print(f"\n测试结果:")
    print(f"  基础计算功能: {'✓' if success1 else '✗'}")
    print(f"  数据加载器集成: {'✓' if success2 else '✗'}")
    
    if success1 and success2:
        print("\n所有测试通过！yaw稳定性功能已成功集成。")
    else:
        print("\n部分测试失败，请检查实现。")

if __name__ == "__main__":
    main()
