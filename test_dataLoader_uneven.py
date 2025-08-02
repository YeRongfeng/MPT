"""
test_dataLoader_uneven.py - 测试不平坦地面数据加载器

1. 测试加载数据集
2. 测试数据集idx=0的返回值: map, anchor, labels
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import torch
from dataLoader_uneven import UnevenPathDataLoader, hashTable, receptive_field

import sys
sys.modules['numpy._core'] = np
sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
sys.modules['numpy._core.multiarray'] = np.core.multiarray

if __name__ == '__main__':
    # 测试加载数据集
    env_list = ['desert']
    dataFolder = '/home/yrf/MPT/data'
    dataset = UnevenPathDataLoader(env_list, dataFolder)
    
    # 测试数据集idx=0的返回值
    path_index = 333
    sample = dataset[path_index]
    print(sample['map'].shape)
    print(sample['anchor'].shape)
    print(sample['labels'].shape)
    
    # 读取真实的轨迹数据
    env_path = os.path.join(dataFolder, env_list[0])
    path_file = os.path.join(env_path, f'path_{path_index}.p')
    with open(path_file, 'rb') as f:
        path_data = pickle.load(f)
    trajectory = path_data['path']  # [N, 3]
    
    # 可视化
    # 创建坐标网格
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)

    # 创建图形
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.3)  # 增加子图间的水平间距

    # 绘制高程热力图 - 修正索引方式
    elevation_masked = np.ma.masked_invalid(sample['map'][0, 0, :, :])  # 第0通道是normal_z
    im = ax[0].imshow(elevation_masked, extent=[-5, 5, -5, 5],
                    origin='lower', cmap='jet', aspect='equal')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax[0], shrink=0.8, fraction=0.046, pad=0.04)
    cbar.set_label('normal_z', fontsize=12)
    
    # 绘制法向量箭头（只显示部分，避免过于密集）
    step = max(1, min(100, 100) // 20)  # 自适应步长

    for i in range(0, 100, step):
        for j in range(0, 100, step):
            # 修正索引：normal_z是第0通道，cos是第2通道，sin是第3通道
            if not (np.isnan(sample['map'][0, 0, i, j]) or
                    np.isnan(sample['map'][0, 2, i, j]) or
                    np.isnan(sample['map'][0, 3, i, j])):

                # 栅格中心坐标
                cx = x[j]
                cy = y[i]

                # 法向量的XY分量合成向量对应的cos和sin
                cos = sample['map'][0, 2, i, j]  # 第2通道是cos
                sin = sample['map'][0, 3, i, j]  # 第3通道是sin
                
                nxy_norm = 1 - (sample['map'][0, 0, i, j]) ** 2  # 第0通道是normal_z
                
                nx = cos * nxy_norm
                ny = sin * nxy_norm

                # 箭头长度（根据地图尺寸自适应）
                arrow_scale = min(10, 10) / 30

                # 绘制箭头
                ax[0].arrow(cx, cy, nx * arrow_scale, ny * arrow_scale,
                        head_width=arrow_scale*0.3, head_length=arrow_scale*0.2,
                        fc='red', ec='red', alpha=0.9, zorder=15, linewidth=1.5)
    
    # 设置网格
    ax[0].grid(True, alpha=0.3)
    ax[0].set_title('Normal_z and Surface Normals')
    
    # 绘制上下文 - 修正索引方式
    context = sample['map'][0, 1, :, :]  # 第1通道是context
    im = ax[1].imshow(context, extent=[-5, 5, -5, 5],
                    origin='lower', cmap='jet', aspect='equal')
    
    # 添加起点和终点图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='start (-1)'),
        Patch(facecolor='red', label='target (1)')
    ]
    ax[1].scatter(trajectory[0, 0], trajectory[0, 1], color='green', label='start (-1)', s=50)
    ax[1].scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', label='target (1)', s=50)
    ax[1].legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=2)
    ax[1].set_title('Context Map')
    ax[1].grid(True, alpha=0.3)
    
    # 绘制锚点信息
    anchor = sample['anchor']  # [num_layers, max_anchors]
    labels = sample['labels']  # [num_layers, max_anchors]
    
    print(f"锚点数据形状: {anchor.shape}")
    print(f"标签数据形状: {labels.shape}")
    print(f"正样本数={torch.sum(labels == 1)}, 负样本数={torch.sum(labels == 0)}")
    
    num_layers = anchor.shape[0]
    
    # 为每个图层绘制不同深度的实心矩形
    num_trajectory_points = num_layers // 2  # 前半部分是正样本图层
    
    for layer_idx in range(num_trajectory_points):  # 只绘制正样本图层
        # 只绘制正样本锚点
        layer_anchors = anchor[layer_idx]
        layer_labels = labels[layer_idx]
        
        positive_mask = layer_labels == 1
        positive_anchors = layer_anchors[positive_mask]
        
        # 计算颜色深度：时序越后（layer_idx越大），颜色越深
        color_intensity = 1.0 - 0.8 * (layer_idx / max(1, num_trajectory_points - 1))  # 从1.0到0.2
        white_mix = 0 # 混合白色的比例，0表示纯色，1表示纯白
        # white_mix = 0.3  # 混合白色的比例，0表示纯色，1表示纯白
        
        # 计算混合白色后的RGB值
        red = color_intensity * (1 - white_mix) + white_mix
        green = 0 * (1 - white_mix) + white_mix  
        blue = 0 * (1 - white_mix) + white_mix
        
        for anchor_idx in positive_anchors:
            if anchor_idx == -1:  # 跳过填充值
                continue
                
            anchor_idx = anchor_idx.item()
            hash_col, hash_row = hashTable[anchor_idx]
            
            # 将像素坐标转换为实际坐标
            cx = -5 + hash_col * 0.1   # 列对应x坐标
            cy = -5 + hash_row * 0.1   # 行对应y坐标
            
            # 计算感受野范围（实际尺寸）
            rf_size = receptive_field * 0.1  # 感受野实际尺寸（米）
            rect_x = cx - rf_size / 2
            rect_y = cy - rf_size / 2
            rect_width = rf_size
            rect_height = rf_size
            
            # 绘制实心矩形，使用混合白色的颜色
            ax[2].add_patch(plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                        fill=True, 
                                        facecolor=(red, green, blue),  # 混合白色后的RGB
                                        edgecolor='none', 
                                        linewidth=1))
            
    # 绘制轨迹 - 渐变颜色显示时序，箭头显示方向
    path_xy = trajectory[:, :2]  # 只取x和y坐标
    n_points = len(path_xy)
    
    # 使用渐变颜色绘制轨迹线段
    for i in range(n_points - 1):
        # 计算颜色渐变：从绿色(起点)到红色(终点)
        color_ratio = i / (n_points - 1)
        # color = plt.cm.RdYlGn_r(color_ratio)  # 反向红黄绿色图
        color = plt.cm.rainbow(color_ratio)  # 彩虹色图

        # 绘制线段
        ax[2].plot([path_xy[i, 0], path_xy[i+1, 0]], 
                  [path_xy[i, 1], path_xy[i+1, 1]], 
                  color=color, linewidth=3, alpha=1.0)
    
    # 绘制关键点：起点、终点和方向箭头
    ax[2].scatter(path_xy[0, 0], path_xy[0, 1], color='blue', s=100, 
                 marker='o', label='Start', zorder=10, edgecolor='blue', linewidth=2)
    ax[2].scatter(path_xy[-1, 0], path_xy[-1, 1], color='red', s=100, 
                 marker='s', label='Goal', zorder=10, edgecolor='red', linewidth=2)
    
    # 绘制方向箭头（每隔几个点绘制一个，避免过于密集）
    arrow_step = max(1, n_points // 10)  # 大约显示10个箭头
    path_theta = trajectory[:, 2]  # 取方向角度
    for i in range(0, n_points, arrow_step):
        arrow_scale = 0.2  # 箭头长度
        dx_norm = arrow_scale * np.cos(path_theta[i])  # x方向分量
        dy_norm = arrow_scale * np.sin(path_theta[i])  # y方向分量
        
        # 绘制箭头 - 使用白色并显示在最顶层
        ax[2].arrow(path_xy[i, 0], path_xy[i, 1], dx_norm, dy_norm,
                    head_width=0.1, head_length=0.08, 
                    fc='white', ec='white', alpha=0.6, zorder=15, linewidth=1.5)

    ax[2].set_xlim([-5, 5])
    ax[2].set_ylim([-5, 5])
    ax[2].set_title(f'Anchor Points ({num_trajectory_points} trajectory points)')
    ax[2].grid(True, alpha=0.3)
    
    plt.show()
