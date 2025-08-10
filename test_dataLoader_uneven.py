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
from dataLoader_uneven import UnevenPathDataLoader, hashTable, receptive_field, geom2pix, compute_terrain_direction
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

import sys
sys.modules['numpy._core'] = np
sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
sys.modules['numpy._core.multiarray'] = np.core.multiarray

if __name__ == '__main__':
    # 测试加载数据集
    env_list = ['desert']
    # env_list = ['hill']
    dataFolder = '/home/yrf/MPT/data/test_training/val'
    dataset = UnevenPathDataLoader(env_list, dataFolder)
    
    # 测试数据集idx=0的返回值
    path_index = 148
    sample = dataset[path_index]
    print(sample['map'].shape)
    print(sample['anchor'].shape)
    print(sample['labels'].shape)
    
    # 读取真实的轨迹数据
    env_path = os.path.join(dataFolder, env_list[0])
    path_file = os.path.join(env_path, f'path_{path_index}.p')
    with open(path_file, 'rb') as f:
        path_data = pickle.load(f)
    trajectory = path_data['path']  # [N+2, 3]
    
    # 可视化
    # 创建坐标网格
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)

    # 创建图形
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    plt.subplots_adjust(wspace=0.3)  # 增加子图间的水平间距

    # 绘制高程热力图(图1)
    elevation_masked = np.ma.masked_invalid(sample['map'][0, 2, :, :])  # 第0通道是normal_z
    im = ax[0, 0].imshow(elevation_masked, extent=[-5, 5, -5, 5],
                    origin='lower', cmap='jet', aspect='equal')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax[0, 0], shrink=0.8, fraction=0.046, pad=0.04)
    cbar.set_label('normal_z', fontsize=12)
    
    # 绘制法向量箭头（只显示部分，避免过于密集）
    step = max(1, min(100, 100) // 20)  # 自适应步长

    for i in range(0, 100, step):
        for j in range(0, 100, step):
            # 修正索引：normal_x是第0通道，normal_y是第1通道，normal_z是第2通道
            if not (np.isnan(sample['map'][0, 0, i, j]) or
                    np.isnan(sample['map'][0, 1, i, j]) or
                    np.isnan(sample['map'][0, 2, i, j])):

                # 栅格中心坐标
                cx = x[j]
                cy = y[i]
                
                nx = sample['map'][0, 0, i, j]
                ny = sample['map'][0, 1, i, j]

                # 箭头长度（根据地图尺寸自适应）
                arrow_scale = min(10, 10) / 30

                # 绘制箭头
                ax[0, 0].arrow(cx, cy, nx * arrow_scale, ny * arrow_scale,
                        head_width=arrow_scale*0.3, head_length=arrow_scale*0.2,
                        fc='red', ec='red', alpha=0.9, zorder=15, linewidth=1.5)
    
    # 设置网格
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].set_title('Normal_z and Surface Normals')

    # 绘制上下文（图2）
    context = sample['map'][0, 3, :, :]  # 第1通道是context
    im = ax[0, 1].imshow(context, extent=[-5, 5, -5, 5],
                    origin='lower', cmap='jet', aspect='equal')
    
    # 添加起点和终点图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='start (-1)'),
        Patch(facecolor='red', label='target (1)')
    ]
    ax[0, 1].scatter(trajectory[0, 0], trajectory[0, 1], color='green', label='start (-1)', s=50)
    ax[0, 1].scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', label='target (1)', s=50)
    ax[0, 1].legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=2)
    ax[0, 1].set_title('Context Map')
    ax[0, 1].grid(True, alpha=0.3)
    
    # 绘制锚点信息
    anchor = sample['anchor']  # [num_layers, max_anchors]
    labels = sample['labels']  # [num_layers, max_anchors]
    
    print(f"锚点数据形状: {anchor.shape}")
    print(f"标签数据形状: {labels.shape}")
    print(f"正样本数={torch.sum(labels == 1)}, 负样本数={torch.sum(labels == 0)}")
    
    num_layers = anchor.shape[0]
    
    # # 绘制角度导向场（图3）
    # # 准备轨迹点数据
    # traj_points = trajectory[1:-1, :]  # 去掉起点和终点，只取中间轨迹点
    
    # # 初始化热扩散场，使用12x12的锚点网格大小
    # grid_size = 12
    # D = np.zeros((grid_size, grid_size))
    
    # # 设置轨迹点热源
    # for traj_point in traj_points:
    #     x, y, theta = traj_point
        
    #     # 使用geom2pix获取像素坐标
    #     pixel_row, pixel_col = geom2pix((x, y), res=0.1, size=(100, 100))
        
    #     # 在hashTable中寻找最接近的锚点，并转换为网格坐标
    #     min_distance = float('inf')
    #     best_grid_row, best_grid_col = 0, 0
        
    #     for anchor_idx, (hash_col, hash_row) in enumerate(hashTable):
    #         distance = np.sqrt((pixel_col - hash_col)**2 + (pixel_row - hash_row)**2)
    #         if distance < min_distance:
    #             min_distance = distance
    #             # 将anchor_idx转换为网格坐标
    #             best_grid_row = anchor_idx // grid_size
    #             best_grid_col = anchor_idx % grid_size
        
    #     # 在网格中设置高斯热源
    #     for di in range(-2, 3):
    #         for dj in range(-2, 3):
    #             gi, gj = best_grid_row + di, best_grid_col + dj
    #             if 0 <= gi < grid_size and 0 <= gj < grid_size:
    #                 # 高斯权重
    #                 weight = np.exp(-(di**2 + dj**2) / (2 * 1.0**2))
    #                 D[gi, gj] += weight
    
    # # 热扩散模拟
    # kernel = np.array([[0.05, 0.2, 0.05], [0.2, 0.4, 0.2], [0.05, 0.2, 0.05]])
    # for _ in range(30):
    #     D = convolve2d(D, kernel, mode='same', boundary='symm')
    
    # # 归一化
    # if D.max() > 0:
    #     D = D / D.max()
    
    # # 创建角度场
    # A_super = np.zeros((grid_size, grid_size, 2))  # [cosφ, sinφ] - 存储基础流向
    # A_terrain = np.zeros((grid_size, grid_size, 2))  # [cosφ, sinφ] - 存储地形修正后的方向
    
    # # 轨迹点方向注入
    # for traj_point in traj_points:
    #     x, y, theta = traj_point
        
    #     # 获取网格坐标
    #     pixel_row, pixel_col = geom2pix((x, y), res=0.1, size=(100, 100))
    #     min_distance = float('inf')
    #     best_grid_row, best_grid_col = 0, 0
        
    #     for anchor_idx, (hash_col, hash_row) in enumerate(hashTable):
    #         distance = np.sqrt((pixel_col - hash_col)**2 + (pixel_row - hash_row)**2)
    #         if distance < min_distance:
    #             min_distance = distance
    #             best_grid_row = anchor_idx // grid_size
    #             best_grid_col = anchor_idx % grid_size
        
    #     A_super[best_grid_row, best_grid_col] = [np.cos(theta), np.sin(theta)]
    
    # # 方向传播 - 从起点发出汇聚于终点的流场
    # start_point = np.array([trajectory[0, 0], trajectory[0, 1]])    # 起点
    # end_point = np.array([trajectory[-1, 0], trajectory[-1, 1]])    # 终点
    
    # # 调试计数器
    # total_points = 0
    # terrain_corrected_points = 0
    # terrain_corrections = []  # 记录地形修正信息: [(位置x, 位置y, 修正前方向, 修正后方向)]
    
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         if D[i, j] > 0.1:  # 在影响区域内
    #             total_points += 1
    #             # 将网格坐标转换为实际坐标
    #             anchor_idx = i * grid_size + j
    #             if anchor_idx < len(hashTable):
    #                 hash_col, hash_row = hashTable[anchor_idx]
    #                 grid_x = -5 + hash_col * 0.1
    #                 grid_y = -5 + hash_row * 0.1
    #                 current_pos = np.array([grid_x, grid_y])
                    
    #                 # 基础流向：从起点发出，汇聚于终点
    #                 # 计算到起点和终点的距离
    #                 dist_to_start = np.linalg.norm(current_pos - start_point)
    #                 dist_to_end = np.linalg.norm(current_pos - end_point)
                    
    #                 # 基本流向：指向终点
    #                 if dist_to_end > 0:
    #                     base_flow = (end_point - current_pos) / dist_to_end
    #                 else:
    #                     base_flow = np.array([0, 0])
                    
    #                 # 轨迹引导：找到最近的轨迹点和方向
    #                 min_traj_dist = float('inf')
    #                 nearest_theta = 0
                    
    #                 for traj_point in traj_points:
    #                     tx, ty, ttheta = traj_point
    #                     dist = np.sqrt((grid_x - tx)**2 + (grid_y - ty)**2)
    #                     if dist < min_traj_dist:
    #                         min_traj_dist = dist
    #                         nearest_theta = ttheta
                    
    #                 # 轨迹方向
    #                 traj_flow = np.array([np.cos(nearest_theta), np.sin(nearest_theta)])
                    
    #                 # 融合权重：距离轨迹越近，轨迹引导越强
    #                 if min_traj_dist < 1.5:  # 轨迹影响范围
    #                     traj_weight = np.exp(-min_traj_dist / 0.5)  # 指数衰减
    #                     final_dir = (1 - traj_weight) * base_flow + traj_weight * traj_flow
    #                 else:
    #                     final_dir = base_flow
                    
    #                 # 归一化基础流向并存储
    #                 final_norm = np.linalg.norm(final_dir)
    #                 if final_norm > 0:
    #                     final_dir = final_dir / final_norm
    #                 A_super[i, j] = final_dir  # 存储基础流向
                    
    #                 # 获取地形法向量信息进行地形修正
    #                 terrain_corrected_dir = final_dir.copy()  # 初始化为基础流向
    #                 map_row = int(hash_row)
    #                 map_col = int(hash_col)
    #                 if 0 <= map_row < 100 and 0 <= map_col < 100:
    #                     # 从地图数据中获取法向量
    #                     nz = sample['map'][0, 0, map_row, map_col]  # normal_z
    #                     cos_val = sample['map'][0, 4, map_row, map_col]  # cos
    #                     sin_val = sample['map'][0, 5, map_row, map_col]  # sin
                        
    #                     if not (np.isnan(nz) or np.isnan(cos_val) or np.isnan(sin_val)):
    #                         # terrain_corrected_points += 1
    #                         nxy_norm = np.sqrt(1 - nz**2) if nz**2 < 1 else 0
    #                         nx = cos_val * nxy_norm
    #                         ny = sin_val * nxy_norm
                            
    #                         # # 调试：检查地形参数
    #                         # terrain_slope = np.arctan2(np.sqrt(nx**2 + ny**2), abs(nz))
    #                         # h = 8
    #                         # b_val = h * np.tan(terrain_slope)
    #                         # print(f"位置({grid_x:.1f},{grid_y:.1f}): nz={nz:.3f}, slope={terrain_slope:.3f}rad={terrain_slope*180/np.pi:.1f}°, b={b_val:.2f}")
                            
    #                         terrain_dir = compute_terrain_direction(nx, ny, nz, final_dir[0], final_dir[1])
                            
    #                         # 调试：检查terrain_dir和final_dir是否相同
    #                         direction_similarity = np.dot(final_dir, terrain_dir)
    #                         if abs(direction_similarity - 1.0) >= 0.001:
    #                             terrain_corrected_points += 1
    #                             print(f"  -> 有地形修正! 相似度={direction_similarity:.6f}")
                                
    #                             # 计算修正前后的角度差异
    #                             angle_change = np.arccos(np.clip(direction_similarity, -1, 1)) * 180 / np.pi
    #                             print(f"  -> 角度变化: {angle_change:.1f}°")
                            
    #                         # 选择与当前方向更接近的地形方向
    #                         similarity = np.dot(final_dir, terrain_dir)
    #                         if similarity < 0:
    #                             terrain_dir = -terrain_dir
                                
    #                         # 地形影响权重
    #                         # terrain_weight = nz  # 可调节地形影响强度
    #                         # terrain_weight = 0  # 无地形影响
    #                         # terrain_weight = 0.5  # 中等地形影响
    #                         terrain_weight = 1  # 完全地形约束
                            
    #                         # 记录地形修正前的方向（基础流向）
    #                         before_terrain = final_dir.copy()
                            
    #                         # 进行地形修正
    #                         terrain_corrected_dir = (1 - terrain_weight) * final_dir + terrain_weight * terrain_dir
                            
    #                         # 归一化地形修正后的方向
    #                         terrain_norm = np.linalg.norm(terrain_corrected_dir)
    #                         if terrain_norm > 0:
    #                             terrain_corrected_dir = terrain_corrected_dir / terrain_norm
                            
    #                         # 记录地形修正信息
    #                         terrain_corrections.append((grid_x, grid_y, before_terrain.copy(), terrain_corrected_dir.copy()))
                            
    #                         # 计算修正前后的角度差异
    #                         if terrain_weight > 0:
    #                             angle_change = np.arccos(np.clip(np.dot(before_terrain, terrain_corrected_dir), -1, 1)) * 180 / np.pi
    #                             if angle_change > 5:
    #                                 print(f"地形修正: 位置({grid_x:.1f},{grid_y:.1f}), 修正前{before_terrain}, 修正后{terrain_corrected_dir}, 变化{angle_change:.1f}°")
                    
    #                 # 存储地形修正后的方向
    #                 A_terrain[i, j] = terrain_corrected_dir
    
    # # 打印调试信息
    # print(f"总共处理了 {total_points} 个网格点")
    # print(f"其中 {terrain_corrected_points} 个点进行了地形修正")
    
    # # 绘制角度导向场
    # ax[1, 0].imshow(D, extent=[-5, 5, -5, 5], origin='lower', cmap='hot', alpha=0.6)
    
    # # 绘制基础流向箭头（蓝色）
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         if D[i, j] > 0.1:
    #             # 计算箭头位置
    #             anchor_idx = i * grid_size + j
    #             if anchor_idx < len(hashTable):
    #                 hash_col, hash_row = hashTable[anchor_idx]
    #                 arrow_x = -5 + hash_col * 0.1
    #                 arrow_y = -5 + hash_row * 0.1
                    
    #                 # 获取基础流向
    #                 dx, dy = A_super[i, j]
    #                 if dx != 0 or dy != 0:
    #                     arrow_scale = 0.3
    #                     ax[1, 0].arrow(arrow_x, arrow_y, dx * arrow_scale, dy * arrow_scale,
    #                                  head_width=0.1, head_length=0.08,
    #                                  fc='blue', ec='blue', alpha=0.6, linewidth=1)
    
    # # 绘制地形修正后的方向（红色，只绘制有修正的位置）
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         if D[i, j] > 0.1:
    #             # 计算箭头位置
    #             anchor_idx = i * grid_size + j
    #             if anchor_idx < len(hashTable):
    #                 hash_col, hash_row = hashTable[anchor_idx]
    #                 arrow_x = -5 + hash_col * 0.1
    #                 arrow_y = -5 + hash_row * 0.1
                    
    #                 # 获取地形修正后的方向
    #                 dx, dy = A_terrain[i, j]
    #                 # 检查是否与基础流向不同（说明进行了地形修正）
    #                 base_dx, base_dy = A_super[i, j]
    #                 if (abs(dx - base_dx) > 0.01 or abs(dy - base_dy) > 0.01) and (dx != 0 or dy != 0):
    #                     arrow_scale = 0.35
    #                     ax[1, 0].arrow(arrow_x, arrow_y, dx * arrow_scale, dy * arrow_scale,
    #                                  head_width=0.12, head_length=0.09,
    #                                  fc='g', ec='g', alpha=0.5, linewidth=2, zorder=25)
    
    # # 绘制轨迹
    # ax[1, 0].plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, alpha=0.8, label='Trajectory')
    # ax[1, 0].scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=50, marker='o', label='Start')
    # ax[1, 0].scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=50, marker='s', label='Goal')
    
    # # # 添加箭头图例
    # # from matplotlib.patches import FancyArrow
    # # legend_elements = [
    # #     FancyArrow(0, 0, 0, 0, fc='blue', ec='blue', label='Base Flow (Before Terrain)'),
    # #     FancyArrow(0, 0, 0, 0, fc='red', ec='red', label='After Terrain Correction')
    # # ]
    
    # ax[1, 0].set_xlim([-5, 5])
    # ax[1, 0].set_ylim([-5, 5])
    # ax[1, 0].set_title('Angle Guidance Field with Terrain Correction')
    # ax[1, 0].grid(True, alpha=0.3)
    # # ax[1, 0].legend(loc='upper left')
    
    # 为每个图层绘制不同深度的实心矩形（图4）
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
            hash_row, hash_col = hashTable[anchor_idx]
            
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
            ax[1, 1].add_patch(plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
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
        ax[1, 1].plot([path_xy[i, 0], path_xy[i+1, 0]], 
                  [path_xy[i, 1], path_xy[i+1, 1]], 
                  color=color, linewidth=3, alpha=1.0)
    
    # 绘制关键点：起点、终点和方向箭头
    ax[1, 1].scatter(path_xy[0, 0], path_xy[0, 1], color='blue', s=100, 
                 marker='o', label='Start', zorder=10, edgecolor='blue', linewidth=2)
    ax[1, 1].scatter(path_xy[-1, 0], path_xy[-1, 1], color='red', s=100, 
                 marker='s', label='Goal', zorder=10, edgecolor='red', linewidth=2)
    
    # 绘制方向箭头（每隔几个点绘制一个，避免过于密集）
    # arrow_step = max(1, n_points // 10)  # 大约显示10个箭头
    arrow_step = 1
    path_theta = trajectory[:, 2]  # 取方向角度
    for i in range(0, n_points, arrow_step):
        arrow_scale = 0.2  # 箭头长度
        dx_norm = arrow_scale * np.cos(path_theta[i])  # x方向分量
        dy_norm = arrow_scale * np.sin(path_theta[i])  # y方向分量
        
        # 绘制箭头 - 使用白色并显示在最顶层
        ax[1, 1].arrow(path_xy[i, 0], path_xy[i, 1], dx_norm, dy_norm,
                    head_width=0.1, head_length=0.08, 
                    fc='white', ec='white', alpha=0.6, zorder=15, linewidth=1.5)

    ax[1, 1].set_xlim([-5, 5])
    ax[1, 1].set_ylim([-5, 5])
    ax[1, 1].set_title(f'Anchor Points ({num_trajectory_points}+2 trajectory points)')
    ax[1, 1].grid(True, alpha=0.3)
    
    plt.show()
