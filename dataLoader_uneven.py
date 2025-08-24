"""
dataLoader_uneven.py - MPT路径规划数据加载器模块
"""

# 【核心依赖库导入】
import torch  # PyTorch深度学习框架：张量计算和自动微分
from torch.utils.data import Dataset  # 数据集基类：提供数据加载的标准接口

import skimage.io  # 图像IO操作：读取地图图像文件
import pickle  # 序列化库：加载路径数据文件
import numpy as np  # 数值计算库：数组操作和数学计算

import os  # 操作系统接口：文件和目录操作
from os import path as osp  # 路径操作：文件路径处理
from einops import rearrange  # 张量重排：高效的维度操作

from torch.nn.utils.rnn import pad_sequence  # 序列填充：处理变长序列的批处理

from utils import geom2pix  # 坐标转换工具：几何坐标到像素坐标的转换

# 添加兼容性处理
import sys
sys.modules['numpy._core'] = np
sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
sys.modules['numpy._core.multiarray'] = np.core.multiarray

# 【全局参数配置】
input_size = 100
output_grid = 12
anchor_spacing = 8
boundary_offset = 6

map_size = (input_size, input_size)  # 地图尺寸：100x100像素的标准地图大小
receptive_field = 38   # 感受野大小：每个锚点影响的像素范围 TODO
res = 0.1              # 地图分辨率：每像素代表0.1米的实际距离

# 【理论最大正样本数自动计算】
# 感受野区域大小：receptive_field * res = 38 * 0.1 = 3.8米
# 锚点间距：anchor_spacing * res = 8 * 0.1 = 0.8米
# x方向最大锚点数：ceil(3.8 / 0.8) = 5个
# y方向最大锚点数：ceil(3.8 / 0.8) = 5个
# 理论最大正样本数：5 × 5 = 25个
import math
receptive_field_size = receptive_field * res  # 感受野的实际大小（米）
anchor_spacing_size = anchor_spacing * res    # 锚点间距的实际大小（米）
max_anchors_per_axis = math.ceil(receptive_field_size / anchor_spacing_size)  # 每个轴向最大锚点数
MAX_POSITIVE_ANCHORS = max_anchors_per_axis * max_anchors_per_axis  # 理论最大正样本数

# 【锚点网格系统构建】
# 将连续的地图空间离散化为12x12的锚点网格，用于Transformer的token化处理

# X轴锚点坐标：从6像素开始，每8像素一个锚点，转换为几何坐标: 
# [6, 14, 22, ..., 94] * res - 5 = [-4.4, -3.6, -2.8, ..., 4.4] 米
X = np.arange(boundary_offset, output_grid*anchor_spacing+boundary_offset, anchor_spacing)*res - 5

# Y轴锚点坐标：从6像素开始，每8像素一个锚点，转换为几何坐标: 
# [6, 14, 22, ..., 94] * res - 5 = [-4.4, -3.6, -2.8, ..., 4.4] 米
Y = np.arange(boundary_offset, output_grid*anchor_spacing+boundary_offset, anchor_spacing)*res - 5

# 创建2D网格：生成所有锚点的几何坐标
grid_2d = np.meshgrid(X, Y)  # 创建X-Y坐标网格

# 网格点重排：将2D网格转换为(N, 2)的点列表，N=144个锚点
# 注意：必须与hashTable的生成顺序保持一致！
# hashTable按行优先顺序：for r in range(output_grid) for c in range(output_grid)
# 因此grid_points也必须按相同顺序重排
XX, YY = grid_2d[0], grid_2d[1]  # XX是x坐标矩阵，YY是y坐标矩阵
grid_points = np.array([[XX[r, c], YY[r, c]] 
                       for r in range(output_grid) for c in range(output_grid)])  # 形状：(144, 2)

# print(grid_points)

# 哈希表：锚点索引到像素坐标的映射表
# 修正：与grid_points保持一致的[x, y]顺序（而不是[r, c]=[y, x]顺序）
hashTable = [(anchor_spacing*c+boundary_offset, anchor_spacing*r+boundary_offset)
             for r in range(output_grid) for c in range(output_grid)]

# print(hashTable[:5])  # 打印前5个锚点坐标

# 【网格系统说明】
# 1. 锚点分布：12x12=144个锚点均匀分布在地图上
# 2. 像素间距：每个锚点间隔8像素，对应0.8米的实际距离
# 3. 边界偏移：起始偏移6像素，确保锚点不在地图边缘
# 4. 坐标对应：每个锚点代表一个8x8像素的区域
# 5. 索引映射：通过hashTable实现1D索引到2D像素坐标的转换

def geom2pixMatpos(pos, res=0.1, size=(100, 100)):
    # 计算输入位置到所有锚点的距离
    distances = np.linalg.norm(grid_points - pos, axis=1)  # 形状：(100,)
    
    # 筛选距离阈值内的锚点索引
    # indices = np.where(distances <= receptive_field * res * 0.5)  # 阈值：18 * 0.1 * 0.5 = 0.9米
    # indices = np.where(distances <= receptive_field * res * 0.4)  # 阈值：18 * 0.1 * 0.4 = 0.72米

    # 筛选感受野区域内包含输入位置的锚点索引（感受野区域是矩形）
    indices = np.where((grid_points[:, 0] >= pos[0] - receptive_field * res / 2) &
                       (grid_points[:, 0] <= pos[0] + receptive_field * res / 2) &
                       (grid_points[:, 1] >= pos[1] - receptive_field * res / 2) &
                       (grid_points[:, 1] <= pos[1] + receptive_field * res / 2))

    return indices  # 返回正样本锚点索引元组

def geom2pix(pos, res=0.1, size=(100, 100)):
    """
    几何坐标到像素坐标的转换函数
    
    Args:
        pos: 几何坐标 (x, y)，单位：米
        res: 地图分辨率，米/像素
        size: 地图尺寸 (height, width)
    
    Returns:
        tuple: 像素坐标 (row, col)
    """
    # 根据地图边界 (-5, 5, -5, 5) 和分辨率 0.1 进行转换
    x, y = pos
    
    # 将几何坐标转换为像素坐标
    # x: -5 到 5 映射到 0 到 100
    # y: -5 到 5 映射到 0 到 100
    col = int((x + 5.0) / res)
    row = int((y + 5.0) / res)
    
    # 边界检查
    row = max(0, min(size[0] - 1, row))
    col = max(0, min(size[1] - 1, col))
    
    return (row, col)


def PaddedSequence(batch):
    """
    固定尺寸批处理整理函数（用于Transformer训练）
    
    【核心功能】
    由于DataLoader已经确保所有样本的anchor和labels具有固定尺寸(MAX_POSITIVE_ANCHORS)，
    此函数只需简单地将批处理数据堆叠即可，无需复杂的填充处理。
    
    【处理流程】
    1. 过滤无效样本：移除None值的样本
    2. 数据堆叠：直接堆叠所有数据到批处理维度
    
    Args:
        batch (list): 批处理样本列表
            每个元素包含：
            - 'map': 地图张量，形状为(1, C, H, W)
            - 'anchor': 锚点序列，形状为(2*N, MAX_POSITIVE_ANCHORS) - 已固定尺寸
            - 'labels': 标签序列，形状为(2*N, MAX_POSITIVE_ANCHORS) - 已固定尺寸
            - 'trajectory': 轨迹序列，形状为(N+2, 3)
    
    Returns:
        dict: 整理后的批处理数据
            - 'map': 批处理地图，形状为(B, C, H, W)
            - 'anchor': 批处理锚点序列，形状为(B, 2*N, MAX_POSITIVE_ANCHORS)
            - 'labels': 批处理标签序列，形状为(B, 2*N, MAX_POSITIVE_ANCHORS)
            - 'length': 每个样本的序列长度，形状为(B,) - 用于兼容性
            - 'trajectory': 批处理轨迹序列，形状为(B, N+2, 3)
    """
    # 过滤有效样本：移除None值，确保数据完整性
    valid_batch = [batch_i for batch_i in batch if batch_i is not None]
    
    # 由于所有样本已经具有固定尺寸，直接堆叠即可
    data = {
        'map': torch.stack([batch_i['map'] for batch_i in valid_batch]),  # [B, C, H, W]
        # 'pose': torch.stack([batch_i['pose'] for batch_i in valid_batch]),  # [B, 2, 4] - 起点和终点位姿
        'anchor': torch.stack([batch_i['anchor'] for batch_i in valid_batch]),  # [B, 2*N, MAX_POSITIVE_ANCHORS]
        'labels': torch.stack([batch_i['labels'] for batch_i in valid_batch]),  # [B, 2*N, MAX_POSITIVE_ANCHORS]
        'length': torch.tensor([batch_i['anchor'].shape[0] for batch_i in valid_batch]),  # [B,] - 序列长度
        'trajectory': torch.stack([batch_i['trajectory'] for batch_i in valid_batch]),  # [B, N+2, 3]
        'yaw_stability': torch.stack([batch_i['yaw_stability'] for batch_i in valid_batch]),  # [B, H, W, 36] - yaw分箱倾覆状态
        'cost_map': torch.stack([batch_i['cost_map'] for batch_i in valid_batch]),  # [B, H, W, yaw_bins] - 成本图
    }
    
    return data

def get_encoder_input(normal_z, goal_state, start_state, normal_x, normal_y):
    """
    构造编码输入，包含数据验证和修复机制
    """
    
    # # 确保 nz 全为非负数
    # normal_z = torch.abs(normal_z)  # 确保法向量Z分量非负
    
    # # 输入数据验证和修复
    # def validate_and_fix_normal_component(component, component_name, default_value=0.0):
    #     """验证和修复法向量分量"""
    #     if not np.all(np.isfinite(component)):
    #         invalid_count = np.sum(~np.isfinite(component))
    #         print(f"Warning: {component_name} contains {invalid_count} invalid values, applying fixes")
    #         component = np.nan_to_num(component, nan=default_value, posinf=1.0, neginf=-1.0)
        
    #     # 约束到合理范围
    #     component = np.clip(component, -1.0, 1.0)
    #     return component
    
    # # 修复各个法向量分量
    # normal_x = validate_and_fix_normal_component(normal_x, "normal_x", 0.0)
    # normal_y = validate_and_fix_normal_component(normal_y, "normal_y", 0.0) 
    # normal_z = validate_and_fix_normal_component(normal_z, "normal_z", 1.0)  # Z默认向上
    
    # 逐点归一化法向量，确保单位长度
    for i in range(normal_z.shape[0]):
        for j in range(normal_z.shape[1]):
            norm_vec = np.array([normal_x[i, j], normal_y[i, j], normal_z[i, j]])
            norm_length = np.linalg.norm(norm_vec)
            
            if norm_length > 1e-8:
                norm_vec = norm_vec / norm_length
            else:
                norm_vec = np.array([0.0, 0.0, 1.0])  # 默认向上
            
            normal_x[i, j] = norm_vec[0]
            normal_y[i, j] = norm_vec[1]
            normal_z[i, j] = norm_vec[2]
    
    # 构造编码输入[H, W, 6]
    goal_pos = goal_state[:2]  # 终点位置 (x, y)
    start_pos = start_state[:2]  # 起点位置 (x, y)
    goal_angle = goal_state[2]  # 终点朝向
    start_angle = start_state[2]  # 起点朝向
    
    goal_index = geom2pix(goal_pos, res=res, size=normal_z.shape[:2])
    start_index = geom2pix(start_pos, res=res, size=normal_z.shape[:2])

    # 起点区域（使用start_index）
    start_start_y = max(0, start_index[0] - receptive_field//2)
    start_start_x = max(0, start_index[1] - receptive_field//2)
    start_end_y = min(normal_z.shape[0], start_index[0] + receptive_field//2)
    start_end_x = min(normal_z.shape[1], start_index[1] + receptive_field//2)

    # 终点区域（使用goal_index）
    goal_start_y = max(0, goal_index[0] - receptive_field//2)
    goal_start_x = max(0, goal_index[1] - receptive_field//2)
    goal_end_y = min(normal_z.shape[0], goal_index[0] + receptive_field//2)
    goal_end_x = min(normal_z.shape[1], goal_index[1] + receptive_field//2)
    
    # 上下文地图： 起点终点的 位置标记 + 朝向的余弦值 + 朝向的正弦值，组成3通道输入
    context_map = np.zeros((*normal_z.shape[:2], 3))  # [H, W, 3]
    context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x, 0] = 1.0  # 终点标记为1
    context_map[start_start_y:start_end_y, start_start_x:start_end_x, 0] = -1.0  # 起点标记为-1
    
    # 检查角度值是否有效
    if not (np.isfinite(goal_angle) and np.isfinite(start_angle)):
        print("Warning: Invalid angles detected, using default values")
        goal_angle = 0.0
        start_angle = 0.0
    
    context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x, 1] = np.cos(goal_angle)  # 终点朝向
    context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x, 2] = np.sin(goal_angle)  # 终点朝向
    context_map[start_start_y:start_end_y, start_start_x:start_end_x, 1] = np.cos(start_angle)  # 起点朝向
    context_map[start_start_y:start_end_y, start_start_x:start_end_x, 2] = np.sin(start_angle)  # 起点朝向

    # 拼接为6通道
    encoded_input = np.concatenate((normal_x[:, :, None], normal_y[:, :, None], normal_z[:, :, None], context_map[:, :, :3]), axis=2)
    
    # 最终检查编码输入的有效性
    if not np.all(np.isfinite(encoded_input)):
        print("Warning: Final encoded input contains invalid values, applying final cleanup")
        encoded_input = np.nan_to_num(encoded_input, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return encoded_input

# def get_encoder_input(normal_z, goal_pos, start_pos, normal_x, normal_y):
#     goal_index = geom2pix(goal_pos, res=res, size=normal_z.shape[:2])
#     start_index = geom2pix(start_pos, res=res, size=normal_z.shape[:2])

#     # 起点区域（使用start_index）
#     start_start_y = max(0, start_index[0] - receptive_field//2)
#     start_start_x = max(0, start_index[1] - receptive_field//2)
#     start_end_y = min(normal_z.shape[0], start_index[0] + receptive_field//2)
#     start_end_x = min(normal_z.shape[1], start_index[1] + receptive_field//2)

#     # 终点区域（使用goal_index）
#     goal_start_y = max(0, goal_index[0] - receptive_field//2)
#     goal_start_x = max(0, goal_index[1] - receptive_field//2)
#     goal_end_y = min(normal_z.shape[0], goal_index[0] + receptive_field//2)
#     goal_end_x = min(normal_z.shape[1], goal_index[1] + receptive_field//2)
    
#     # 上下文地图： 起点终点的位置标记
#     context_map = np.zeros(normal_z.shape[:2])  # [H, W]
#     context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0  # 终点标记为1
#     context_map[start_start_y:start_end_y, start_start_x:start_end_x] = -1.0  # 起点标记为-1

#     # 构造θ=<nx,ny>->(cosθ, sinθ)的映射
#     angle_map = np.zeros((normal_z.shape[0], normal_z.shape[1], 2))  # [H, W, 2]
#     for i in range(normal_z.shape[0]):
#         for j in range(normal_z.shape[1]):
#             n_xy_norm = np.linalg.norm([normal_x[i, j], normal_y[i, j]])
#             if n_xy_norm == 0:
#                 # 如果法向量为零，避免除以零
#                 angle_map[i, j, 0] = 0.0
#                 angle_map[i, j, 1] = 0.0
#             else:
#                 # 归一化法向量
#                 angle_map[i, j, 0] = normal_x[i, j] / n_xy_norm
#                 angle_map[i, j, 1] = normal_y[i, j] / n_xy_norm

#     # 拼接为4通道
#     encoded_input = np.concatenate((normal_z[:, :, None], context_map[:, :, None], angle_map[:, :, :2]), axis=2)  # [H, W, 4]
#     return encoded_input

def compute_terrain_direction(nx, ny, nz, cos_theta, sin_theta):
    """
    根据地形法向量调整运动方向，确保方向在可达范围内
    
    Args:
        nx: 法向量x分量 (标量或数组)
        ny: 法向量y分量 (标量或数组)  
        nz: 法向量z分量 (标量或数组)
        cos_theta: 目标方向角度的余弦值 (标量或数组)
        sin_theta: 目标方向角度的正弦值 (标量或数组)
    
    Returns:
        np.ndarray: 调整后的[cos_theta, sin_theta]
    """
    # 转换为numpy数组以支持向量化操作
    nx = np.asarray(nx)
    ny = np.asarray(ny)
    nz = np.asarray(nz)
    cos_theta = np.asarray(cos_theta)
    sin_theta = np.asarray(sin_theta)
    
    # 如果输入为空，直接返回
    if nx.size == 0 or ny.size == 0 or nz.size == 0:
        return np.array([cos_theta, sin_theta])
    
    # 从cos和sin计算当前角度
    current_theta = np.arctan2(sin_theta, cos_theta)
    
    # 地形约束参数
    h = 8           # 机器人高度
    min_edge = 10   # 最小边长约束
    max_edge = 20   # 最大边长约束
    
    # 计算地形倾斜角度对应的tan值
    # 这里假设地形倾斜角度与法向量相关
    terrain_slope = np.arctan2(np.sqrt(nx**2 + ny**2), np.abs(nz))
    
    # 计算约束参数b
    b_vals = h * np.tan(terrain_slope)
    
    # 根据b值分类地形约束类型
    mask_reachable = b_vals < min_edge  # 完全可达
    mask_partial = (b_vals >= min_edge) & (b_vals < max_edge)  # 部分可达
    mask_complex = (b_vals >= max_edge) & (b_vals < np.sqrt(max_edge**2 + min_edge**2))  # 复杂约束
    mask_unreachable = b_vals >= np.sqrt(max_edge**2 + min_edge**2)  # 完全不可达
    
    # 初始化调整后的角度
    adjusted_theta = current_theta.copy()
    
    # 处理完全不可达区域：设置为零方向
    if np.any(mask_unreachable):
        adjusted_theta[mask_unreachable] = 0.0
    
    # # 处理完全可达区域：应用地形约束但不限制方向
    # if np.any(mask_reachable):
    #     reachable_indices = np.where(mask_reachable)[0]
    #     reachable_theta = current_theta[mask_reachable]
    #     reachable_nx = nx[mask_reachable]
    #     reachable_ny = ny[mask_reachable]
    #     reachable_nz = nz[mask_reachable]
        
    #     # 计算地形法向量的影响：从局部坐标系转换到全局坐标系
    #     normal_proj_angles = np.arctan2(reachable_ny, reachable_nx)
        
    #     # 对于完全可达区域，我们仍然要考虑地形倾斜的影响
    #     # 使用简化的地形校正：theta_global = theta_local + terrain_correction
    #     terrain_correction = normal_proj_angles * 0.3  # 地形影响权重可调
        
    #     # 应用地形校正
    #     corrected_theta = reachable_theta + terrain_correction
        
    #     # 标准化角度到[-π, π]
    #     corrected_theta = np.where(corrected_theta > np.pi, corrected_theta - 2*np.pi, corrected_theta)
    #     corrected_theta = np.where(corrected_theta < -np.pi, corrected_theta + 2*np.pi, corrected_theta)
        
    #     # 更新调整后的角度
    #     if adjusted_theta.ndim == 0:  # 标量情况
    #         adjusted_theta = corrected_theta[0] if len(corrected_theta) > 0 else adjusted_theta
    #     else:  # 数组情况
    #         adjusted_theta[mask_reachable] = corrected_theta
    
    # 处理部分可达区域
    if np.any(mask_partial):
        # print(f"进入部分可达区域处理，共{np.sum(mask_partial)}个点")
        partial_b = b_vals[mask_partial]
        partial_theta = current_theta[mask_partial]
        partial_nx = nx[mask_partial]
        partial_ny = ny[mask_partial]
        partial_nz = nz[mask_partial]
        
        # print(f"部分可达区域 b值范围: {partial_b.min():.2f} - {partial_b.max():.2f}")
        # print(f"部分可达区域 角度范围: {partial_theta.min():.3f} - {partial_theta.max():.3f} rad")
        
        # 计算约束边界角度
        s1_vals = np.arcsin(min_edge / partial_b)
        e1_vals = np.pi - s1_vals
        s2_vals = -s1_vals
        e2_vals = -np.pi + s1_vals
        
        # print(f"局部约束边界 s1: {s1_vals.min():.3f}-{s1_vals.max():.3f}, e1: {e1_vals.min():.3f}-{e1_vals.max():.3f}")
        # print(f"局部约束边界 s2: {s2_vals.min():.3f}-{s2_vals.max():.3f}, e2: {e2_vals.min():.3f}-{e2_vals.max():.3f}")
        
        # 考虑地形法向量的影响：从局部坐标系转换到全局坐标系
        # theta_global = arctan2(ny,nx) + arctan(nz*tan(theta_local))
        normal_proj_angles = np.arctan2(partial_ny, partial_nx)
        # print(f"法向量投影角度范围: {normal_proj_angles.min():.3f} - {normal_proj_angles.max():.3f} rad")
        
        # 计算全局坐标系下的边界参数，注意处理arctan的角度范围
        # 对于 arctan(nz*tan(theta_local))，需要根据原始角度的象限来调整结果
        def safe_arctan_transform(nz_vals, theta_local_vals):
            """安全的arctan变换，考虑角度的正确象限"""
            tan_vals = np.tan(theta_local_vals)
            arctan_result = np.arctan(nz_vals * tan_vals)
            
            # 如果原始角度在第二或第三象限（cos < 0），需要调整arctan结果
            # 第二象限：θ ∈ (π/2, π) => cos < 0, sin > 0
            # 第三象限：θ ∈ (-π, -π/2) => cos < 0, sin < 0
            cos_local = np.cos(theta_local_vals)
            sin_local = np.sin(theta_local_vals)
            
            # 调整第二象限的角度：arctan结果需要加π
            second_quadrant = (cos_local < 0) & (sin_local > 0)
            arctan_result = np.where(second_quadrant, arctan_result + np.pi, arctan_result)
            
            # 调整第三象限的角度：arctan结果需要减π
            third_quadrant = (cos_local < 0) & (sin_local < 0)
            arctan_result = np.where(third_quadrant, arctan_result - np.pi, arctan_result)
            
            return arctan_result
        
        # 计算全局坐标系下的边界参数
        s1_transform = safe_arctan_transform(partial_nz, s1_vals)
        e1_transform = safe_arctan_transform(partial_nz, e1_vals)
        s2_transform = safe_arctan_transform(partial_nz, s2_vals)
        e2_transform = safe_arctan_transform(partial_nz, e2_vals)
        
        s1_global = normal_proj_angles + s1_transform
        e1_global = normal_proj_angles + e1_transform
        s2_global = normal_proj_angles + s2_transform
        e2_global = normal_proj_angles + e2_transform
        
        # 标准化角度到[-π, π]
        s1_global = np.where(s1_global > np.pi, s1_global - 2*np.pi, s1_global)
        s1_global = np.where(s1_global < -np.pi, s1_global + 2*np.pi, s1_global)
        e1_global = np.where(e1_global > np.pi, e1_global - 2*np.pi, e1_global)
        e1_global = np.where(e1_global < -np.pi, e1_global + 2*np.pi, e1_global)
        s2_global = np.where(s2_global > np.pi, s2_global - 2*np.pi, s2_global)
        s2_global = np.where(s2_global < -np.pi, s2_global + 2*np.pi, s2_global)
        e2_global = np.where(e2_global > np.pi, e2_global - 2*np.pi, e2_global)
        e2_global = np.where(e2_global < -np.pi, e2_global + 2*np.pi, e2_global)
        
        # print(f"全局约束边界 s1: {s1_global.min():.3f}-{s1_global.max():.3f}, e1: {e1_global.min():.3f}-{e1_global.max():.3f}")
        # print(f"全局约束边界 s2: {s2_global.min():.3f}-{s2_global.max():.3f}, e2: {e2_global.min():.3f}-{e2_global.max():.3f}")
        
        # 检查是否在不可达区域（现在使用全局坐标系下的角度）
        unreachable_mask = ((partial_theta > s1_global) & (partial_theta < e1_global)) | \
                          ((partial_theta > e2_global) & (partial_theta < s2_global))
        
        # print(f"不可达角度检查: {np.sum(unreachable_mask)} 个角度被判定为不可达")
        
        # 对不可达角度进行调整：找到最近的可达边界
        if np.any(unreachable_mask):
            # print(f"开始修正不可达角度...")
            unreachable_indices = np.where(unreachable_mask)[0]
            partial_indices = np.where(mask_partial)[0]  # 获取部分可达区域在原数组中的索引
            
            for idx in unreachable_indices:
                current_angle = partial_theta[idx]
                # print(f"  修正角度 {idx}: 当前角度={current_angle:.3f} rad")
                # 找到最近的边界（全局坐标系下）
                boundaries = [s1_global[idx], e1_global[idx], s2_global[idx], e2_global[idx]]
                # print(f"    边界值: s1={s1_global[idx]:.3f}, e1={e1_global[idx]:.3f}, s2={s2_global[idx]:.3f}, e2={e2_global[idx]:.3f}")
                closest_boundary = min(boundaries, key=lambda x: abs(current_angle - x))
                # print(f"    最近边界: {closest_boundary:.3f} rad")
                # 调整角度 - 使用原数组中的正确索引
                original_idx = partial_indices[idx]
                if adjusted_theta.ndim == 0:  # 标量情况
                    adjusted_theta = closest_boundary
                else:  # 数组情况
                    adjusted_theta[original_idx] = closest_boundary
                # print(f"    角度已修正为: {closest_boundary:.3f} rad，原索引={original_idx}")
        # else:
            # print("没有不可达角度需要修正")
    
    # 处理复杂约束区域
    if np.any(mask_complex):
        complex_b = b_vals[mask_complex]
        complex_theta = current_theta[mask_complex]
        complex_nx = nx[mask_complex]
        complex_ny = ny[mask_complex]
        complex_nz = nz[mask_complex]
        
        # 计算复杂约束的边界参数
        r1_vals = np.arcsin(min_edge / complex_b)
        r2_vals = np.arccos(max_edge / complex_b)
        
        # 计算所有边界角度
        s1_vals = -r2_vals
        e1_vals = r2_vals
        s2_vals = r1_vals
        e2_vals = np.pi - r1_vals
        p1_vals = np.pi - r2_vals
        p2_vals = -np.pi + r2_vals
        s3_vals = -np.pi + r1_vals
        e3_vals = -r1_vals
        
        # 考虑地形法向量的影响：从局部坐标系转换到全局坐标系
        # theta_global = arctan2(ny,nx) + arctan(nz*tan(theta_local))
        normal_proj_angles = np.arctan2(complex_ny, complex_nx)
        
        # 计算全局坐标系下的边界参数，注意处理arctan的角度范围
        def safe_arctan_transform(nz_vals, theta_local_vals):
            """安全的arctan变换，考虑角度的正确象限"""
            tan_vals = np.tan(theta_local_vals)
            arctan_result = np.arctan(nz_vals * tan_vals)
            
            # 如果原始角度在第二或第三象限（cos < 0），需要调整arctan结果
            cos_local = np.cos(theta_local_vals)
            sin_local = np.sin(theta_local_vals)
            
            # 调整第二象限的角度：arctan结果需要加π
            second_quadrant = (cos_local < 0) & (sin_local > 0)
            arctan_result = np.where(second_quadrant, arctan_result + np.pi, arctan_result)
            
            # 调整第三象限的角度：arctan结果需要减π
            third_quadrant = (cos_local < 0) & (sin_local < 0)
            arctan_result = np.where(third_quadrant, arctan_result - np.pi, arctan_result)
            
            return arctan_result
        
        # 计算全局坐标系下的边界参数
        s1_transform = safe_arctan_transform(complex_nz, s1_vals)
        e1_transform = safe_arctan_transform(complex_nz, e1_vals)
        s2_transform = safe_arctan_transform(complex_nz, s2_vals)
        e2_transform = safe_arctan_transform(complex_nz, e2_vals)
        p1_transform = safe_arctan_transform(complex_nz, p1_vals)
        p2_transform = safe_arctan_transform(complex_nz, p2_vals)
        s3_transform = safe_arctan_transform(complex_nz, s3_vals)
        e3_transform = safe_arctan_transform(complex_nz, e3_vals)
        
        s1_global = normal_proj_angles + s1_transform
        e1_global = normal_proj_angles + e1_transform
        s2_global = normal_proj_angles + s2_transform
        e2_global = normal_proj_angles + e2_transform
        p1_global = normal_proj_angles + p1_transform
        p2_global = normal_proj_angles + p2_transform
        s3_global = normal_proj_angles + s3_transform
        e3_global = normal_proj_angles + e3_transform
        
        # 标准化所有角度到[-π, π]
        def normalize_angle(angle):
            angle = np.where(angle > np.pi, angle - 2*np.pi, angle)
            angle = np.where(angle < -np.pi, angle + 2*np.pi, angle)
            return angle
        
        s1_global = normalize_angle(s1_global)
        e1_global = normalize_angle(e1_global)
        s2_global = normalize_angle(s2_global)
        e2_global = normalize_angle(e2_global)
        p1_global = normalize_angle(p1_global)
        p2_global = normalize_angle(p2_global)
        s3_global = normalize_angle(s3_global)
        e3_global = normalize_angle(e3_global)
        
        # 检查是否在不可达区域（复杂约束的多个区间，使用全局坐标系）
        unreachable_mask = (
            ((complex_theta > s1_global) & (complex_theta < e1_global)) |
            ((complex_theta > s2_global) & (complex_theta < e2_global)) |
            ((complex_theta > s3_global) & (complex_theta < e3_global)) |
            (complex_theta < p2_global) |
            (complex_theta > p1_global)
        )
        
        # 对不可达角度进行调整：找到最近的可达边界
        if np.any(unreachable_mask):
            unreachable_indices = np.where(unreachable_mask)[0]
            complex_indices = np.where(mask_complex)[0]  # 获取复杂约束区域在原数组中的索引
            
            for idx in unreachable_indices:
                current_angle = complex_theta[idx]
                # 找到最近的边界（包括所有复杂约束的边界，全局坐标系下）
                boundaries = [
                    s1_global[idx], e1_global[idx], s2_global[idx], e2_global[idx],
                    s3_global[idx], e3_global[idx], p1_global[idx], p2_global[idx]
                ]
                closest_boundary = min(boundaries, key=lambda x: abs(current_angle - x))
                # 调整角度 - 使用原数组中的正确索引
                original_idx = complex_indices[idx]
                if adjusted_theta.ndim == 0:  # 标量情况
                    adjusted_theta = closest_boundary
                else:  # 数组情况
                    adjusted_theta[original_idx] = closest_boundary
    
    # 将调整后的角度转换回cos和sin
    adjusted_cos = np.cos(adjusted_theta)
    adjusted_sin = np.sin(adjusted_theta)
    
    return np.array([adjusted_cos, adjusted_sin])

def compute_map_yaw_bins(normal_x, normal_y, normal_z, yaw_bins=18):
    """
    计算地图上每个点的分箱角度是否会倾覆（PyTorch版本，高效批量处理）
    
    基于相同的物理约束来判断每个yaw_bin角度是否会导致倾覆，
    不会倾覆的角度分箱标为1，会倾覆的标为0。
    
    Args:
        normal_x: 地形法向量x分量 (H, W) 
        normal_y: 地形法向量y分量 (H, W)
        normal_z: 地形法向量z分量 (H, W)
        yaw_bins: 朝向角度的分箱数量，默认为18

    Returns:
        torch.Tensor: 每个点每个角度分箱的倾覆状态，形状为(H, W, yaw_bins)，1表示不会倾覆，0表示会倾覆
    """
    # 确保输入是torch张量
    if not isinstance(normal_x, torch.Tensor):
        normal_x = torch.tensor(normal_x, dtype=torch.float32)
        normal_y = torch.tensor(normal_y, dtype=torch.float32) 
        normal_z = torch.tensor(normal_z, dtype=torch.float32)
    
    device = normal_x.device
    H, W = normal_x.shape
    
    # 地形约束参数
    h = 8.0         # 机器人高度
    min_edge = 10.0 # 最小边长约束
    max_edge = 20.0 # 最大边长约束
    
    # 计算地形倾斜角度对应的tan值
    terrain_slope = torch.arctan2(torch.sqrt(normal_x**2 + normal_y**2), torch.abs(normal_z))
    
    # 计算约束参数b
    b_vals = h * torch.tan(terrain_slope)
    
    # 根据b值分类地形约束类型
    mask_reachable = b_vals < min_edge  # 完全可达
    mask_partial = (b_vals >= min_edge) & (b_vals < max_edge)  # 部分可达
    mask_complex = (b_vals >= max_edge) & (b_vals < torch.sqrt(torch.tensor(max_edge**2 + min_edge**2, device=device)))  # 复杂约束
    mask_unreachable = b_vals >= torch.sqrt(torch.tensor(max_edge**2 + min_edge**2, device=device))  # 完全不可达
    
    # 初始化结果数组：所有角度分箱都标记为不会倾覆(1)
    yaw_stability = torch.ones((H, W, yaw_bins), dtype=torch.float32, device=device)
    
    # 定义角度分箱的中心角度：从-π到π均匀分布
    bin_angles = torch.linspace(-torch.pi, torch.pi, yaw_bins + 1, device=device)[:-1]  # 去掉最后一个
    
    def safe_arctan_transform(nz_vals, theta_local_vals):
        """安全的arctan变换，考虑角度的正确象限"""
        tan_vals = torch.tan(theta_local_vals)
        arctan_result = torch.arctan(nz_vals * tan_vals)
        
        # 如果原始角度在第二或第三象限（cos < 0），需要调整arctan结果
        cos_local = torch.cos(theta_local_vals)
        sin_local = torch.sin(theta_local_vals)
        
        # 调整第二象限的角度：arctan结果需要加π
        second_quadrant = (cos_local < 0) & (sin_local > 0)
        arctan_result = torch.where(second_quadrant, arctan_result + torch.pi, arctan_result)
        
        # 调整第三象限的角度：arctan结果需要减π
        third_quadrant = (cos_local < 0) & (sin_local < 0)
        arctan_result = torch.where(third_quadrant, arctan_result - torch.pi, arctan_result)
        
        return arctan_result

    def normalize_angle(angle):
        """标准化角度到[-π, π]"""
        angle = torch.where(angle > torch.pi, angle - 2*torch.pi, angle)
        angle = torch.where(angle < -torch.pi, angle + 2*torch.pi, angle)
        return angle
    
    def check_angle_in_range_vectorized(angles, starts, ends):
        """向量化检查角度是否在范围内"""
        # angles: (yaw_bins,), starts: (N,), ends: (N,)
        # 返回: (N, yaw_bins) 布尔张量
        angles = angles.unsqueeze(0)  # (1, yaw_bins)
        starts = starts.unsqueeze(1)  # (N, 1)
        ends = ends.unsqueeze(1)      # (N, 1)
        
        angles = normalize_angle(angles)
        starts = normalize_angle(starts)
        ends = normalize_angle(ends)
        
        # 正常情况：start <= end
        normal_case = starts <= ends
        in_range_normal = (angles >= starts) & (angles <= ends) & normal_case
        
        # 跨越边界情况：start > end
        cross_boundary = starts > ends
        in_range_cross = ((angles >= starts) | (angles <= ends)) & cross_boundary
        
        return in_range_normal | in_range_cross
    
    # 处理完全不可达区域：所有角度都标记为会倾覆(0)
    yaw_stability[mask_unreachable] = 0.0
    
    # 处理部分可达区域 - 向量化处理
    if torch.any(mask_partial):
        # 获取部分可达区域的坐标和值
        partial_indices = torch.where(mask_partial)
        partial_b = b_vals[mask_partial]
        partial_nx = normal_x[mask_partial]
        partial_ny = normal_y[mask_partial]
        partial_nz = normal_z[mask_partial]
        
        # 批量计算约束边界角度（局部坐标系）
        s1_vals = torch.arcsin(min_edge / partial_b)
        e1_vals = torch.pi - s1_vals
        s2_vals = -s1_vals
        e2_vals = -torch.pi + s1_vals
        
        # 考虑地形法向量的影响：从局部坐标系转换到全局坐标系
        normal_proj_angles = torch.arctan2(partial_ny, partial_nx)
        
        # 计算全局坐标系下的边界参数
        s1_transforms = safe_arctan_transform(partial_nz, s1_vals)
        e1_transforms = safe_arctan_transform(partial_nz, e1_vals)
        s2_transforms = safe_arctan_transform(partial_nz, s2_vals)
        e2_transforms = safe_arctan_transform(partial_nz, e2_vals)
        
        s1_globals = normalize_angle(normal_proj_angles + s1_transforms)
        e1_globals = normalize_angle(normal_proj_angles + e1_transforms)
        s2_globals = normalize_angle(normal_proj_angles + s2_transforms)
        e2_globals = normalize_angle(normal_proj_angles + e2_transforms)
        
        # 向量化检查每个角度分箱是否在不可达区域
        in_unreachable_region1 = check_angle_in_range_vectorized(bin_angles, s1_globals, e1_globals)  # (N, yaw_bins)
        in_unreachable_region2 = check_angle_in_range_vectorized(bin_angles, e2_globals, s2_globals)  # (N, yaw_bins)
        
        unreachable_mask = in_unreachable_region1 | in_unreachable_region2  # (N, yaw_bins)
        
        # 更新yaw_stability
        for idx, (i, j) in enumerate(zip(partial_indices[0], partial_indices[1])):
            yaw_stability[i, j, unreachable_mask[idx]] = 0.0
    
    # 处理复杂约束区域 - 向量化处理
    if torch.any(mask_complex):
        # 获取复杂约束区域的坐标和值
        complex_indices = torch.where(mask_complex)
        complex_b = b_vals[mask_complex]
        complex_nx = normal_x[mask_complex]
        complex_ny = normal_y[mask_complex]
        complex_nz = normal_z[mask_complex]
        
        # 批量计算复杂约束的边界参数
        r1_vals = torch.arcsin(min_edge / complex_b)
        r2_vals = torch.arccos(max_edge / complex_b)
        
        # 计算所有边界角度（局部坐标系）
        s1_vals = -r2_vals
        e1_vals = r2_vals
        s2_vals = r1_vals
        e2_vals = torch.pi - r1_vals
        p1_vals = torch.pi - r2_vals
        p2_vals = -torch.pi + r2_vals
        s3_vals = -torch.pi + r1_vals
        e3_vals = -r1_vals
        
        # 考虑地形法向量的影响：从局部坐标系转换到全局坐标系
        normal_proj_angles = torch.arctan2(complex_ny, complex_nx)
        
        # 计算全局坐标系下的边界参数
        s1_transforms = safe_arctan_transform(complex_nz, s1_vals)
        e1_transforms = safe_arctan_transform(complex_nz, e1_vals)
        s2_transforms = safe_arctan_transform(complex_nz, s2_vals)
        e2_transforms = safe_arctan_transform(complex_nz, e2_vals)
        p1_transforms = safe_arctan_transform(complex_nz, p1_vals)
        p2_transforms = safe_arctan_transform(complex_nz, p2_vals)
        s3_transforms = safe_arctan_transform(complex_nz, s3_vals)
        e3_transforms = safe_arctan_transform(complex_nz, e3_vals)
        
        s1_globals = normalize_angle(normal_proj_angles + s1_transforms)
        e1_globals = normalize_angle(normal_proj_angles + e1_transforms)
        s2_globals = normalize_angle(normal_proj_angles + s2_transforms)
        e2_globals = normalize_angle(normal_proj_angles + e2_transforms)
        p1_globals = normalize_angle(normal_proj_angles + p1_transforms)
        p2_globals = normalize_angle(normal_proj_angles + p2_transforms)
        s3_globals = normalize_angle(normal_proj_angles + s3_transforms)
        e3_globals = normalize_angle(normal_proj_angles + e3_transforms)
        
        # 向量化检查每个角度分箱是否在不可达区域
        in_unreachable1 = check_angle_in_range_vectorized(bin_angles, s1_globals, e1_globals)
        in_unreachable2 = check_angle_in_range_vectorized(bin_angles, s2_globals, e2_globals)
        in_unreachable3 = check_angle_in_range_vectorized(bin_angles, s3_globals, e3_globals)
        
        # 处理p1和p2边界（单侧边界）
        bin_angles_expanded = bin_angles.unsqueeze(0)  # (1, yaw_bins)
        p1_expanded = p1_globals.unsqueeze(1)  # (N, 1)
        p2_expanded = p2_globals.unsqueeze(1)  # (N, 1)
        
        in_unreachable_p1 = bin_angles_expanded > p1_expanded
        in_unreachable_p2 = bin_angles_expanded < p2_expanded
        
        unreachable_mask = (in_unreachable1 | in_unreachable2 | in_unreachable3 | 
                           in_unreachable_p1 | in_unreachable_p2)  # (N, yaw_bins)
        
        # 更新yaw_stability
        for idx, (i, j) in enumerate(zip(complex_indices[0], complex_indices[1])):
            yaw_stability[i, j, unreachable_mask[idx]] = 0.0

    return yaw_stability

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_cost_map_from_yaw_stability(
    yaw_stability,            # torch.Tensor or np.ndarray, shape (H,W,Y), 1 safe / 0 occupied
    voxel_size_xy=0.1,
    yaw_weight=0.2,
    d_safe=0.0,
    kalpa=0.6,
    use_scipy=True,
    return_esdf=False,
):
    """
    快速版本：优先使用 scipy.ndimage.distance_transform_edt（C 实现）。
    对 yaw 周期性做三倍平铺处理以正确计算环绕距离。
    """
    # --- 标准化输入 ---
    input_is_torch = isinstance(yaw_stability, torch.Tensor)
    if input_is_torch:
        device = yaw_stability.device
        ys = yaw_stability.detach().to('cpu').numpy()
    else:
        ys = np.asarray(yaw_stability)

    H, W, Y = ys.shape
    occupied = (ys <= 0.5)   # True 表示占据（不安全 / capsized）

    # 快速路径：scipy 的 EDT（推荐）
    if use_scipy:
        try:
            from scipy.ndimage import distance_transform_edt
            # yaw 每格对应的弧度
            delta_theta = 2.0 * math.pi / float(Y)
            sampling = (voxel_size_xy, voxel_size_xy, yaw_weight * delta_theta)

            # 三倍平铺以考虑周期性
            occupied_tiled = np.concatenate([occupied, occupied, occupied], axis=2)  # (H,W,3Y)
            # distance_transform_edt 计算的是到 False (0) 的距离，如果我们要距离到占据点：
            # 将 occupied_tiled (True占据) 转成 inverted map (False 为占据), 然后 edt gives distance to nearest True in inverted => distance to nearest occupied
            # 更直接：distance_transform_edt(~occupied_tiled, sampling=...)
            inv = ~occupied_tiled
            dist_tiled = distance_transform_edt(inv, sampling=sampling)  # (H,W,3Y)   dtype: float64

            # 取中间段 (Y : 2Y)
            esdf = dist_tiled[:, :, Y:2*Y].astype(np.float32)  # (H,W,Y)

            # 计算 cost
            z = (-(esdf - d_safe) / (kalpa + 1e-12))
            z = np.clip(z, -50.0, 50.0)
            costs = 1.0 / (1.0 + np.exp(-z))  # sigmoid

            if input_is_torch:
                costs_t = torch.from_numpy(costs).to(device=device, dtype=torch.float32)
                if return_esdf:
                    esdf_t = torch.from_numpy(esdf).to(device=device, dtype=torch.float32)
                    return costs_t, esdf_t
                return costs_t
            else:
                if return_esdf:
                    return costs, esdf
                return costs

        except Exception as e:
            # 如果 scipy 不可用或调用失败，退回下面的 PyTorch 近似实现（告警）
            print("WARNING: scipy EDT 快速路径失败或不可用，退回 PyTorch 实现。错误：", e)
            use_scipy = False

    # --- 退回：改良的 PyTorch 分块最近邻（慢但健壮） ---
    # 此实现是你原来方法的优化版：尽量减少 Python loop, 仍使用 torch.cdist 分块
    if input_is_torch:
        ys_t = yaw_stability.to(dtype=torch.float32)
        device = ys_t.device
    else:
        ys_t = torch.from_numpy(ys.astype(np.float32))

    occupied_mask = (ys_t <= 0.5)
    occ_idx = torch.nonzero(occupied_mask, as_tuple=False)  # (M,3)
    M = occ_idx.shape[0]
    N = H * W * Y

    if M == 0:
        # 没有占据点 -> 大距离
        max_xy = math.hypot(H * voxel_size_xy, W * voxel_size_xy)
        max_yaw = math.pi * yaw_weight
        max_dist = math.sqrt(max_xy**2 + max_yaw**2) + 1.0
        esdf_map = torch.ones((H, W, Y), dtype=torch.float32) * float(max_dist)
        z = (-(esdf_map - d_safe) / (kalpa + 1e-12))
        costs = torch.sigmoid(torch.clamp(z, min=-50.0, max=50.0))
        if input_is_torch:
            if return_esdf:
                return costs, esdf_map
            return costs
        else:
            c_np = costs.cpu().numpy()
            if return_esdf:
                return c_np, esdf_map.cpu().numpy()
            return c_np

    # 预计算 yaw 坐标
    ks = torch.arange(Y, dtype=torch.float32)
    theta_k = (ks + 0.5) * (2.0 * math.pi / float(Y))
    yaw_cos = torch.cos(theta_k) * yaw_weight
    yaw_sin = torch.sin(theta_k) * yaw_weight

    occ_i = occ_idx[:,0].to(dtype=torch.float32)
    occ_j = occ_idx[:,1].to(dtype=torch.float32)
    occ_k = occ_idx[:,2].to(dtype=torch.long)
    occ_x = occ_i * voxel_size_xy
    occ_y = occ_j * voxel_size_xy
    occ_yaw_x = yaw_cos[occ_k]
    occ_yaw_y = yaw_sin[occ_k]
    occ_coords = torch.stack([occ_x, occ_y, occ_yaw_x, occ_yaw_y], dim=1).to(dtype=torch.float32)

    # 分块遍历所有 N 点，按 chunk_size 控制显存
    chunk_size = 200_000  # 可调整
    min_dists = torch.empty(N, dtype=torch.float32)
    device = occ_coords.device
    # 如果在GPU上，把 occ_coords 放GPU加速
    if torch.cuda.is_available():
        occ_coords = occ_coords.cuda()
        device = occ_coords.device

    def idx_to_coords_block(start, end):
        idxs = torch.arange(start, end, device=device, dtype=torch.long)
        i = (idxs // (W * Y)).to(torch.float32)
        rem = idxs % (W * Y)
        j = (rem // Y).to(torch.float32)
        k = (rem % Y).to(torch.long)
        x = i * voxel_size_xy
        y = j * voxel_size_xy
        yaw_x = yaw_cos[k].to(device=device)
        yaw_y = yaw_sin[k].to(device=device)
        coords_block = torch.stack([x, y, yaw_x, yaw_y], dim=1)
        return coords_block

    start = 0
    while start < N:
        end = min(N, start + chunk_size)
        coords_block = idx_to_coords_block(start, end)
        # 把 coords_block 与 occ_coords 放在同设备
        if coords_block.device != occ_coords.device:
            coords_block = coords_block.to(device=occ_coords.device)
        # 计算 cdist 并取最小
        dists = torch.cdist(coords_block, occ_coords)  # (B, M)
        min_block, _ = torch.min(dists, dim=1)
        min_dists[start:end] = min_block.cpu()
        start = end

    esdf_map = min_dists.view(H, W, Y).to(dtype=torch.float32)
    z = (-(esdf_map - d_safe) / (kalpa + 1e-12))
    costs_t = torch.sigmoid(torch.clamp(z, min=-50.0, max=50.0))

    if input_is_torch:
        if return_esdf:
            return costs_t.to(device=device), esdf_map.to(device=device)
        return costs_t.to(device=device)
    else:
        c_np = costs_t.cpu().numpy()
        if return_esdf:
            return c_np, esdf_map.cpu().numpy()
        return c_np


class UnevenPathDataLoader(Dataset):
    """
    UnevenPathDataLoader: 不平坦地面的路径数据加载器
    
    【核心功能】
    从不平坦地面的路径规划数据集中加载训练样本，用于训练模型在复杂地形下的路径规划能力。
    
    【数据集特点】
    1. 地图类型：包含不平坦地面的复杂环境
    2. 地图尺寸：10m×10m，分辨率0.01m
    3. 路径类型：包含成功和失败的路径规划任务
    4. 数据格式：Pickle文件，包含地图和路径信息:
    (1)地图文件：`map.p`
    ```python
    {
        'tensor': np.array,          # [H, W, 4] 四通道张量
        'bounds': (min_x, max_x, min_y, max_y),
        'resolution': 0.2,           # 栅格分辨率
        'map_name': 'desert',        # 地图名称
        'channels': ['elevation', 'normal_x', 'normal_y', 'normal_z'],
        'shape': (height, width, 4)
    }
    ```
    例如：
    ```python
    {'tensor': array([
        [[ 1.91332285e+00,  1.66046888e-01, -2.47567687e-02, 9.85807061e-01],
         [ 1.89383935e+00,  1.66046888e-01, -2.47567687e-02, 9.85807061e-01],
         [ 1.87698874e+00,  1.80939287e-01, -3.28837261e-02, 9.82944369e-01],
         ...,
         [ 7.89104671e-01,  7.29798600e-02, -3.43800820e-02, 9.96740639e-01],
         [ 7.80242312e-01,  9.87786874e-02, -4.03737016e-02, 9.94290054e-01],
         [ 7.70666865e-01,  9.87786874e-02, -4.03737016e-02, 9.94290054e-01]],

        ...,

        [[ 1.12184868e+00, -3.71789820e-02,  4.94028963e-02, 9.98086691e-01],
         [ 1.12459070e+00, -3.71789820e-02,  4.94028963e-02, 9.98086691e-01],
         [ 1.12810575e+00, -5.22360252e-03,  3.20198573e-02, 9.99473572e-01],
         ...,  
         [ 8.72300527e-01,  9.26663429e-02, -1.00500155e-02, 9.95646477e-01],
         [ 8.60928348e-01,  1.48325890e-01,  3.07161896e-03, 9.88933742e-01],
         [ 8.48599134e-01,  1.48325890e-01,  3.07161896e-03, 9.88933742e-01]]], shape=(100, 100, 4)), 
      'bounds': (-5.0, 5.0, -5.0, 5.0), 
      'resolution': 0.1, 
      'map_name': 'desert', 
      'channels': ['elevation', 'normal_x', 'normal_y', 'normal_z'], 
      'shape': (100, 100, 4)}
    ```
    (2)轨迹文件：`path_{id}.p`
    ```python
    {
        'valid': True,               # 是否有效路径
        'path': np.array,            # [N, 3] 轨迹点 [x, y, yaw]
        'map_name': 'desert'         # 关联的地图名称
    }
    ```
    (3)目录格式：
    ```
    ├── desert/
        |── map.p
        ├── path_0.p
        ├── path_1.p
        └── ...
    ├── forest/
        ├── map.p
        ├── path_0.p
        ├── path_1.p
        └── ...
    └── ...
    ```
    
    【数据加载示例】
    ```python
    import pickle
    import numpy as np

    # 加载地图数据
    with open('map.p', 'rb') as f:
        map_data = pickle.load(f)

    tensor = map_data['tensor']  # [H, W, 4]
    elevation = tensor[:, :, 0]
    normal_x = tensor[:, :, 1]
    normal_y = tensor[:, :, 2]
    normal_z = tensor[:, :, 3]

    # 加载轨迹数据
    with open('path_0.p', 'rb') as f:
        path_data = pickle.load(f)

    trajectory = path_data['path']  # [N, 3]
    map_name = path_data['map_name']  # 'desert'
    ```

    """

    def __init__(self, env_list, dataFolder):
        self.num_env = len(env_list)
        self.env_list = env_list
        self.dataFolder = dataFolder
        self.env_index = {env_name: i for i, env_name in enumerate(env_list)}
        self.indexDict = []
        
        for env_name in env_list:
            env_path = osp.join(dataFolder, env_name)
            # 只计算path_*.p文件的数量
            path_files = [f for f in os.listdir(env_path) if f.startswith('path_') and f.endswith('.p')]
            for i in range(len(path_files)):
                self.indexDict.append((self.env_index[env_name], i))
        
        print(f"不平坦地面数据加载器初始化完成：{self.num_env}个环境，{len(self.indexDict)}个路径样本")
    
    def __len__(self):
        return len(self.indexDict)
    
    def __getitem__(self, idx):
        env_index, path_index = self.indexDict[idx]
        env_name = self.env_list[env_index]
        env_path = osp.join(self.dataFolder, env_name)
        map_file = osp.join(env_path, 'map.p')
        path_file = osp.join(env_path, f'path_{path_index}.p')
        
        # 1. 加载地图数据
        with open(map_file, 'rb') as f:
            map_data = pickle.load(f)
        map_tensor = map_data['tensor']  # [H, W, 4]
        elevation = map_tensor[:, :, 0]
        normal_x = map_tensor[:, :, 1]
        normal_y = map_tensor[:, :, 2]
        normal_z = map_tensor[:, :, 3]
        
        # 2. 加载路径数据
        with open(path_file, 'rb') as f:
            path_data = pickle.load(f)
        valid = path_data['valid']  # 是否有效路径
        if not valid:
            return None  # 如果路径无效，返回None    
        
        trajectory = path_data['path']  # [N+2, 3]
        
        # 3. 生成编码输入
        path = trajectory[:, :3]  # [N+2, 3]
        
        map_input = torch.concatenate((
            # torch.tensor(elevation, dtype=torch.float32).unsqueeze(0),  # [1, H, W]
            torch.tensor(normal_x, dtype=torch.float32).unsqueeze(0),  # [1, H, W]
            torch.tensor(normal_y, dtype=torch.float32).unsqueeze(0),  # [1, H, W]
            torch.tensor(normal_z, dtype=torch.float32).unsqueeze(0)   # [1, H, W]
        ), dim=0)

        # start_pose = torch.tensor([path[0, 0], path[0, 1], np.cos(path[0, 2]), np.sin(path[0, 2])], dtype=torch.float32)  # [4] 起点位姿 [x, y, cos(yaw), sin(yaw)]
        # goal_pose = torch.tensor([path[-1, 0], path[-1, 1], np.cos(path[-1, 2]), np.sin(path[-1, 2])], dtype=torch.float32)  # [4] 终点位姿 [x, y, cos(yaw), sin(yaw)]
        # pose_input = torch.stack((start_pose, goal_pose), dim=0)  # [2, 4] 起点和终点位姿
        
        encoded_input = get_encoder_input(
            np.abs(normal_z),        # 确保使用的法向量z分量为正值
            goal_state=path[-1, :],  # 终点位姿
            start_state=path[0, :],  # 起点位姿
            normal_x=normal_x, 
            normal_y=normal_y
        )
        
        # encoded_input = get_encoder_input(
        #     normal_z, 
        #     goal_pos=path[-1, :2],  # 终点位置
        #     start_pos=path[0, :2],  # 起点位置
        #     normal_x=normal_x, 
        #     normal_y=normal_y
        # )
        
        # goal_index = geom2pix(path[-1, :2], res=map_data['resolution'], size=elevation.shape[:2])
        # start_index = geom2pix(path[0, :2], res=map_data['resolution'], size=elevation.shape[:2])
        
        # # 起点区域（使用start_index）
        # start_start_y = max(0, start_index[0] - receptive_field//2)
        # start_start_x = max(0, start_index[1] - receptive_field//2)
        # start_end_y = min(elevation.shape[0], start_index[0] + receptive_field//2)
        # start_end_x = min(elevation.shape[1], start_index[1] + receptive_field//2)
        
        # # 终点区域（使用goal_index）
        # goal_start_y = max(0, goal_index[0] - receptive_field//2)
        # goal_start_x = max(0, goal_index[1] - receptive_field//2)
        # goal_end_y = min(elevation.shape[0], goal_index[0] + receptive_field//2)
        # goal_end_x = min(elevation.shape[1], goal_index[1] + receptive_field//2)
        
        # # 构造编码输入
        # context_map = np.zeros(elevation.shape[:2])  # [H, W]
        # context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0  # 终点标记为1
        # context_map[start_start_y:start_end_y, start_start_x:start_end_x] = -1.0  # 起点标记为-1
        
        # # 构造θ=<nx,ny>->(cosθ, sinθ)的映射
        # angle_map = np.zeros((elevation.shape[0], elevation.shape[1], 2))  # [H, W, 2]
        # for i in range(elevation.shape[0]):
        #     for j in range(elevation.shape[1]):
        #         n_xy_norm = np.linalg.norm([normal_x[i, j], normal_y[i, j]])
                
        #         angle_map[i, j, 0] = normal_x[i, j] / n_xy_norm
        #         angle_map[i, j, 1] = normal_y[i, j] / n_xy_norm
        
        # # 拼接为4通道
        # encoded_input = np.concatenate((normal_z[:, :, None], context_map[:, :, None], angle_map[:, :, :2]), axis=2)  # [H, W, 4]
        
        # 4. 提取正/负样本锚点
        # 为每个轨迹点创建独立的正样本图层
        # trajectory = trajectory[1:-1, :]  # [N, 3]
        path_xy = trajectory[1:-1, :2]  # [N, 2] 去掉起点和终点
        num_trajectory_points = len(path_xy)
        
        # print(f"轨迹点数量: {num_trajectory_points}")
        # print(f"轨迹形状: {trajectory.shape}")
        
        # 为每个轨迹点找到对应的锚点（固定尺寸处理）
        positive_anchors_per_point = []  # 每个轨迹点对应的锚点列表
        for pos in path_xy:
            indices, = geom2pixMatpos(pos, res=res, size=map_tensor.shape[:2])
            current_anchors = list(set(indices))
            
            # 固定尺寸处理：确保每个轨迹点的正样本数量都是MAX_POSITIVE_ANCHORS
            if len(current_anchors) > MAX_POSITIVE_ANCHORS:
                # 如果超过最大值，随机采样到固定数量
                current_anchors = np.random.choice(current_anchors, size=MAX_POSITIVE_ANCHORS, replace=False).tolist()
            else:
                # 如果不足最大值，用-1填充到固定数量
                current_anchors.extend([-1] * (MAX_POSITIVE_ANCHORS - len(current_anchors)))
            
            positive_anchors_per_point.append(current_anchors)
        
        # 现在所有轨迹点的正样本数量都是MAX_POSITIVE_ANCHORS
        max_anchors = MAX_POSITIVE_ANCHORS
                       
        # 生成负样本：为每个轨迹点生成对应的负样本
        # 每个轨迹点的负样本应该是该点正样本的补集，而不是全体正样本的补集
        all_anchor_indices = set(range(len(hashTable)))
        negative_anchors_per_point = []
        # max_anchors = max(0, len(all_anchor_indices) - max_anchors)  # 负样本锚点数量
        
        for i in range(num_trajectory_points):
            # 获取当前轨迹点的正样本锚点
            current_positive_anchors = set([a for a in positive_anchors_per_point[i] if a != -1])
            
            # 当前轨迹点的负样本候选：全体锚点减去当前点的正样本
            available_negative_anchors = list(all_anchor_indices - current_positive_anchors)
            
            # 为当前轨迹点生成负样本
            if len(available_negative_anchors) >= max_anchors:
                neg_anchors = np.random.choice(available_negative_anchors, size=max_anchors, replace=False).tolist()
            else:
                neg_anchors = available_negative_anchors + [-1] * (max_anchors - len(available_negative_anchors))
            negative_anchors_per_point.append(neg_anchors)
        
        # 构建最终的锚点和标签张量
        all_positive = torch.tensor(positive_anchors_per_point)  # [num_trajectory_points, max_anchors]
        all_negative = torch.tensor(negative_anchors_per_point)  # [num_trajectory_points, max_anchors]
        
        anchor = torch.cat([all_positive, all_negative], dim=0)  # [2*num_trajectory_points, max_anchors]
        
        # 创建标签：前半部分为正样本(1)，后半部分为负样本(0)
        positive_labels = torch.ones_like(all_positive)
        negative_labels = torch.zeros_like(all_negative)
        labels = torch.cat([positive_labels, negative_labels], dim=0)  # [2*num_trajectory_points, max_anchors]
        
        # 将填充位置(-1)的标签设为-1，训练时忽略
        labels[anchor == -1] = -1
        
        # # 5. 计算yaw_bins倾覆状态
        yaw_stability = compute_map_yaw_bins(normal_x, normal_y, normal_z, yaw_bins=36)  # [H, W, 36]
        cost_map = generate_cost_map_from_yaw_stability(
            yaw_stability, 
            voxel_size_xy=0.1, 
            yaw_weight=2.1, 
            d_safe=0.15, 
            kalpa=0.1, 
            return_esdf=False
        )

        # 转换为PyTorch张量
        return {
            'map': torch.as_tensor(encoded_input, dtype=torch.float).permute(2, 0, 1),  # 地图：(C, H, W) - 转换为channels-first格式
            # 'map': torch.as_tensor(map_input, dtype=torch.float),  # 地图：(C, H, W) - map_input已经是正确的channels-first格式，无需permute
            # 'pose': torch.as_tensor(pose_input, dtype=torch.float),  # 起点和终点位姿：(2, 4)
            'anchor': anchor,  # 锚点索引：(N, M)
            'labels': labels,  # 锚点标签：(N, M)
            'trajectory': torch.as_tensor(trajectory, dtype=torch.float),  # 轨迹点：[N, 3]
            'yaw_stability': torch.as_tensor(yaw_stability, dtype=torch.float),  # yaw分箱倾覆状态：[H, W, 36]
            'cost_map': torch.as_tensor(cost_map, dtype=torch.float),  # 成本图：[H, W, yaw_bins]
            'elevation': torch.as_tensor(elevation, dtype=torch.float),  # 高程图：[H, W]
        }
