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
receptive_field = 18   # 感受野大小：每个锚点影响的像素范围 TODO
res = 0.1              # 地图分辨率：每像素代表0.05米的实际距离

# 【锚点网格系统构建】
# 将连续的地图空间离散化为24x24的锚点网格，用于Transformer的token化处理

# X轴锚点坐标：从6像素开始，每8像素一个锚点，转换为几何坐标: 
# [6, 14, 22, ..., 94] * res - 5 = [-4.4, -3.6, -2.8, ..., 4.4] 米
X = np.arange(boundary_offset, output_grid*anchor_spacing+boundary_offset, anchor_spacing)*res - 5

# Y轴锚点坐标：从6像素开始，每8像素一个锚点，转换为几何坐标: 
# [6, 14, 22, ..., 94] * res - 5 = [-4.4, -3.6, -2.8, ..., 4.4] 米
Y = np.arange(boundary_offset, output_grid*anchor_spacing+boundary_offset, anchor_spacing)*res - 5

# 创建2D网格：生成所有锚点的几何坐标
grid_2d = np.meshgrid(X, Y)  # 创建X-Y坐标网格

# 网格点重排：将2D网格转换为(N, 2)的点列表，N=144个锚点
grid_points = rearrange(grid_2d, 'c h w->(h w) c')  # 形状：(144, 2)

# 哈希表：锚点索引到像素坐标的映射表
hashTable = [(anchor_spacing*r+boundary_offset, anchor_spacing*c+boundary_offset)
             for c in range(output_grid) for r in range(output_grid)]

# 【网格系统说明】
# 1. 锚点分布：10x10=100个锚点均匀分布在地图上
# 2. 像素间距：每个锚点间隔10像素，对应1米的实际距离
# 3. 边界偏移：起始偏移5像素，确保锚点不在地图边缘
# 4. 坐标对应：每个锚点代表一个10x10像素的区域
# 5. 索引映射：通过hashTable实现1D索引到2D像素坐标的转换

def geom2pixMatpos(pos, res=0.1, size=(100, 100)):
    # 计算输入位置到所有锚点的距离
    distances = np.linalg.norm(grid_points - pos, axis=1)  # 形状：(100,)
    
    # 筛选距离阈值内的锚点索引
    indices = np.where(distances <= receptive_field * res * 0.4)  # 阈值：18 * 0.1 * 0.4 = 0.72米

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
    变长序列批处理整理函数（用于Transformer训练）
    
    【核心功能】
    将不同长度的序列样本整理成统一的批处理格式，通过填充机制处理变长数据。
    专门用于Transformer模型的训练，支持注意力掩码和长度信息的传递。
    
    【处理流程】
    1. 过滤无效样本：移除None值的样本
    2. 地图数据拼接：将所有地图在batch维度上拼接
    3. 序列数据填充：使用pad_sequence处理变长的anchor和labels
    4. 长度信息记录：保存每个样本的实际长度用于掩码
    5. 轨迹数据处理：将轨迹数据堆叠成批处理
    
    Args:
        batch (list): 批处理样本列表
            每个元素包含：
            - 'map': 地图张量，形状为(1, H, W, 4)
            - 'anchor': 锚点序列，形状为(num_layers, max_anchors)
            - 'labels': 标签序列，形状为(num_layers, max_anchors)
            - 'trajectory': 轨迹序列，形状为(N, 3)
    
    Returns:
        dict: 整理后的批处理数据
            - 'map': 批处理地图，形状为(B, H, W, 4)
            - 'anchor': 填充后的锚点序列，形状为(B, num_layers, max_anchors_global)
            - 'labels': 填充后的标签序列，形状为(B, num_layers, max_anchors_global)
            - 'length': 每个样本的实际长度，形状为(B,)
            - 'trajectory': 批处理轨迹序列，形状为(B, N, 3)
    """
    data = {}
    # 过滤有效样本：移除None值，确保数据完整性
    valid_batch = [batch_i for batch_i in batch if batch_i is not None]
    
    # 地图数据拼接：在batch维度上连接所有地图
    data['map'] = torch.cat([batch_i['map'] for batch_i in valid_batch], dim=0)
    
    # 找到所有样本中的最大锚点数量
    max_anchors_global = max(batch_i['anchor'].shape[1] for batch_i in valid_batch)
    
    # 填充锚点和标签到统一尺寸
    padded_anchors = []
    padded_labels = []
    
    for batch_i in valid_batch:
        anchor = batch_i['anchor']  # [num_layers, max_anchors_i]
        labels = batch_i['labels']  # [num_layers, max_anchors_i]
        
        current_max_anchors = anchor.shape[1]
        
        if current_max_anchors < max_anchors_global:
            # 需要填充到最大长度
            pad_size = max_anchors_global - current_max_anchors
            
            # 填充锚点（用-1填充）
            anchor_pad = torch.full((anchor.shape[0], pad_size), -1, dtype=anchor.dtype)
            anchor_padded = torch.cat([anchor, anchor_pad], dim=1)
            
            # 填充标签（用-1填充）
            labels_pad = torch.full((labels.shape[0], pad_size), -1, dtype=labels.dtype)
            labels_padded = torch.cat([labels, labels_pad], dim=1)
        else:
            anchor_padded = anchor
            labels_padded = labels
            
        padded_anchors.append(anchor_padded)
        padded_labels.append(labels_padded)
    
    # 现在所有张量都有相同的形状，可以安全地堆叠
    data['anchor'] = torch.stack(padded_anchors)
    data['labels'] = torch.stack(padded_labels)
    
    # 长度信息记录：每个样本的层数（实际上是固定的）
    data['length'] = torch.tensor([batch_i['anchor'].shape[0] for batch_i in valid_batch])
    
    # 处理轨迹数据：将所有轨迹数据堆叠成批处理
    if 'trajectory' in valid_batch[0]:
        data['trajectory'] = torch.stack([batch_i['trajectory'] for batch_i in valid_batch])
    
    return data

# def get_encoder_input(normal_z, goal_state, start_state, normal_x, normal_y):
#     # 构造编码输入[H, W, 6]
#     goal_pos = goal_state[:2]  # 终点位置 (x, y)
#     start_pos = start_state[:2]  # 起点位置 (x, y)
#     goal_angle = goal_state[2]  # 终点朝向
#     start_angle = start_state[2]  # 起点朝向
    
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
    
#     # 上下文地图： 起点终点的 位置标记 + 朝向的余弦值 + 朝向的正弦值，组成3通道输入
#     context_map = np.zeros((*normal_z.shape[:2], 3))  # [H, W, 3]
#     context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x, 0] = 1.0  # 终点标记为1
#     context_map[start_start_y:start_end_y, start_start_x:start_end_x, 0] = -1.0  # 起点标记为-1
#     context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x, 1] = np.cos(goal_angle)  # 终点朝向
#     context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x, 2] = np.sin(goal_angle)  # 终点朝向
#     context_map[start_start_y:start_end_y, start_start_x:start_end_x, 1] = np.cos(start_angle)  # 起点朝向
#     context_map[start_start_y:start_end_y, start_start_x:start_end_x, 2] = np.sin(start_angle)  # 起点朝向

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
    
#     # 拼接为6通道
#     encoded_input = np.concatenate((normal_z[:, :, None], context_map[:, :, :3], angle_map[:, :, :2]), axis=2)  # [H, W, 6]
#     return encoded_input

def get_encoder_input(normal_z, goal_pos, start_pos, normal_x, normal_y):
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
    
    # 上下文地图： 起点终点的位置标记
    context_map = np.zeros(normal_z.shape[:2])  # [H, W]
    context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0  # 终点标记为1
    context_map[start_start_y:start_end_y, start_start_x:start_end_x] = -1.0  # 起点标记为-1

    # 构造θ=<nx,ny>->(cosθ, sinθ)的映射
    angle_map = np.zeros((normal_z.shape[0], normal_z.shape[1], 2))  # [H, W, 2]
    for i in range(normal_z.shape[0]):
        for j in range(normal_z.shape[1]):
            n_xy_norm = np.linalg.norm([normal_x[i, j], normal_y[i, j]])
            if n_xy_norm == 0:
                # 如果法向量为零，避免除以零
                angle_map[i, j, 0] = 0.0
                angle_map[i, j, 1] = 0.0
            else:
                # 归一化法向量
                angle_map[i, j, 0] = normal_x[i, j] / n_xy_norm
                angle_map[i, j, 1] = normal_y[i, j] / n_xy_norm

    # 拼接为4通道
    encoded_input = np.concatenate((normal_z[:, :, None], context_map[:, :, None], angle_map[:, :, :2]), axis=2)  # [H, W, 4]
    return encoded_input

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
        trajectory = path_data['path']  # [N, 3]
        
        # 3. 生成编码输入
        path = trajectory[:, :3]  # [N, 3]
        
        encoded_input = get_encoder_input(
            normal_z, 
            goal_pos=path[-1, :2],  # 终点位置
            start_pos=path[0, :2],  # 起点位置
            normal_x=normal_x, 
            normal_y=normal_y
        )
        
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
        path_xy = trajectory[:, :2]  # [N, 2]
        num_trajectory_points = len(path_xy)
        
        # print(f"轨迹点数量: {num_trajectory_points}")
        # print(f"轨迹形状: {trajectory.shape}")
        
        # 为每个轨迹点找到对应的锚点
        positive_anchors_per_point = []  # 每个轨迹点对应的锚点列表
        for pos in path_xy:
            indices, = geom2pixMatpos(pos, res=res, size=map_tensor.shape[:2])
            # print(f"位置 {pos} 对应的锚点索引: {indices}")
            positive_anchors_per_point.append(list(set(indices)))

        # 找到最大锚点数量，用于填充
        max_anchors = max(len(anchors) for anchors in positive_anchors_per_point) if positive_anchors_per_point else 0
        
        # 填充正样本锚点到统一长度
        for anchors in positive_anchors_per_point:
            while len(anchors) < max_anchors:
                anchors.append(-1)  # 用-1填充
        
        # 生成负样本：为每个轨迹点生成相同数量的负样本
        all_anchor_indices = set(range(len(hashTable)))
        used_positive_anchors = set()
        for anchors in positive_anchors_per_point:
            used_positive_anchors.update([a for a in anchors if a != -1])
        
        available_negative_anchors = list(all_anchor_indices - used_positive_anchors)
        
        negative_anchors_per_point = []
        for _ in range(num_trajectory_points):
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
        
        # 转换为PyTorch张量
        return {
            'map': torch.as_tensor(encoded_input, dtype=torch.float).permute(2, 0, 1)[None, :],  # 地图：(1, 4, H, W) - 转换为channels-first格式
            'anchor': anchor,  # 锚点索引：(N, M)
            'labels': labels,  # 锚点标签：(N, M)
            'trajectory': torch.as_tensor(trajectory, dtype=torch.float),  # 轨迹点：[N, 3]
        }
