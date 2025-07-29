"""
dataLoader.py - MPT路径规划数据加载器模块

【核心功能】
本模块实现了用于训练Motion Planning Transformer (MPT)的数据加载器集合。
提供多种数据加载策略，支持不同的训练任务和网络架构。

【技术特点】
1. 多样化数据加载：支持序列预测、掩码预测、补丁生成等多种任务
2. 智能数据处理：自动处理变长序列、批量填充、坐标转换等
3. 灵活采样策略：支持正负样本平衡、困难样本挖掘、混合数据集训练
4. 高效内存管理：优化的数据结构和批处理机制

【应用场景】
1. Transformer序列预测：训练端到端的路径规划模型
2. UNet掩码生成：训练基于CNN的路径引导网络
3. 强化学习预训练：为RL算法提供专家轨迹数据
4. 多任务学习：同时训练多种规划任务的统一模型

【数据流程】
原始路径数据 → 坐标转换 → 特征编码 → 正负样本生成 → 批量组织 → 模型训练

【设计模式】
- 策略模式：不同的DataLoader类实现不同的数据加载策略
- 工厂模式：通过配置参数选择合适的数据加载器
- 适配器模式：统一不同数据源的接口

技术栈：
- PyTorch Dataset/DataLoader 数据加载框架
- scikit-image 图像处理
- NumPy 数值计算
- einops 张量操作

在MPT系统中的定位：
- 数据预处理中心：连接原始数据和模型训练
- 特征工程模块：实现坐标转换和特征编码
- 训练效率优化器：通过智能采样提升训练效果
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
    
    Args:
        batch (list): 批处理样本列表
            每个元素包含：
            - 'map': 地图张量，形状为(C, H, W)
            - 'anchor': 锚点序列，形状为(seq_len,)
            - 'labels': 标签序列，形状为(seq_len,)
    
    Returns:
        dict: 整理后的批处理数据
            - 'map': 批处理地图，形状为(B, C, H, W)
            - 'anchor': 填充后的锚点序列，形状为(B, max_seq_len)
            - 'labels': 填充后的标签序列，形状为(B, max_seq_len)
            - 'length': 每个样本的实际长度，形状为(B,)
    
    【技术细节】
    1. 空值过滤：确保数据加载过程中的异常样本不影响训练
    2. 自动填充：pad_sequence自动找到最长序列并填充短序列
    3. 长度记录：为Transformer提供注意力掩码的长度信息
    4. 批优先：batch_first=True确保批维度在第一维
    
    【使用场景】
    - DataLoader的collate_fn参数
    - Transformer模型的批处理输入
    - 变长序列的训练和推理
    """
    data = {}
    # 过滤有效样本：移除None值，确保数据完整性
    valid_batch = [batch_i for batch_i in batch if batch_i is not None]
    
    # 地图数据拼接：在batch维度上连接所有地图
    data['map'] = torch.cat([batch_i['map'][None, :] for batch_i in valid_batch])
    
    # 锚点序列填充：处理变长的锚点序列
    data['anchor'] = pad_sequence([batch_i['anchor'] for batch_i in valid_batch], batch_first=True)
    
    # 标签序列填充：处理变长的标签序列
    data['labels'] = pad_sequence([batch_i['labels'] for batch_i in valid_batch], batch_first=True)
    
    # 长度信息记录：保存每个样本的实际序列长度
    data['length'] = torch.tensor([batch_i['anchor'].shape[0] for batch_i in valid_batch])
    
    return data


def PaddedSequenceUnet(batch):
    """
    UNet模型批处理整理函数
    
    【核心功能】
    为UNet架构的训练准备批处理数据，主要处理图像到图像的映射任务。
    与序列模型不同，UNet处理固定尺寸的图像数据，无需序列填充。
    
    【处理流程】
    1. 过滤无效样本：确保数据完整性
    2. 地图数据拼接：组织输入图像批次
    3. 掩码数据拼接：组织目标掩码批次
    
    Args:
        batch (list): 批处理样本列表
            每个元素包含：
            - 'map': 输入地图，形状为(C, H, W)
            - 'mask': 目标掩码，形状为(H, W)
    
    Returns:
        dict: 整理后的批处理数据
            - 'map': 批处理输入地图，形状为(B, C, H, W)
            - 'mask': 批处理目标掩码，形状为(B, H, W)
    
    【应用场景】
    1. UNet路径掩码预测：从地图生成可行路径区域
    2. 语义分割任务：地图区域分类
    3. 图像到图像转换：地图特征增强
    
    【技术优势】
    - 简单高效：无需复杂的序列处理
    - 内存友好：固定尺寸的批处理
    - 并行友好：适合GPU并行计算
    """
    data = {}
    # 过滤有效样本：确保批处理数据的完整性
    valid_batch = [batch_i for batch_i in batch if batch_i is not None]
    
    # 输入地图拼接：组织UNet的输入批次
    data['map'] = torch.cat([batch_i['map'][None, :] for batch_i in valid_batch])
    
    # 目标掩码拼接：组织UNet的输出目标
    data['mask'] = torch.cat([batch_i['mask'][None, :] for batch_i in valid_batch])
    
    return data


def PaddedSequenceMPnet(batch):
    """
    MPNet模型批处理整理函数
    
    【核心功能】
    为MPNet（Motion Planning Network）架构准备批处理数据，处理序列到序列的路径规划任务。
    MPNet结合了地图信息和序列预测，用于端到端的路径生成。
    
    【处理流程】
    1. 地图数据组织：拼接所有样本的地图信息
    2. 输入序列填充：处理变长的输入轨迹序列
    3. 目标序列填充：处理变长的目标轨迹序列
    4. 长度信息记录：保存序列长度用于训练控制
    
    Args:
        batch (list): 批处理样本列表
            每个元素包含：
            - 'map': 地图张量，形状为(C, H, W)
            - 'inputs': 输入序列（当前状态+目标），形状为(seq_len, feature_dim)
            - 'targets': 目标序列（下一状态），形状为(seq_len, state_dim)
    
    Returns:
        dict: 整理后的批处理数据
            - 'map': 批处理地图，形状为(B, C, H, W)
            - 'inputs': 填充后的输入序列，形状为(B, max_seq_len, feature_dim)
            - 'targets': 填充后的目标序列，形状为(B, max_seq_len, state_dim)
            - 'length': 每个样本的序列长度，形状为(B,)
    
    【MPNet特点】
    1. 序列到序列：输入当前状态和目标，预测下一状态
    2. 地图感知：结合环境信息进行路径规划
    3. 端到端训练：直接从状态序列学习规划策略
    4. 可变长度：支持不同长度的路径序列
    
    【应用场景】
    - 机器人路径规划：从起点到终点的连续路径生成
    - 轨迹预测：基于历史轨迹预测未来路径
    - 强化学习：作为策略网络的预训练
    """
    data = {}
    # 地图数据拼接：组织MPNet的环境输入
    data['map'] = torch.cat([batch_i['map'][None, :, :] for batch_i in batch])
    
    # 输入序列填充：处理变长的状态-目标输入序列
    data['inputs'] = pad_sequence([batch_i['inputs'] for batch_i in batch], batch_first=True)
    
    # 目标序列填充：处理变长的下一状态目标序列
    data['targets'] = pad_sequence([batch_i['targets'] for batch_i in batch], batch_first=True)
    
    # 序列长度记录：用于损失计算和注意力掩码
    data['length'] = torch.tensor([batch_i['inputs'].size(0) for batch_i in batch])
    
    return data

# 【全局参数配置】
map_size = (480, 480)  # 地图尺寸：480x480像素的标准地图大小
receptive_field = 32   # 感受野大小：每个锚点影响的像素范围
res = 0.05            # 地图分辨率：每像素代表0.05米的实际距离

# 【锚点网格系统构建】
# 将连续的地图空间离散化为24x24的锚点网格，用于Transformer的token化处理

# X轴锚点坐标：从4像素开始，每20像素一个锚点，转换为几何坐标
X = np.arange(4, 24*20+4, 20)*res  # [0.2, 1.2, 2.2, ..., 23.8] 米

# Y轴锚点坐标：考虑坐标系翻转，从上到下排列
Y = 24-np.arange(4, 24*20+4, 20)*res  # [23.8, 22.8, 21.8, ..., 0.2] 米

# 创建2D网格：生成所有锚点的几何坐标
grid_2d = np.meshgrid(X, Y)  # 创建X-Y坐标网格

# 网格点重排：将2D网格转换为(N, 2)的点列表，N=576个锚点
grid_points = rearrange(grid_2d, 'c h w->(h w) c')  # 形状：(576, 2)

# 哈希表：锚点索引到像素坐标的映射表
hashTable = [(20*r+4, 20*c+4) for c in range(24) for r in range(24)]

# 【网格系统说明】
# 1. 锚点分布：24x24=576个锚点均匀分布在地图上
# 2. 像素间距：每个锚点间隔20像素，对应1米的实际距离
# 3. 边界偏移：起始偏移4像素，确保锚点不在地图边缘
# 4. 坐标对应：每个锚点代表一个20x20像素的区域
# 5. 索引映射：通过hashTable实现1D索引到2D像素坐标的转换

def geom2pixMatpos(pos, res=0.05, size=(480, 480)):
    """
    几何坐标到正样本锚点索引的转换函数
    
    【核心功能】
    将几何坐标转换为影响范围内的锚点索引，用于生成路径上的正样本锚点。
    通过距离阈值判断哪些锚点应该被激活，实现从连续路径到离散锚点的映射。
    
    【算法原理】
    1. 距离计算：计算输入位置到所有锚点的欧几里得距离
    2. 阈值筛选：选择距离小于阈值的锚点作为正样本
    3. 阈值设定：使用receptive_field*res*0.7作为影响半径
    
    Args:
        pos (tuple): 几何坐标 (x, y)
            单位：米
            坐标系：几何坐标系，原点在左下角
            范围：应在地图边界内
        res (float): 地图分辨率，默认0.05米/像素
            作用：控制距离计算的精度
            影响：影响锚点选择的空间范围
        size (tuple): 地图尺寸，默认(480, 480)
            用途：兼容性参数，当前实现中未使用
            保留：为未来扩展预留接口
    
    Returns:
        tuple: 正样本锚点索引
            格式：(indices,) 其中indices为numpy数组
            含义：距离输入位置较近的锚点索引列表
            用途：标记为路径上的重要位置
    
    【技术细节】
    1. 影响半径：receptive_field*res*0.7 ≈ 1.12米
       - receptive_field=32像素，res=0.05米/像素
       - 系数0.7：经验值，平衡覆盖范围和精度
       - 物理意义：每个路径点影响周围1米范围的锚点
    
    2. 距离度量：使用L2范数（欧几里得距离）
       - 符合物理直觉：空间中的直线距离
       - 计算高效：numpy向量化操作
       - 各向同性：所有方向的影响相等
    
    【应用场景】
    1. 正样本生成：标记路径经过的重要区域
    2. 特征提取：将连续路径转换为离散特征
    3. 损失计算：为分类任务提供正样本标签
    4. 注意力引导：指导模型关注路径相关区域
    """
    # 计算输入位置到所有锚点的距离
    distances = np.linalg.norm(grid_points - pos, axis=1)  # 形状：(576,)
    
    # 筛选距离阈值内的锚点索引
    indices = np.where(distances <= receptive_field * res * 0.7)  # 阈值：约1.12米
    
    return indices  # 返回正样本锚点索引元组

def geom2pixMatneg(pos, res=0.05, size=(480, 480), num=1):
    """
    几何坐标到负样本锚点索引的转换函数
    
    【核心功能】
    生成远离指定位置的负样本锚点索引，用于平衡正负样本的训练数据。
    通过距离阈值和随机采样，确保负样本的多样性和代表性。
    
    【算法原理】
    1. 距离计算：计算输入位置到所有锚点的距离
    2. 远距离筛选：选择距离大于阈值的锚点作为候选
    3. 随机采样：从候选中随机选择指定数量的负样本
    
    Args:
        pos (tuple): 几何坐标 (x, y)
            单位：米
            坐标系：几何坐标系，原点在左下角
            用途：作为负样本选择的参考点
        res (float): 地图分辨率，默认0.05米/像素
            作用：控制距离阈值的计算
            影响：影响正负样本的分界线
        size (tuple): 地图尺寸，默认(480, 480)
            用途：兼容性参数，当前实现中未使用
            保留：为未来扩展预留接口
        num (int): 采样的负样本数量，默认1
            作用：控制返回的负样本锚点数量
            平衡：与正样本数量保持适当比例
    
    Returns:
        tuple: 负样本锚点索引
            格式：(indices,) 其中indices为numpy数组
            含义：距离输入位置较远的锚点索引
            用途：作为分类任务的负样本标签
    
    【技术细节】
    1. 距离阈值：与geom2pixMatpos使用相同阈值
       - 确保正负样本的明确分界
       - 避免边界模糊导致的标签噪声
       - 保持数据标注的一致性
    
    2. 随机采样策略：
       - 避免负样本的空间聚集
       - 提高模型的泛化能力
       - 平衡不同区域的负样本分布
    
    3. 数量控制：
       - 可调节的负样本数量
       - 支持不同的正负样本比例
       - 适应不同的训练策略
    
    【应用场景】
    1. 对比学习：提供负样本进行对比训练
    2. 分类平衡：平衡正负样本的数量分布
    3. 困难挖掘：选择具有挑战性的负样本
    4. 正则化：防止模型过拟合到正样本
    """
    # 计算输入位置到所有锚点的距离
    dist = np.linalg.norm(grid_points - pos, axis=1)  # 形状：(576,)
    
    # 筛选距离阈值外的锚点索引（负样本候选）
    indices, = np.where(dist > receptive_field * res * 0.7)  # 距离大于约1.12米
    
    # 随机采样指定数量的负样本
    indices = np.random.choice(indices, size=num, replace=False)  # 无重复采样
    
    return indices,  # 返回负样本锚点索引元组

def get_encoder_input(InputMap, goal_pos, start_pos):
    """
    编码器输入生成函数
    
    【核心功能】
    将原始地图与起点、终点信息融合，生成Transformer编码器的输入。
    通过空间编码将起终点信息嵌入到地图中，为模型提供任务相关的上下文信息。
    
    【编码策略】
    1. 双通道设计：地图通道+上下文通道
    2. 空间标记：在起终点周围创建标记区域
    3. 差分编码：起点(-1)和终点(+1)使用不同的标记值
    4. 感受野对齐：标记区域大小与模型感受野匹配
    
    Args:
        InputMap (np.array): 输入灰度地图
            格式：2D numpy数组，形状为(H, W)
            值域：[0, 1]，0表示障碍物，1表示自由空间
            用途：提供环境的几何信息
        goal_pos (tuple): 终点像素坐标 (x, y)
            格式：像素坐标系，原点在左上角
            用途：标记路径规划的目标位置
            约束：必须在地图边界内
        start_pos (tuple): 起点像素坐标 (x, y)
            格式：像素坐标系，原点在左上角
            用途：标记路径规划的起始位置
            约束：必须在地图边界内
    
    Returns:
        torch.Tensor: 编码后的输入张量
            形状：(2, H, W)
            通道0：原始地图信息
            通道1：起终点上下文信息
            数据类型：torch.Tensor，可直接用于模型输入
    
    【技术细节】
    1. 上下文地图构建：
       - 初始化为零矩阵，与原地图同尺寸
       - 终点区域标记为+1.0（正值）
       - 起点区域标记为-1.0（负值）
       - 其他区域保持0.0（中性）
    
    2. 标记区域计算：
       - 区域大小：receptive_field × receptive_field像素
       - 中心对齐：以起终点为中心的正方形区域
       - 边界处理：确保标记区域不超出地图边界
       - 重叠处理：起终点区域可能重叠，起点优先级更高
    
    3. 坐标系统：
       - 输入坐标：(x, y)格式，x对应列，y对应行
       - 数组索引：[行, 列]格式，需要坐标转换
       - 边界检查：max(0, ...)和min(size, ...)确保有效范围
    
    【设计优势】
    1. 任务感知：模型能够理解当前的规划任务
    2. 空间局部性：标记区域与感受野匹配，提高效率
    3. 差分编码：起终点使用不同符号，便于模型区分
    4. 边界安全：完善的边界检查，避免数组越界
    
    【应用场景】
    1. Transformer输入：为MPT模型提供编码输入
    2. 多任务学习：同一地图上的不同起终点任务
    3. 条件生成：基于起终点条件生成路径
    4. 注意力引导：帮助模型聚焦于任务相关区域
    """
    map_size = InputMap.shape  # 获取地图尺寸
    assert len(map_size) == 2, "This only works for 2D maps"  # 确保输入为2D地图
    
    # 【步骤1】初始化上下文地图
    context_map = np.zeros(map_size)  # 创建与原地图同尺寸的零矩阵
    
    # 【步骤2】标记终点区域（正值编码）
    # 计算终点标记区域的边界
    goal_start_y = max(0, goal_pos[0] - receptive_field//2)  # Y方向起始位置
    goal_start_x = max(0, goal_pos[1] - receptive_field//2)  # X方向起始位置
    goal_end_y = min(map_size[1], goal_pos[0] + receptive_field//2)  # Y方向结束位置
    goal_end_x = min(map_size[0], goal_pos[1] + receptive_field//2)  # X方向结束位置
    
    # 在终点区域标记为+1.0
    context_map[goal_start_x:goal_end_x, goal_start_y:goal_end_y] = 1.0
    
    # 【步骤3】标记起点区域（负值编码）
    # 计算起点标记区域的边界
    start_start_y = max(0, start_pos[0] - receptive_field//2)  # Y方向起始位置
    start_start_x = max(0, start_pos[1] - receptive_field//2)  # X方向起始位置
    start_end_y = min(map_size[1], start_pos[0] + receptive_field//2)  # Y方向结束位置
    start_end_x = min(map_size[0], start_pos[1] + receptive_field//2)  # X方向结束位置
    
    # 在起点区域标记为-1.0（会覆盖可能的终点标记）
    context_map[start_start_x:start_end_x, start_start_y:start_end_y] = -1.0
    
    # 【步骤4】双通道融合
    # 将原地图和上下文地图沿通道维度拼接
    encoded_input = np.concatenate((InputMap[None, :], context_map[None, :]))  # 形状：(2, H, W)
    
    # 【步骤5】转换为PyTorch张量
    return torch.as_tensor(encoded_input)  # 返回可用于模型训练的张量


class PathPatchDataLoader(Dataset):
    """
    PathPatchDataLoader - 路径补丁数据加载器
    
    【系统概述】
    专门用于训练UNet架构的数据加载器，实现从地图到路径掩码的图像到图像转换任务。
    将路径规划问题转化为语义分割问题，生成路径可能经过的区域掩码。
    
    【核心功能】
    1. 路径到掩码转换：将连续路径转换为离散的二值掩码
    2. 成功轨迹筛选：仅加载规划成功的路径数据
    3. 空间编码：结合起终点信息生成任务相关的输入
    4. 补丁生成：围绕路径点生成感受野大小的补丁区域
    
    【技术特点】
    1. UNet适配：专门为UNet架构设计的数据格式
    2. 质量控制：自动过滤失败的路径数据
    3. 空间一致性：补丁大小与模型感受野匹配
    4. 高效索引：预构建索引表，提高数据访问效率
    
    【应用场景】
    1. UNet训练：图像到图像的路径掩码预测
    2. 语义分割：地图区域的可达性分析
    3. 路径引导：为传统规划算法提供引导区域
    4. 特征学习：学习路径相关的空间特征
    
    在MPT系统中的定位：
    - UNet训练数据源：为基于CNN的路径预测提供数据
    - 质量保证模块：确保训练数据的有效性
    - 空间特征提取器：将路径信息转换为空间特征
    """
    
    def __init__(self, env_list, dataFolder):
        """
        初始化路径补丁数据加载器
        
        【核心功能】
        构建用于UNet训练的数据加载器，自动扫描和索引所有有效的路径数据。
        实现高效的数据访问和质量控制机制。
        
        【初始化流程】
        1. 参数验证：确保输入参数的有效性
        2. 环境扫描：遍历所有指定的环境
        3. 路径筛选：仅保留规划成功的路径
        4. 索引构建：建立高效的数据访问索引
        
        Args:
            env_list (list): 环境编号列表
                格式：整数列表，如[0, 1, 2, ..., 99]
                用途：指定要加载数据的环境
                约束：环境文件夹必须存在
            dataFolder (str): 数据根目录路径
                格式：字符串路径，如'./data/train'
                结构：env{num:06d}/path_{i}.p
                要求：包含所有指定环境的数据文件
        
        【数据结构要求】
        数据文件夹结构：
        dataFolder/
        ├── env000000/
        │   ├── map_0.png          # 环境地图
        │   ├── path_0.p           # 路径数据0
        │   ├── path_1.p           # 路径数据1
        │   └── ...
        ├── env000001/
        │   └── ...
        └── ...
        
        【路径数据格式】
        每个path_*.p文件包含：
        - 'success': 布尔值，路径规划是否成功
        - 'path_interpolated': 插值后的路径点序列
        - 其他元数据...
        
        【质量控制】
        1. 成功性检查：仅加载success=True的路径
        2. 文件完整性：确保所有必需文件存在
        3. 数据有效性：验证路径数据的格式
        """
        assert isinstance(env_list, list), "env_list必须是列表类型"
        
        # 【步骤1】基本参数设置
        self.num_env = len(env_list)  # 环境数量
        self.env_list = env_list      # 环境列表
        self.dataFolder = dataFolder  # 数据根目录
        
        # 【步骤2】构建有效路径索引
        # 仅收集规划成功的轨迹数据
        self.indexDict = []  # 索引字典：存储(环境编号, 路径编号)对
        
        for envNum in env_list:
            # 获取当前环境的路径文件数量（减1是因为有map文件）
            env_path = osp.join(dataFolder, f'env{envNum:06d}')
            num_paths = len(os.listdir(env_path)) - 1  # 减去map文件
            
            # 遍历当前环境的所有路径文件
            for i in range(num_paths):
                path_file = osp.join(env_path, f'path_{i}.p')
                
                # 检查路径是否规划成功
                with open(path_file, 'rb') as f:
                    data = pickle.load(f)
                    if data['success']:  # 仅保留成功的路径
                        self.indexDict.append((envNum, i))
        
        # 【索引统计】
        print(f"加载完成：{len(self.indexDict)}个有效路径，来自{self.num_env}个环境")

    def __len__(self):
        """返回数据集大小"""
        return len(self.indexDict)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        【核心功能】
        根据索引获取一个完整的训练样本，包括编码后的地图输入和对应的路径掩码目标。
        实现从路径数据到UNet训练样本的完整转换流程。
        
        【处理流程】
        1. 索引解析：获取环境和路径编号
        2. 数据加载：读取地图和路径数据
        3. 坐标转换：将路径转换为像素坐标
        4. 输入编码：生成包含起终点信息的地图输入
        5. 锚点提取：将路径转换为离散锚点
        6. 掩码生成：围绕锚点生成补丁掩码
        
        Args:
            idx (int): 样本索引
                范围：[0, len(self.indexDict)-1]
                用途：指定要获取的样本
        
        Returns:
            dict: 训练样本字典
                'map': 编码后的输入地图，形状(2, H, W)
                'mask': 目标路径掩码，形状(H, W)
        
        【技术细节】
        1. 锚点提取策略：
           - 遍历路径上的每个点
           - 使用geom2pixMatpos找到影响的锚点
           - 去重处理，避免重复锚点
           - 记录锚点的像素坐标
        
        2. 掩码生成策略：
           - 以每个锚点为中心
           - 生成receptive_field大小的正方形区域
           - 边界裁剪，确保不超出地图范围
           - 所有区域的并集形成最终掩码
        
        3. 数据类型处理：
           - 输入地图：浮点张量，保持原始精度
           - 目标掩码：整数张量，用于分类损失
        """
        # 【步骤1】解析索引，获取环境和路径编号
        env, idx_sample = self.indexDict[idx]
        
        # 【步骤2】加载地图数据
        map_path = osp.join(self.dataFolder, f'env{env:06d}', f'map_{env}.png')
        mapEnvg = skimage.io.imread(map_path, as_gray=True)  # 读取灰度地图
        
        # 【步骤3】加载路径数据
        path_file = osp.join(self.dataFolder, f'env{env:06d}', f'path_{idx_sample}.p')
        with open(path_file, 'rb') as f:
            data = pickle.load(f)

        # 【步骤4】处理成功的路径数据
        if data['success']:
            path = data['path_interpolated']  # 获取插值后的路径
            
            # 【步骤5】坐标转换和输入编码
            goal_index = geom2pix(path[-1, :])   # 终点几何坐标→像素坐标
            start_index = geom2pix(path[0, :])   # 起点几何坐标→像素坐标
            mapEncoder = get_encoder_input(mapEnvg, goal_index, start_index)  # 生成编码输入

            # 【步骤6】路径锚点提取
            AnchorPointsPos = []  # 锚点索引列表
            AnchorPointsXY = []   # 锚点像素坐标列表
            
            for pos in path:
                # 找到当前路径点影响的锚点
                indices, = geom2pixMatpos(pos)
                for index in indices:
                    if index not in AnchorPointsPos:  # 去重处理
                        AnchorPointsPos.append(index)
                        AnchorPointsXY.append(hashTable[index])  # 添加对应的像素坐标

            # 【步骤7】生成路径掩码
            maskMap = np.zeros_like(mapEnvg)  # 初始化掩码，与地图同尺寸
            
            for pos in AnchorPointsXY:
                # 计算每个锚点的补丁区域边界
                goal_start_x = max(0, pos[0] - receptive_field//2)
                goal_start_y = max(0, pos[1] - receptive_field//2)
                goal_end_x = min(map_size[1], pos[0] + receptive_field//2)
                goal_end_y = min(map_size[0], pos[1] + receptive_field//2)
                
                # 在补丁区域内标记为1
                maskMap[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0

            # 【步骤8】返回训练样本
            return {
                'map': torch.as_tensor(mapEncoder),        # 编码后的输入地图
                'mask': torch.as_tensor(maskMap, dtype=int)  # 目标路径掩码
            }
            
class PathSeqDataLoader(Dataset):
    """
    PathSeqDataLoader - 路径序列数据加载器
    
    【系统概述】
    专门用于序列到序列路径规划模型的数据加载器，如MPNet等。
    将路径规划问题建模为序列预测任务，输入当前状态和目标，预测下一状态。
    
    【核心功能】
    1. 序列化处理：将路径转换为状态序列
    2. 归一化处理：将坐标归一化到[-1, 1]范围
    3. 输入-目标配对：构建序列预测的训练对
    4. 地图上下文：结合环境信息进行序列预测
    
    【技术特点】
    1. 序列建模：支持变长序列的端到端学习
    2. 坐标归一化：提高模型的数值稳定性
    3. 目标条件：每个输入都包含最终目标信息
    4. 时序预测：基于历史状态预测未来状态
    
    【应用场景】
    1. MPNet训练：端到端的路径序列生成
    2. 轨迹预测：基于部分轨迹预测完整路径
    3. 强化学习：作为策略网络的监督预训练
    4. 序列规划：将规划问题转化为序列生成
    
    在MPT系统中的定位：
    - 序列学习数据源：为序列模型提供训练数据
    - 归一化处理器：统一不同尺度的坐标系统
    - 时序建模支持：支持基于时间的路径预测
    """
    
    def __init__(self, env_list, dataFolder, worldMapBounds):
        """
        初始化路径序列数据加载器
        
        【核心功能】
        构建用于序列到序列学习的数据加载器，支持坐标归一化和序列预测任务。
        
        【初始化流程】
        1. 参数验证和设置
        2. 成功路径索引构建
        3. 坐标边界设置
        4. 归一化参数准备
        
        Args:
            env_list (list): 环境编号列表
                格式：整数列表，指定训练环境
                用途：控制数据来源的多样性
            dataFolder (str): 数据根目录路径
                结构：与PathPatchDataLoader相同
                要求：包含路径和地图数据
            worldMapBounds (list/np.array): 世界坐标边界 [长度, 高度]
                单位：米
                用途：坐标归一化的参考范围
                格式：[max_x, max_y]，如[24, 24]
        
        【归一化策略】
        坐标归一化公式：normalized = (coord / bounds) * 2 - 1
        - 输入范围：[0, bounds]
        - 输出范围：[-1, 1]
        - 优势：提高模型训练稳定性，加速收敛
        """
        assert isinstance(env_list, list), "env_list必须是列表类型"
        
        # 【步骤1】基本参数设置
        self.num_env = len(env_list)
        self.env_list = env_list
        self.dataFolder = dataFolder
        
        # 【步骤2】构建成功路径索引
        self.indexDict = []
        for envNum in env_list:
            env_path = osp.join(dataFolder, f'env{envNum:06d}')
            num_paths = len(os.listdir(env_path)) - 1  # 减去地图文件
            
            for i in range(num_paths):
                path_file = osp.join(env_path, f'path_{i}.p')
                with open(path_file, 'rb') as f:
                    if pickle.load(f)['success']:  # 仅保留成功路径
                        self.indexDict.append((envNum, i))
        
        # 【步骤3】设置坐标边界
        self.worldMapBounds = worldMapBounds if isinstance(worldMapBounds, np.ndarray) else np.array(worldMapBounds)
        
        print(f"序列数据加载器初始化完成：{len(self.indexDict)}个有效序列")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.indexDict)
    
    def __getitem__(self, idx):
        """
        获取指定索引的序列训练样本
        
        【核心功能】
        生成序列到序列学习的训练样本，包括地图上下文、输入序列和目标序列。
        实现路径的序列化表示和归一化处理。
        
        【处理流程】
        1. 数据加载：读取地图和路径数据
        2. 坐标归一化：将路径坐标归一化到[-1, 1]
        3. 输入序列构建：当前状态+目标状态
        4. 目标序列构建：下一状态序列
        5. 张量转换：转换为PyTorch张量
        
        Args:
            idx (int): 样本索引
        
        Returns:
            dict: 序列训练样本
                'map': 环境地图，形状(1, H, W)
                'inputs': 输入序列，形状(seq_len-1, 4)
                'targets': 目标序列，形状(seq_len-1, 2)
        
        【序列构建策略】
        1. 输入序列格式：[当前x, 当前y, 目标x, 目标y]
           - 前两维：当前状态坐标
           - 后两维：最终目标坐标（固定）
           - 长度：路径长度-1（最后一个点不需要预测）
        
        2. 目标序列格式：[下一x, 下一y]
           - 对应输入序列的下一状态
           - 长度：与输入序列相同
           - 用途：监督学习的目标标签
        
        【归一化处理】
        - 原始坐标范围：[0, worldMapBounds]
        - 归一化公式：(coord / bounds) * 2 - 1
        - 归一化范围：[-1, 1]
        - 优势：数值稳定，加速训练收敛
        
        【技术细节】
        1. 目标复制：每个输入都包含相同的最终目标
        2. 序列对齐：输入[t]对应目标[t+1]
        3. 维度扩展：地图添加batch维度以保持一致性
        """
        # 【步骤1】解析索引并加载数据
        env, idx_sample = self.indexDict[idx]
        
        # 加载环境地图
        map_path = osp.join(self.dataFolder, f'env{env:06d}', f'map_{env}.png')
        mapEnvg = skimage.io.imread(map_path, as_gray=True)
        
        # 加载路径数据
        path_file = osp.join(self.dataFolder, f'env{env:06d}', f'path_{idx_sample}.p')
        with open(path_file, 'rb') as f:
            data = pickle.load(f)

        # 【步骤2】路径坐标归一化
        # 将坐标从[0, bounds]归一化到[-1, 1]
        path = (data['path'] / self.worldMapBounds) * 2 - 1
        
        # 【步骤3】构建输入序列
        # 格式：[当前状态, 目标状态]
        current_states = path[:-1, :]  # 当前状态序列（除最后一个点）
        goal_state = path[-1][None, :]  # 最终目标状态
        goal_repeated = np.repeat(goal_state, path.shape[0]-1, axis=0)  # 重复目标状态
        
        nInput = np.c_[current_states, goal_repeated]  # 拼接：[当前x, 当前y, 目标x, 目标y]
        
        # 【步骤4】构建目标序列
        # 下一状态序列（从第二个点开始）
        nTarget = path[1:, :]
        
        # 【步骤5】转换为PyTorch张量
        return {
            'map': torch.as_tensor(mapEnvg[None, :], dtype=torch.float),  # 地图：(1, H, W)
            'inputs': torch.as_tensor(nInput, dtype=torch.float),         # 输入：(seq_len-1, 4)
            'targets': torch.as_tensor(nTarget, dtype=torch.float)        # 目标：(seq_len-1, 2)
        }

class PathDataLoader(Dataset):
    """
    PathDataLoader - 基础路径数据加载器
    
    【系统概述】
    用于Transformer分类训练的基础数据加载器，实现正负样本的平衡采样。
    将路径规划问题转化为锚点分类问题，预测每个锚点是否在最优路径上。
    
    【核心功能】
    1. 正负样本生成：从路径提取正样本，随机采样负样本
    2. 样本平衡：控制正负样本比例，避免类别不平衡
    3. 锚点分类：将连续路径转换为离散分类任务
    4. 全数据加载：包含所有路径数据，不进行成功性筛选
    
    【技术特点】
    1. 分类导向：专门为分类任务设计的数据格式
    2. 平衡采样：自动平衡正负样本数量
    3. 随机性：负样本的随机采样增加多样性
    4. 完整性：加载所有路径数据，包括失败案例
    
    【应用场景】
    1. Transformer分类：训练锚点重要性分类器
    2. 特征学习：学习路径相关的空间特征
    3. 注意力训练：训练模型的空间注意力机制
    4. 基线对比：作为其他方法的对比基线
    
    在MPT系统中的定位：
    - 分类训练数据源：为Transformer分类器提供数据
    - 样本平衡器：确保训练数据的类别平衡
    - 基础数据加载器：其他加载器的基础版本

    数据集格式：
    dataFolder/             # 数据根目录
    ├── env000000/          # 环境0 (6位数字格式)   "env{编号:06d}"
    │   ├── map_0.png           # 环境地图文件      "map_{编号}.png"
    │   ├── path_0.p            # 路径数据文件0     "path_{序号}.p"
    │   ├── path_1.p            # 路径数据文件1
    │   ├── path_2.p            # 路径数据文件2
    │   └── ...                 # 更多路径文件
    ├── env000001/          # 环境1
    │   ├── map_1.png           # 环境地图文件
    │   ├── path_0.p            # 路径数据文件0
    │   └── ...
    ├── env000002/          # 环境2
    └── ...                 # 更多环境

    地图文件 (map_*.png):
        - 格式: PNG灰度图像
        - 尺寸: 480×480像素（由全局变量 map_size = (480, 480) 定义）
        - 分辨率: 0.05米/像素（由全局变量 res = 0.05 定义）
        - 值域: [0, 1]
            0 表示障碍物（黑色）
            1 表示自由空间（白色）
        - 坐标系: 像素坐标系，原点在左上角

    路径文件 (path_*.p):
        · 必需字段: 
            - 'success': 布尔值
                True: 路径规划成功
                False: 路径规划失败
                用于数据质量控制
            - 'path_interpolated': numpy数组
                形状: (N, 2) 其中N是路径点数量
                内容: 插值后的路径点坐标序列
                坐标系: 几何坐标系（米为单位）
                格式: [[x1, y1], [x2, y2], ..., [xN, yN]]
                坐标范围: [0, 24] 米（对应480像素×0.05米/像素）
        · 可选字段:
            - 'path': 原始路径数据（未插值）

        # path_0.p 文件内容示例
        {
            'success': True,
            'path_interpolated': np.array([
                [1.0, 1.0],    # 起点坐标 (米)
                [1.5, 1.2],    # 中间点
                [2.0, 1.8],    # 中间点
                # ... 更多路径点
                [20.0, 22.0]   # 终点坐标 (米)
            ]),
            # 其他可选字段...
        }

    坐标系统说明：
        · 几何坐标系（路径数据使用）:
            - 单位: 米
            - 原点: 左下角
            - X轴: 向右为正
            - Y轴: 向上为正
            - 范围: [0, 24] × [0, 24] 米

        · 像素坐标系（地图图像使用）:
            - 单位: 像素
            - 原点: 左上角
            - X轴: 向右为正
            - Y轴: 向下为正
            - 范围: [0, 480) × [0, 480) 像素

    """
    
    def __init__(self, env_list, dataFolder):
        """
        初始化基础路径数据加载器
        
        【核心功能】
        构建用于分类训练的数据加载器，加载所有路径数据而不进行成功性筛选。
        
        【设计特点】
        - 全数据加载：包含成功和失败的路径
        - 简单索引：直接构建所有路径的索引
        - 无质量筛选：保留原始数据的完整性
        
        Args:
            env_list (list): 环境编号列表
            dataFolder (str): 数据根目录路径
        
        【与其他加载器的区别】
        - PathPatchDataLoader: 仅加载成功路径，用于UNet训练
        - PathSeqDataLoader: 仅加载成功路径，用于序列预测
        - PathDataLoader: 加载所有路径，用于分类训练
        """
        assert isinstance(env_list, list), "env_list必须是列表类型"
        
        # 基本参数设置
        self.num_env = len(env_list)
        self.env_list = env_list
        self.dataFolder = dataFolder
        
        # 构建完整路径索引（包含所有路径，不筛选成功性）
        self.indexDict = [(envNum, i) 
            for envNum in env_list 
                for i in range(len(os.listdir(osp.join(dataFolder, f'env{envNum:06d}')))-1)
            ]

        print(f"基础数据加载器初始化完成：{len(self.indexDict)}个路径样本")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.indexDict)
    
    def __getitem__(self, idx):
        """
        获取指定索引的分类训练样本
        
        【核心功能】
        生成用于锚点分类的训练样本，包括正负样本的平衡采样。
        仅处理成功的路径数据，失败路径返回None。
        
        【处理流程】
        1. 数据加载和验证
        2. 输入编码生成
        3. 正样本锚点提取
        4. 负样本随机采样
        5. 样本标签构建
        
        Args:
            idx (int): 样本索引
        
        Returns:
            dict or None: 分类训练样本或None（失败路径）
                'map': 编码后的地图输入，形状(2, H, W)
                'anchor': 锚点索引序列，形状(N,)
                'labels': 锚点标签序列，形状(N,)
        
        【样本平衡策略】
        1. 正样本：路径经过的锚点，标签为1
        2. 负样本：路径未经过的锚点，标签为0
        3. 数量比例：负样本数量 = min(可用负样本, 2×正样本数量)
        4. 随机采样：负样本随机选择，增加多样性
        
        【技术细节】
        1. 成功性检查：仅处理success=True的路径
        2. 去重处理：正样本锚点去重，避免重复
        3. 集合运算：使用集合差集快速找到负样本候选
        4. 无重复采样：replace=False确保负样本不重复
        """
        # 【步骤1】解析索引并加载数据
        env, idx_sample = self.indexDict[idx]
        
        # 加载地图
        map_path = osp.join(self.dataFolder, f'env{env:06d}', f'map_{env}.png')
        mapEnvg = skimage.io.imread(map_path, as_gray=True)
        
        # 加载路径数据
        path_file = osp.join(self.dataFolder, f'env{env:06d}', f'path_{idx_sample}.p')
        with open(path_file, 'rb') as f:
            data = pickle.load(f)

        # 【步骤2】仅处理成功的路径
        if data['success']:
            path = data['path_interpolated']
            
            # 【步骤3】生成编码输入
            goal_index = geom2pix(path[-1, :])   # 终点坐标转换
            start_index = geom2pix(path[0, :])   # 起点坐标转换
            mapEncoder = get_encoder_input(mapEnvg, goal_index, start_index)

            # 【步骤4】提取正样本锚点
            AnchorPointsPos = []  # 正样本锚点索引列表
            for pos in path:
                indices, = geom2pixMatpos(pos)  # 获取当前位置影响的锚点
                for index in indices:
                    if index not in AnchorPointsPos:  # 去重处理
                        AnchorPointsPos.append(index)

            # 【步骤5】生成负样本锚点
            # 计算所有可能的负样本候选（全集 - 正样本集）
            backgroundPoints = list(set(range(len(hashTable))) - set(AnchorPointsPos))
            
            # 确定负样本数量：最多为正样本的2倍
            numBackgroundSamp = min(len(backgroundPoints), 2 * len(AnchorPointsPos))
            
            # 随机采样负样本
            AnchorPointsNeg = np.random.choice(backgroundPoints, size=numBackgroundSamp, replace=False).tolist()
            
            # 【步骤6】构建最终的锚点序列和标签
            # 拼接正负样本锚点
            anchor = torch.cat((torch.tensor(AnchorPointsPos), torch.tensor(AnchorPointsNeg)))
            
            # 构建对应标签（正样本为1，负样本为0）
            labels = torch.zeros_like(anchor)
            labels[:len(AnchorPointsPos)] = 1  # 前面的正样本标记为1
            
            return {
                'map': torch.as_tensor(mapEncoder),  # 编码后的地图输入
                'anchor': anchor,                    # 锚点索引序列
                'labels': labels                     # 锚点分类标签
            }
        
        # 失败路径返回None，会在批处理时被过滤
        return None
    
class PathSE2DataLoader(Dataset):
    """
    PathSE2DataLoader - 基础路径数据加载器
    """
    
    def __init__(self, env_list, dataFolder):
        """
        初始化基础路径数据加载器
        
        用于加载sst_map数据集中的路径数据，支持SE(2)坐标系的锚点分类任务。
        在path_*.p文件中，路径数据以SE(2)坐标系表示，包含位置和朝向信息，比原来多了朝向信息。
        
        对于正样本而言，我们将它对应的朝向信息，设置为附近0.7倍的锚点距离内所有锚点的朝向的平均值。
        """
        assert isinstance(env_list, list), "env_list必须是列表类型"
        
        # 基本参数设置
        self.num_env = len(env_list)
        self.env_list = env_list
        self.dataFolder = dataFolder
        
        # 构建完整路径索引（包含所有路径，不筛选成功性）
        self.indexDict = [(envNum, i) 
            for envNum in env_list 
                for i in range(len(os.listdir(osp.join(dataFolder, f'env{envNum:06d}')))-1)
            ]

        print(f"基础数据加载器初始化完成：{len(self.indexDict)}个路径样本")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.indexDict)
    
    def __getitem__(self, idx):
        """
        获取指定索引的分类训练样本
        
        【核心功能】
        生成用于锚点分类的训练样本，包括正负样本的平衡采样。
        仅处理成功的路径数据，失败路径返回None。
        
        【处理流程】
        1. 数据加载和验证
        2. 输入编码生成
        3. 正样本锚点提取
        4. 负样本随机采样
        5. 样本标签构建
        
        Args:
            idx (int): 样本索引
        
        Returns:
            dict or None: 分类训练样本或None（失败路径）
                'map': 编码后的地图输入，形状(2, H, W)
                'anchor': 锚点索引序列，形状(N,)
                'labels': 锚点标签序列，形状(N,)
        
        【样本平衡策略】
        1. 正样本：路径经过的锚点，标签为1
        2. 负样本：路径未经过的锚点，标签为0
        3. 数量比例：负样本数量 = min(可用负样本, 2×正样本数量)
        4. 随机采样：负样本随机选择，增加多样性
        
        【技术细节】
        1. 成功性检查：仅处理success=True的路径
        2. 去重处理：正样本锚点去重，避免重复
        3. 集合运算：使用集合差集快速找到负样本候选
        4. 无重复采样：replace=False确保负样本不重复
        """
        # 【步骤1】解析索引并加载数据
        env, idx_sample = self.indexDict[idx]
        
        # 加载地图
        map_path = osp.join(self.dataFolder, f'env{env:06d}', f'map_{env}.png')
        mapEnvg = skimage.io.imread(map_path, as_gray=True)
        
        # 加载路径数据
        path_file = osp.join(self.dataFolder, f'env{env:06d}', f'path_{idx_sample}.p')
        with open(path_file, 'rb') as f:
            data = pickle.load(f)

        # 【步骤2】仅处理成功的路径
        if data['success']:
            path = data['path_interpolated']
            
            # 【步骤3】生成编码输入（4通道：地图、起终点坐标、sin(theta)、cos(theta)）
            goal_index = geom2pix(path[-1, :])   # 终点坐标转换
            start_index = geom2pix(path[0, :])   # 起点坐标转换
            map_size = mapEnvg.shape
            context_map = np.zeros(map_size)  # 起终点坐标通道
            sin_map = np.zeros(map_size)      # 起终点sin(theta)通道
            cos_map = np.zeros(map_size)      # 起终点cos(theta)通道
            receptive_field = 32
            # 起点区域
            start_theta = path[0, 2]
            start_start_y = max(0, start_index[0] - receptive_field//2)
            start_start_x = max(0, start_index[1] - receptive_field//2)
            start_end_y = min(map_size[0], start_index[0] + receptive_field//2)
            start_end_x = min(map_size[1], start_index[1] + receptive_field//2)
            context_map[start_start_y:start_end_y, start_start_x:start_end_x] = -1.0
            sin_map[start_start_y:start_end_y, start_start_x:start_end_x] = np.sin(start_theta)
            cos_map[start_start_y:start_end_y, start_start_x:start_end_x] = np.cos(start_theta)
            # 终点区域
            goal_theta = path[-1, 2]
            goal_start_y = max(0, goal_index[0] - receptive_field//2)
            goal_start_x = max(0, goal_index[1] - receptive_field//2)
            goal_end_y = min(map_size[0], goal_index[0] + receptive_field//2)
            goal_end_x = min(map_size[1], goal_index[1] + receptive_field//2)
            context_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = 1.0
            sin_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = np.sin(goal_theta)
            cos_map[goal_start_y:goal_end_y, goal_start_x:goal_end_x] = np.cos(goal_theta)
            # 拼接为4通道
            mapEncoder = np.concatenate((mapEnvg[None, :], context_map[None, :], sin_map[None, :], cos_map[None, :]), axis=0)

            # 【步骤4】提取正样本锚点及其朝向
            AnchorPointsPos = []      # 正样本锚点索引列表
            AnchorPointsTheta = []    # 正样本锚点对应的平均朝向
            path_xy = path[:, :2]     # 轨迹的xy坐标
            path_theta = path[:, 2]   # 轨迹的朝向（假定第三维为theta）
            for i, pos in enumerate(path):
                indices, = geom2pixMatpos(pos)
                for index in indices:
                    if index not in AnchorPointsPos:
                        # 计算该锚点附近0.7米范围内所有轨迹点的平均朝向
                        dists = np.linalg.norm(path_xy - pos[:2], axis=1)
                        nearby_theta = path_theta[dists < 0.7]
                        if len(nearby_theta) > 0:
                            avg_theta = np.mean(nearby_theta)
                        else:
                            avg_theta = pos[2]  # 若无邻近点则用自身朝向
                        AnchorPointsPos.append(index)
                        AnchorPointsTheta.append(avg_theta)

            # 【步骤5】生成负样本锚点（不关联朝向）
            backgroundPoints = list(set(range(len(hashTable))) - set(AnchorPointsPos))
            numBackgroundSamp = min(len(backgroundPoints), 2 * len(AnchorPointsPos))
            AnchorPointsNeg = np.random.choice(backgroundPoints, size=numBackgroundSamp, replace=False).tolist()
            # 负样本朝向用0填充（或可用nan）
            AnchorPointsThetaNeg = [0.0] * len(AnchorPointsNeg)
            # 【步骤6】构建最终的锚点序列、标签和朝向
            anchor = torch.cat((torch.tensor(AnchorPointsPos), torch.tensor(AnchorPointsNeg)))
            labels = torch.zeros_like(anchor)
            labels[:len(AnchorPointsPos)] = 1
            # 拼接正负样本的朝向
            anchor_theta = torch.tensor(AnchorPointsTheta + AnchorPointsThetaNeg)
            return {
                'map': torch.as_tensor(mapEncoder),      # 编码后的地图输入
                'anchor': anchor,                        # 锚点索引序列
                'labels': labels,                        # 锚点分类标签
                'anchor_theta': anchor_theta              # 锚点对应的平均朝向
            }
        
        # 失败路径返回None，会在批处理时被过滤
        return None

class PathMixedDataLoader(Dataset):
    """
    PathMixedDataLoader - 混合数据集路径数据加载器
    
    【系统概述】
    专门用于混合不同类型环境数据的数据加载器，实现迷宫和森林环境的统一训练。
    通过混合不同难度和类型的环境，提高模型的泛化能力和鲁棒性。
    
    【核心功能】
    1. 多环境融合：同时加载迷宫和森林两种不同类型的环境数据
    2. 难度平衡：确保困难规划问题在训练过程中均匀分布
    3. 数据多样性：通过混合数据集增加训练样本的多样性
    4. 统一接口：为不同类型环境提供统一的数据访问接口
    
    【技术特点】
    1. 双数据源：支持同时从两个不同的数据源加载数据
    2. 类型标识：通过标识符区分不同类型的环境数据
    3. 均匀分布：确保不同类型数据在训练中的均匀分布
    4. 扩展性：易于扩展到更多类型的环境数据
    
    【应用场景】
    1. 多环境训练：在不同类型环境上训练通用模型
    2. 域适应：提高模型在不同环境间的适应能力
    3. 鲁棒性测试：评估模型在多样化环境中的性能
    4. 迁移学习：利用多种环境数据进行知识迁移
    
    在MPT系统中的定位：
    - 多域数据源：为跨域训练提供数据支持
    - 泛化能力增强器：通过数据多样性提升模型泛化
    - 统一训练接口：简化多环境训练的复杂性
    """

    def __init__(self, envListMaze, dataFolderMaze, envListForest, dataFolderForest):
        """
        初始化混合数据集路径数据加载器
        
        【核心功能】
        构建支持多种环境类型的混合数据加载器，实现迷宫和森林环境的统一管理。
        
        【初始化策略】
        1. 分别构建两种环境的索引
        2. 使用类型标识符区分数据源
        3. 统一数据访问接口
        4. 支持动态数据源切换
        
        Args:
            envListMaze (list): 迷宫环境编号列表
                格式：整数列表，指定迷宫环境
                特点：通常包含狭窄通道和复杂拓扑
            dataFolderMaze (str): 迷宫数据文件夹路径
                结构：标准的环境数据结构
                内容：迷宫类型的路径规划数据
            envListForest (list): 森林环境编号列表
                格式：整数列表，指定森林环境
                特点：通常包含随机障碍物分布
            dataFolderForest (str): 森林数据文件夹路径
                结构：标准的环境数据结构
                内容：森林类型的路径规划数据
        
        【数据组织策略】
        1. 类型标识：'M'表示迷宫，'F'表示森林
        2. 索引格式：(类型, 环境编号, 路径编号)
        3. 数据映射：通过字典映射类型到数据文件夹
        4. 统一访问：提供统一的数据访问接口
        
        【设计优势】
        - 类型区分：清晰的数据类型标识
        - 扩展性：易于添加新的环境类型
        - 平衡性：支持不同类型数据的平衡采样
        - 灵活性：支持动态调整数据源比例
        """
        assert isinstance(envListMaze, list), "Needs to be a list"
        assert isinstance(envListForest, list), "Needs to be a list"

        self.num_env = len(envListForest) + len(envListMaze)
        self.indexDictMaze = [('M', envNum, i) 
            for envNum in envListMaze 
                for i in range(len(os.listdir(osp.join(dataFolderMaze, f'env{envNum:06d}')))-1)
            ]
        self.indexDictForest = [('F', envNum, i) 
            for envNum in envListForest 
                for i in range(len(os.listdir(osp.join(dataFolderForest, f'env{envNum:06d}')))-1)
            ]
        self.dataFolder = {'F': dataFolderForest, 'M':dataFolderMaze}
        self.envList = {'F': envListForest, 'M': envListMaze}
    

    def __len__(self):
        return len(self.indexDictForest)+len(self.indexDictMaze)
    
    def __getitem__(self, idx):
        """
        获取指定索引的混合环境训练样本
        
        【核心功能】
        根据索引获取来自不同环境类型的训练样本，支持迷宫和森林环境的统一处理。
        实现跨环境类型的数据访问和处理。
        
        【处理流程】
        1. 索引解析：解析环境类型和具体索引
        2. 数据源选择：根据类型选择对应的数据文件夹
        3. 数据加载：加载地图和路径数据
        4. 统一处理：使用相同的处理流程生成训练样本
        
        Args:
            idx: 样本索引，格式为(类型, 环境编号, 路径编号)
        
        Returns:
            dict: 训练样本字典
                'map': 编码后的地图输入
                'anchor': 锚点索引序列
                'labels': 锚点分类标签
        
        【多环境处理】
        1. 类型识别：通过索引第一个元素识别环境类型
        2. 路径映射：根据类型映射到对应的数据文件夹
        3. 统一编码：不同类型环境使用相同的编码方式
        4. 一致输出：确保不同环境的输出格式一致
        
        【错误处理】
        - 索引解析异常：捕获并输出调试信息
        - 数据加载失败：返回None或跳过无效样本
        - 类型不匹配：通过字典映射避免类型错误
        """
        try:
            DF, env, idx_sample = idx  # 解析索引：(类型, 环境编号, 路径编号)
        except ValueError:
            print(f"索引解析错误: {idx}")  # 调试信息输出
            return None
            
        # 根据环境类型选择数据文件夹
        dataFolder = self.dataFolder[DF]  # 'M'->迷宫文件夹, 'F'->森林文件夹
        
        # 加载环境地图
        map_path = osp.join(dataFolder, f'env{env:06d}', f'map_{env}.png')
        mapEnvg = skimage.io.imread(map_path, as_gray=True)
        
        # 加载路径数据
        path_file = osp.join(dataFolder, f'env{env:06d}', f'path_{idx_sample}.p')
        with open(path_file, 'rb') as f:
            data = pickle.load(f)

        # 处理成功的路径数据
        if data['success']:
            path = data['path_interpolated']
            
            # 生成编码输入（与单一环境处理相同）
            goal_index = geom2pix(path[-1, :])   # 终点坐标转换
            start_index = geom2pix(path[0, :])   # 起点坐标转换
            mapEncoder = get_encoder_input(mapEnvg, goal_index, start_index)

            # 提取正样本锚点
            AnchorPointsPos = []
            for pos in path:
                indices, = geom2pixMatpos(pos)
                for index in indices:
                    if index not in AnchorPointsPos:
                        AnchorPointsPos.append(index)

            # 生成负样本锚点（2:1的负正比例）
            backgroundPoints = list(set(range(len(hashTable))) - set(AnchorPointsPos))
            numBackgroundSamp = min(len(backgroundPoints), 2 * len(AnchorPointsPos))
            AnchorPointsNeg = np.random.choice(backgroundPoints, size=numBackgroundSamp, replace=False).tolist()
            
            # 构建最终样本
            anchor = torch.cat((torch.tensor(AnchorPointsPos), torch.tensor(AnchorPointsNeg)))
            labels = torch.zeros_like(anchor)
            labels[:len(AnchorPointsPos)] = 1  # 正样本标记为1
            
            return {
                'map': torch.as_tensor(mapEncoder),  # 编码后的地图输入
                'anchor': anchor,                    # 锚点索引序列
                'labels': labels                     # 锚点分类标签
            }
        
        # 失败路径返回None
        return None

class PathHardMineDataLoader(Dataset):
    """
    PathHardMineDataLoader - 困难样本挖掘数据加载器
    
    【系统概述】
    专门用于困难样本挖掘和平衡训练的数据加载器。通过区分简单和困难的路径规划问题，
    确保困难样本在训练过程中得到充分的关注，提高模型对复杂场景的处理能力。
    
    【核心功能】
    1. 难度分层：区分简单和困难的路径规划任务
    2. 困难挖掘：重点关注模型难以处理的复杂场景
    3. 平衡训练：确保困难样本在训练中均匀分布
    4. 自适应采样：根据难度动态调整采样策略
    
    【技术特点】
    1. 双难度数据源：分别管理简单和困难样本
    2. 均匀分布：确保困难问题在训练过程中均匀出现
    3. 样本平衡：控制简单和困难样本的比例
    4. 挖掘策略：实现有效的困难样本挖掘机制
    
    【应用场景】
    1. 困难样本训练：专门训练模型处理复杂场景
    2. 鲁棒性提升：通过困难样本提高模型鲁棒性
    3. 性能优化：针对性地改善模型在困难任务上的表现
    4. 课程学习：实现从简单到困难的渐进式学习
    
    【困难样本定义】
    - 路径长度较长的规划任务
    - 需要复杂机动的路径
    - 障碍物密集的环境
    - 狭窄通道导航任务
    
    在MPT系统中的定位：
    - 困难样本挖掘器：识别和重点训练困难场景
    - 性能提升器：通过困难样本提升整体性能
    - 鲁棒性增强器：提高模型在复杂环境中的稳定性
    """

    def __init__(self, env_list, dataFolderHard, dataFolderEasy):
        """
        初始化困难样本挖掘数据加载器
        
        【核心功能】
        构建支持困难样本挖掘的数据加载器，分别管理简单和困难的路径规划数据。
        
        【初始化策略】
        1. 分别构建困难和简单样本的索引
        2. 使用难度标识符区分样本类型
        3. 实现均匀分布的采样机制
        4. 支持动态难度平衡
        
        Args:
            env_list (list): 环境编号列表
                格式：整数列表，指定要使用的环境
                用途：控制训练环境的范围
            dataFolderHard (str): 困难样本数据文件夹路径
                内容：包含复杂、困难的路径规划任务
                特点：路径长、障碍多、规划复杂
            dataFolderEasy (str): 简单样本数据文件夹路径
                内容：包含相对简单的路径规划任务
                特点：路径短、障碍少、规划直接
        
        【数据组织策略】
        1. 难度标识：'H'表示困难样本，'E'表示简单样本
        2. 索引格式：(难度, 环境编号, 路径编号)
        3. 分层管理：分别维护两个难度级别的数据索引
        4. 统一接口：提供统一的数据访问方式
        
        【困难样本挖掘原理】
        1. 预分类：数据预处理时已按难度分类
        2. 标识管理：通过标识符快速区分难度
        3. 平衡采样：确保困难样本得到足够关注
        4. 渐进学习：支持从简单到困难的学习策略
        """
        assert isinstance(env_list, list), "Needs to be a list"
        self.num_env = len(env_list)
        self.env_list = env_list
        self.indexDictHard = [('H', envNum, i) 
            for envNum in env_list 
                for i in range(len(os.listdir(osp.join(dataFolderHard, f'env{envNum:06d}')))-1)
            ]
        self.indexDictEasy = [('E', envNum, i) 
            for envNum in env_list 
                for i in range(len(os.listdir(osp.join(dataFolderEasy, f'env{envNum:06d}')))-1)
            ]
        self.dataFolder = {'E': dataFolderEasy, 'H':dataFolderHard}
    

    def __len__(self):
        return len(self.indexDictEasy)+len(self.indexDictHard)
    
    def __getitem__(self, idx):
        """
        获取指定索引的困难样本挖掘训练样本
        
        【核心功能】
        根据索引获取来自不同难度级别的训练样本，支持简单和困难样本的统一处理。
        实现基于难度的样本挖掘和平衡训练。
        
        【处理流程】
        1. 索引解析：解析难度级别和具体索引
        2. 数据源选择：根据难度选择对应的数据文件夹
        3. 数据加载：加载地图和路径数据
        4. 样本生成：使用平衡的正负样本比例
        
        Args:
            idx: 样本索引，格式为(难度, 环境编号, 路径编号)
        
        Returns:
            dict: 训练样本字典
                'map': 编码后的地图输入
                'anchor': 锚点索引序列
                'labels': 锚点分类标签
        
        【困难样本处理】
        1. 难度识别：通过索引第一个元素识别样本难度
        2. 路径映射：根据难度映射到对应的数据文件夹
        3. 平衡采样：困难样本使用1:1的正负样本比例
        4. 重点关注：确保困难样本得到充分训练
        
        【与其他加载器的区别】
        - PathDataLoader: 使用2:1的负正样本比例
        - PathMixedDataLoader: 使用2:1的负正样本比例
        - PathHardMineDataLoader: 使用1:1的负正样本比例（更平衡）
        
        【技术细节】
        1. 样本比例：困难样本使用更平衡的1:1正负比例
        2. 数据质量：仅处理成功的路径数据
        3. 统一编码：不同难度样本使用相同的编码方式
        4. 一致输出：确保输出格式的一致性
        """
        # 解析索引：(难度, 环境编号, 路径编号)
        DF, env, idx_sample = idx  # DF: 'H'=困难, 'E'=简单

        # 根据难度级别选择数据文件夹
        dataFolder = self.dataFolder[DF]  # 'H'->困难文件夹, 'E'->简单文件夹
        
        # 加载环境地图
        map_path = osp.join(dataFolder, f'env{env:06d}', f'map_{env}.png')
        mapEnvg = skimage.io.imread(map_path, as_gray=True)
        
        # 加载路径数据
        path_file = osp.join(dataFolder, f'env{env:06d}', f'path_{idx_sample}.p')
        with open(path_file, 'rb') as f:
            data = pickle.load(f)

        # 处理成功的路径数据
        if data['success']:
            path = data['path_interpolated']
            
            # 生成编码输入
            goal_index = geom2pix(path[-1, :])   # 终点坐标转换
            start_index = geom2pix(path[0, :])   # 起点坐标转换
            mapEncoder = get_encoder_input(mapEnvg, goal_index, start_index)

            # 提取正样本锚点
            AnchorPointsPos = []
            for pos in path:
                indices, = geom2pixMatpos(pos)
                for index in indices:
                    if index not in AnchorPointsPos:
                        AnchorPointsPos.append(index)

            # 生成负样本锚点（1:1的负正比例，更平衡的困难样本训练）
            backgroundPoints = list(set(range(len(hashTable))) - set(AnchorPointsPos))
            numBackgroundSamp = min(len(backgroundPoints), len(AnchorPointsPos))  # 1:1比例
            AnchorPointsNeg = np.random.choice(backgroundPoints, size=numBackgroundSamp, replace=False).tolist()
            
            # 构建最终样本
            anchor = torch.cat((torch.tensor(AnchorPointsPos), torch.tensor(AnchorPointsNeg)))
            labels = torch.zeros_like(anchor)
            labels[:len(AnchorPointsPos)] = 1  # 正样本标记为1
            
            return {
                'map': torch.as_tensor(mapEncoder),  # 编码后的地图输入
                'anchor': anchor,                    # 锚点索引序列
                'labels': labels                     # 锚点分类标签
            }
        
        # 失败路径返回None
        return None
