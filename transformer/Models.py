"""
Models.py - Transformer架构的路径规划模型实现

【核心功能】
本模块实现了基于Transformer架构的路径规划神经网络模型，专门用于解决2D网格地图中的路径规划问题。
该实现将传统的NLP Transformer架构创新性地应用到空间路径规划任务中。

【技术特点】
1. 空间位置编码：将2D地图坐标转换为Transformer可处理的位置编码
2. 卷积特征提取：使用CNN提取地图的局部特征作为Transformer的输入
3. 端到端学习：直接从地图输入到路径输出的完整学习框架
4. 多尺度处理：支持不同尺寸地图的训练和推理

【在MPT系统中的作用】
- 作为核心的路径规划推理引擎
- 将地图信息编码为高维特征表示
- 输出每个位置的可通行性概率分布
- 与传统路径规划算法形成对比基准

技术栈：
- PyTorch 深度学习框架
- einops 张量操作库（用于维度重排）
- 自定义Transformer层（EncoderLayer, DecoderLayer）
- 卷积神经网络特征提取

使用场景：
- 机器人路径规划
- 游戏AI导航
- 自动驾驶路径决策
- 地图分析和可达性预测

【设计创新点】
1. 将NLP中的序列建模思想应用到空间规划
2. 结合CNN的局部特征提取和Transformer的全局建模能力
3. 动态位置编码适应不同尺寸的地图输入
4. 端到端的可微分路径规划框架

参考文献：
- Attention Is All You Need (Vaswani et al., 2017)
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformer.Layers import EncoderLayer, DecoderLayer, PoseWiseEncoderLayer

from einops.layers.torch import Rearrange
from einops import rearrange

# 【技术改进方向】位置编码优化策略
# 
# 当前实现将位置编码作为1D序列处理，更优的方案是：
# 1. 维护3D张量形式的位置编码 (batch, height, width, d_model)
# 2. 直接与卷积输出进行空间对应的加法操作
# 3. 避免维度重排带来的计算开销和内存碎片
# 4. 保持空间局部性，提升缓存效率
#
# 【实现考虑】
# - 需要重新设计位置编码的生成和索引机制
# - 考虑不同输入尺寸的动态适配
# - 平衡计算效率和代码复杂度

class PositionalEncoding(nn.Module):
    """
    PositionalEncoding - 2D空间位置编码模块
    
    【系统概述】
    将2D地图坐标转换为Transformer可理解的位置编码向量。不同于NLP中的1D序列位置编码，
    本模块专门处理2D空间位置信息，为路径规划任务提供空间感知能力。
    
    【核心创新】
    1. 2D到1D映射：将2D网格坐标映射为1D序列索引，保持空间邻接关系
    2. 正弦位置编码：使用sin/cos函数生成位置特征，具有平移不变性
    3. 动态尺寸适配：支持训练时固定尺寸和推理时可变尺寸
    4. 哈希索引表：高效的坐标到序列索引的转换机制
    
    【技术优势】
    - 位置编码具有良好的泛化性：相对位置关系保持一致
    - 支持任意尺寸地图：通过索引选择实现尺寸适配
    - 计算高效：预计算编码表，运行时仅需索引操作
    - 空间连续性：相邻位置的编码向量相似度高
    
    【应用场景】
    1. 训练阶段：固定尺寸地图的批量处理
    2. 推理阶段：可变尺寸地图的单样本处理
    3. 迁移学习：从小地图训练的模型应用到大地图
    
    【数学原理】
    位置编码公式：PE(pos, 2i) = sin(pos/10000^(2i/d_model))
                  PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    其中pos为位置索引，i为维度索引，d_model为编码维度
    
    在MPT框架中的定位：
    - 空间信息编码器：将几何信息转换为特征表示
    - Transformer输入预处理：为注意力机制提供位置感知
    - 多尺度适配器：处理不同尺寸地图的统一接口
    """
    def __init__(self, d_hid, n_position, train_shape):
        """
        初始化2D空间位置编码模块
        
        【核心功能】
        构建用于2D地图的位置编码系统，包括坐标哈希表和正弦位置编码表的预计算。
        
        【初始化策略】
        1. 预计算最大尺寸的位置编码表，避免运行时计算开销
        2. 构建2D坐标到1D索引的哈希映射表
        3. 分别准备训练和推理阶段的编码表
        
        Args:
            d_hid (int): 注意力特征的维度，通常等于d_model
                范围：通常为64, 128, 256, 512等2的幂次
                作用：决定位置编码向量的表达能力和计算复杂度
            n_position (int): 考虑的最大位置数量，等于max_height × max_width
                计算：支持的最大地图尺寸的像素总数
                约束：必须是完全平方数，以便构建方形网格
                示例：10000表示支持100×100的最大地图
            train_shape (tuple): 训练数据的2D形状 (height, width)
                用途：固定训练时的地图尺寸，优化内存使用
                限制：训练时所有地图必须具有相同尺寸
                示例：(64, 64)表示训练使用64×64的地图
        
        【实现原理】
        1. 计算网格边长：sqrt(n_position)得到正方形网格的边长
        2. 注册缓冲区：使用register_buffer确保编码表随模型移动到GPU
        3. 预计算编码：避免每次前向传播时重复计算位置编码
        
        【内存优化】
        - 使用register_buffer而非Parameter：编码表不需要梯度更新
        - 预计算策略：一次计算，多次使用，提升推理效率
        - 按需索引：仅选择实际需要的位置编码，节省内存
        """
        super(PositionalEncoding, self).__init__()  # 调用父类nn.Module的初始化方法
        self.n_pos_sqrt = int(np.sqrt(n_position))  # 计算网格边长：将总位置数开方得到正方形网格的边长
        self.train_shape = train_shape  # 保存训练时的地图形状，用于后续的位置编码选择
        # Not a parameter
        self.register_buffer('hashIndex', self._get_hash_table(n_position))  # 注册哈希索引表为缓冲区：不参与梯度更新但会随模型移动设备
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))  # 注册完整的正弦位置编码表
        self.register_buffer('pos_table_train', self._get_sinusoid_encoding_table_train(n_position, train_shape))  # 注册训练专用的位置编码表

    def _get_hash_table(self, n_position):
        """
        构建1D索引到2D网格坐标的哈希映射表
        
        【核心功能】
        将连续的1D位置索引重新排列为2D网格形式，建立空间坐标与序列索引的双向映射关系。
        
        【算法原理】
        使用行优先(row-major)顺序将1D索引映射到2D坐标：
        - 索引0 -> (0,0), 索引1 -> (0,1), ..., 索引w-1 -> (0,w-1)
        - 索引w -> (1,0), 索引w+1 -> (1,1), ..., 索引2w-1 -> (1,w-1)
        - 以此类推...
        
        Args:
            n_position (int): 网格上的总位置数量
                要求：必须是完全平方数
                计算：grid_size = sqrt(n_position)
                示例：n_position=100 -> 10×10网格
        
        Returns:
            torch.Tensor: 形状为(sqrt(n_position), sqrt(n_position))的2D张量
                内容：每个位置存储对应的1D索引值
                用途：通过2D坐标快速查找对应的1D位置编码索引
        
        【技术实现】
        使用einops.rearrange进行高效的张量重塑：
        - '(h w) -> h w'：将1D张量按指定的h,w维度重塑为2D
        - 自动推断维度：根据总元素数和指定维度计算另一维度
        - 内存高效：原地操作，无额外内存分配
        
        【应用场景】
        1. 坐标索引：根据(row, col)坐标快速找到位置编码
        2. 区域选择：选择地图中特定矩形区域的位置编码
        3. 空间采样：支持不同尺寸地图的位置编码提取
        """
        return rearrange(torch.arange(n_position), '(h w) -> h w', h=int(np.sqrt(n_position)), w=int(np.sqrt(n_position)))  # 使用einops将1D索引序列重排为2D网格：创建从0到n_position-1的连续索引，按行优先顺序排列成正方形网格

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """
        生成正弦位置编码表
        
        【核心功能】
        基于Transformer原始论文的位置编码方案，生成具有周期性特征的位置编码向量。
        每个位置对应一个d_hid维的编码向量，相邻位置的编码向量具有相似性。
        
        【数学原理】
        位置编码公式：
        - PE(pos, 2i) = sin(pos / 10000^(2i/d_hid))     # 偶数维度使用sin
        - PE(pos, 2i+1) = cos(pos / 10000^((2i+1)/d_hid))   # 奇数维度使用cos
        
        其中：
        - pos: 位置索引 (0 到 n_position-1)
        - i: 编码维度索引 (0 到 d_hid/2-1)
        - 10000: 基础周期，控制不同维度的频率范围
        
        Args:
            n_position (int): 最大位置数量
                范围：通常为序列最大长度或地图最大像素数
                用途：决定编码表的行数
            d_hid (int): 隐藏层维度，即位置编码的特征维度
                要求：通常为偶数，便于sin/cos配对
                影响：维度越高，位置表示越精细
        
        Returns:
            torch.FloatTensor: 形状为(1, n_position, d_hid)的位置编码表
                维度说明：
                - 第1维：batch维度，固定为1
                - 第2维：位置索引，对应n_position个位置
                - 第3维：编码特征，每个位置的d_hid维编码向量
        
        【算法步骤】
        1. 计算角度向量：对每个位置计算所有维度的角度值
        2. 应用三角函数：偶数维度用sin，奇数维度用cos
        3. 构建编码表：组装成完整的位置编码矩阵
        4. 添加batch维度：适配Transformer的输入格式
        
        【技术特性】
        - 周期性：不同频率的sin/cos组合提供丰富的位置模式
        - 相对位置感知：相邻位置的编码向量相似度高
        - 外推能力：训练时未见过的位置也能获得合理编码
        - 计算高效：预计算后仅需索引操作
        
        【性能考虑】
        - 时间复杂度：O(n_position × d_hid)
        - 空间复杂度：O(n_position × d_hid)
        - TODO优化：当前使用numpy实现，可改为纯torch提升GPU利用率
        """
        # TODO: 使用torch替代numpy实现，提升GPU计算效率
        # 当前numpy实现需要CPU-GPU数据传输，影响性能

        def get_position_angle_vec(position):
            """计算单个位置的角度向量"""
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]  # 对每个维度计算角度：position除以10000的幂次，幂次由维度索引决定

        # 【步骤1】生成所有位置的角度矩阵
        # 形状：(n_position, d_hid)，每行对应一个位置的角度向量
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])  # 为每个位置生成角度向量，组成完整的角度矩阵
        
        # 【步骤2】应用三角函数生成位置编码
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数维度(2i)使用sin：对所有位置的偶数维度应用正弦函数
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数维度(2i+1)使用cos：对所有位置的奇数维度应用余弦函数
        
        # 【步骤3】转换为torch张量并添加batch维度
        # [None,:]等价于unsqueeze(0)，添加batch维度适配模型输入
        return torch.FloatTensor(sinusoid_table[None,:])  # 转换numpy数组为torch张量，并在第0维添加batch维度
    
    def _get_sinusoid_encoding_table_train(self, n_position, train_shape):
        """
        生成训练专用的位置编码表
        
        【核心功能】
        从完整的位置编码表中提取训练时实际使用的部分，优化内存使用和计算效率。
        针对固定尺寸的训练数据进行预处理，避免运行时的重复索引操作。
        
        【设计假设】
        1. 训练数据来源假设：所有训练样本具有相同的固定地图尺寸
        2. 地图形状假设：支持矩形地图，不限于正方形（与注释描述不完全一致）
        3. 批量处理假设：训练时批量数据具有统一的空间维度
        
        【优化策略】
        通过预选择训练时需要的位置编码，减少：
        - 运行时索引计算开销
        - GPU内存占用（仅加载必要的编码向量）
        - 数据传输延迟（预计算的编码表直接可用）
        
        Args:
            n_position (int): 完整编码表的最大位置数量
                用途：验证train_shape是否在支持范围内
                约束：必须 >= train_shape[0] × train_shape[1]
            train_shape (tuple): 训练地图的2D尺寸 (height, width)
                格式：(行数, 列数)
                限制：height × width <= n_position
                示例：(32, 32)表示32×32的训练地图
        
        Returns:
            torch.Tensor: 形状为(1, height×width, d_hid)的训练位置编码表
                优化：仅包含训练时实际需要的位置编码
                用途：训练阶段直接使用，无需运行时索引
        
        【实现原理】
        1. 区域选择：从哈希表中选择train_shape对应的矩形区域
        2. 维度重排：将2D坐标索引转换为1D序列索引
        3. 编码提取：根据索引从完整编码表中提取对应的编码向量
        
        【技术细节】
        - 使用切片操作：hashIndex[:height, :width]高效选择矩形区域
        - einops重排：'h w -> (h w)'将2D索引展平为1D
        - torch.index_select：沿指定维度按索引选择张量元素
        
        【内存效率分析】
        - 原始编码表：(1, n_position, d_hid)
        - 训练编码表：(1, height×width, d_hid)
        - 内存节省：(n_position - height×width) × d_hid × 4字节
        """
        # 【步骤1】选择训练地图对应的坐标区域
        # 从完整的哈希索引表中切片出训练尺寸的矩形区域
        selectIndex = rearrange(
            self.hashIndex[:train_shape[0], :train_shape[1]],  # 从哈希表中切片选择训练尺寸的矩形区域：前train_shape[0]行，前train_shape[1]列
            'h w -> (h w)'  # 将2D坐标索引展平为1D序列：使用einops将2D索引矩阵重排为1D向量
        )
        
        # 【步骤2】根据索引提取对应的位置编码
        # dim=1表示沿位置维度进行索引选择
        # 结果形状：(1, train_height×train_width, d_hid)
        return torch.index_select(self.pos_table, dim=1, index=selectIndex)  # 根据选择的索引从完整位置编码表中提取对应的编码向量

    def forward(self, x, conv_shape=None):
        """
        位置编码前向传播函数
        
        【核心功能】
        根据输入特征张量的形状，动态选择并添加对应的位置编码。支持训练和推理两种模式，
        实现灵活的位置编码注入机制。
        
        【双模式设计】
        1. 训练模式(conv_shape=None)：使用随机位置采样，增强模型泛化能力
        2. 推理模式(conv_shape!=None)：使用精确位置映射，确保推理一致性
        
        【算法原理】
        通过将预计算的位置编码与输入特征相加，为Transformer提供空间位置信息。
        位置编码采用残差连接方式，保持原始特征的同时注入位置信息。
        
        Args:
            x (torch.Tensor): 输入特征张量，形状为(batch, seq_len, d_model)
                来源：通常是卷积特征提取后的patch embedding
                约束：seq_len必须对应有效的2D空间区域
                示例：(4, 1024, 512)表示4个样本，每个1024个位置，512维特征
            conv_shape (tuple, optional): 卷积输出的2D形状(height, width)
                训练时：None，启用随机位置采样策略
                推理时：(H, W)，指定精确的空间尺寸
                用途：确定需要选择的位置编码范围
        
        Returns:
            torch.Tensor: 添加位置编码后的特征张量
                形状：与输入x相同(batch, seq_len, d_model)
                内容：原始特征 + 对应位置的位置编码
                特性：保持特征分布的同时注入空间信息
        
        【训练模式详解】(conv_shape=None)
        目的：通过随机位置采样提升模型对不同位置的适应能力
        
        实现步骤：
        1. 随机采样起始坐标：在有效范围内随机选择左上角位置
        2. 计算采样范围：startH到startH+train_height, startW到startW+train_width
        3. 提取位置编码：根据采样区域从完整编码表中选择对应编码
        4. 梯度分离：使用clone().detach()避免位置编码参与梯度更新
        
        【推理模式详解】(conv_shape!=None)
        目的：为特定尺寸的输入提供精确的位置编码
        
        实现步骤：
        1. 直接映射：根据conv_shape确定需要的位置编码范围
        2. 索引选择：从左上角(0,0)开始选择conv_shape大小的区域
        3. 编码注入：将选中的位置编码直接加到输入特征上
        
        【技术细节】
        - 随机采样范围：[0, n_pos_sqrt-train_shape[0])确保采样区域不越界
        - 索引重排：'h w -> (h w)'将2D坐标转换为1D序列索引
        - 残差连接：x + pos_encoding保持特征的原始信息
        - 内存优化：使用index_select避免复制整个编码表
        
        【性能考虑】
        - 时间复杂度：O(seq_len)，主要是索引选择操作
        - 空间复杂度：O(1)，仅创建索引张量
        - GPU友好：所有操作都是张量操作，支持并行计算
        
        【使用示例】
        训练时：pos_enc(features)  # 随机位置采样
        推理时：pos_enc(features, (32, 32))  # 固定32x32尺寸
        """
        if conv_shape is None:  # 判断是否为训练模式：conv_shape为None表示训练模式
            # 【训练模式】随机位置采样策略
            # 
            # 在支持的最大地图范围内随机选择一个train_shape大小的区域
            # 这种随机性有助于模型学习位置不变的特征表示
            startH, startW = torch.randint(0, self.n_pos_sqrt-self.train_shape[0], (2,))  # 随机生成起始坐标：在有效范围内随机选择左上角位置，确保采样区域不越界
            
            # 根据随机起始位置选择对应的位置编码区域
            selectIndex = rearrange(
                self.hashIndex[startH:startH+self.train_shape[0], startW:startW+self.train_shape[1]],  # 从哈希表中切片选择随机位置开始的训练尺寸区域
                'h w -> (h w)'  # 将2D区域索引转换为1D序列索引：使用einops展平2D索引为1D向量
                )
            
            # 添加位置编码，使用detach()防止位置编码参与梯度更新
            return x + torch.index_select(self.pos_table, dim=1, index=selectIndex).clone().detach()  # 残差连接：将选中的位置编码加到输入特征上，clone().detach()确保位置编码不参与反向传播

        # 【推理模式】精确位置映射
        # 
        # 根据实际输入尺寸选择对应的位置编码
        # assert x.shape[0]==1, "仅支持单样本推理"  # 原注释：批量推理的限制
        selectIndex = rearrange(self.hashIndex[:conv_shape[0], :conv_shape[1]], 'h w -> (h w)')  # 根据卷积输出形状选择对应的位置编码索引：从左上角开始选择conv_shape大小的区域并展平
        return x + torch.index_select(self.pos_table, dim=1, index=selectIndex)  # 残差连接：将精确选择的位置编码加到输入特征上


class Encoder(nn.Module):
    """
    Encoder - Transformer编码器用于路径规划
    
    【系统概述】
    将2D地图输入转换为高维特征表示的编码器模块。结合卷积神经网络的局部特征提取能力
    和Transformer的全局建模能力，为路径规划任务提供强大的地图理解能力。
    
    【核心创新】
    1. CNN-Transformer混合架构：先用CNN提取局部特征，再用Transformer建模全局关系
    2. 自适应patch embedding：将地图划分为patches，每个patch作为一个token
    3. 空间位置编码：为每个patch注入2D空间位置信息
    4. 多层自注意力：通过堆叠的注意力层捕获复杂的空间依赖关系
    
    【技术优势】
    - 全局感受野：每个位置都能感知到整个地图的信息
    - 并行计算：相比RNN，Transformer支持高效的并行训练
    - 长距离依赖：自注意力机制天然适合建模远距离的空间关系
    - 可解释性：注意力权重可以可视化，理解模型的决策过程
    
    【应用场景】
    1. 地图特征提取：将原始地图转换为语义丰富的特征表示
    2. 障碍物识别：识别地图中的可通行和不可通行区域
    3. 路径可达性分析：评估不同位置之间的连通性
    4. 全局路径规划：为后续的路径搜索提供指导信息
    
    【架构设计】
    输入地图 -> CNN特征提取 -> Patch Embedding -> 位置编码 -> 多层Transformer -> 输出特征
    
    在MPT框架中的定位：
    - 特征提取器：将地图转换为Transformer可处理的token序列
    - 全局建模器：捕获地图中的长距离空间依赖关系
    - 语义编码器：为路径规划提供高级语义特征
    """

    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape):
        """
        初始化Transformer编码器
        
        【核心功能】
        构建完整的编码器架构，包括卷积特征提取、位置编码、多层Transformer和输出规范化。
        
        【架构设计原则】
        1. 渐进式特征提取：从低级视觉特征到高级语义特征
        2. 多尺度感受野：通过卷积和注意力机制捕获不同尺度的特征
        3. 残差连接：保证深层网络的训练稳定性
        4. 正则化策略：通过dropout和layer norm防止过拟合
        
        Args:
            n_layers (int): Transformer层数
                范围：通常1-12层，平衡性能和计算成本
                影响：层数越多，模型表达能力越强，但计算成本越高
                推荐：路径规划任务通常4-8层效果较好
            n_heads (int): 多头注意力的头数
                约束：d_model必须能被n_heads整除
                范围：通常4, 8, 16等
                作用：不同的头关注不同类型的空间关系
            d_k (int): 每个Key向量的维度
                计算：通常等于d_model // n_heads
                作用：决定注意力计算的精度和复杂度
                平衡：维度过小影响表达能力，过大增加计算成本
            d_v (int): 每个Value向量的维度
                设置：通常与d_k相等
                用途：决定注意力输出的特征维度
            d_model (int): 模型的主要特征维度
                范围：通常128, 256, 512, 1024等2的幂次
                作用：决定模型的整体表达能力
                权衡：更大的维度提供更强表达能力但需要更多计算资源
            d_inner (int): 前馈网络的隐藏层维度
                设置：通常为d_model的2-4倍
                作用：在注意力层之间提供非线性变换
                示例：d_model=512时，d_inner通常设为2048
            pad_idx (int): 填充标记的索引
                用途：处理变长序列时的填充位置
                注意：当前实现中可能未完全使用
                TODO：需要完善填充处理逻辑
            dropout (float): Dropout概率
                范围：0.0-0.5，通常0.1-0.3
                作用：防止过拟合，提升泛化能力
                调优：训练数据少时可适当增大
            n_position (int): 支持的最大位置数
                计算：max_height × max_width
                限制：决定模型能处理的最大地图尺寸
                示例：10000支持100×100的地图
            train_shape (tuple): 训练时的地图形状
                格式：(height, width)
                用途：优化训练时的内存使用
                约束：所有训练样本必须具有相同尺寸
        
        【网络架构详解】
        1. 卷积特征提取器(to_patch_embedding)：
           - 第1层：2->6通道，5×5卷积，提取基础特征
           - 第2层：6->16通道，5×5卷积，增强特征表达
           - 第3层：16->d_model通道，5×5卷积+步长5，生成patch embedding
           - 池化层：2×2最大池化，降低空间分辨率
           - 激活函数：ReLU，引入非线性
        
        2. 维度重排(reorder_dims)：
           - 功能：将4D卷积输出转换为3D序列格式
           - 变换：(batch, channels, height, width) -> (batch, height×width, channels)
           - 目的：适配Transformer的序列输入格式
        
        3. 位置编码(position_enc)：
           - 类型：2D空间位置编码
           - 作用：为每个patch注入位置信息
           - 特点：支持训练和推理的不同模式
        
        4. 多层Transformer(layer_stack)：
           - 结构：n_layers个相同的EncoderLayer
           - 功能：通过自注意力建模全局空间关系
           - 连接：残差连接保证梯度流动
        
        5. 层归一化(layer_norm)：
           - 位置：在Transformer层之前
           - 作用：稳定训练，加速收敛
           - 参数：eps=1e-6防止除零错误
        """
        super().__init__()  # 调用父类nn.Module的初始化方法
        # Convert the image to and input embedding.
        # NOTE: This is one place where we can add convolution networks.
        # Convert the image to linear model

        # NOTE: Padding of 3 is added to the final layer to ensure that 
        # the output of the network has receptive field across the entire map.
        # NOTE: pytorch doesn't have a good way to ensure automatic padding. This
        # allows only for a select few map sizes to be solved using this method.
        self.to_patch_embedding = nn.Sequential(  # 构建卷积特征提取序列：将2D地图转换为patch embeddings
            nn.Conv2d(2, 6, kernel_size=5),  # 第一层卷积：2输入通道->6输出通道，5x5卷积核，提取基础特征
            nn.MaxPool2d(kernel_size=2),  # 最大池化：2x2池化核，降低空间分辨率，增强特征鲁棒性
            nn.ReLU(),  # ReLU激活函数：引入非线性，增强模型表达能力
            nn.Conv2d(6, 16, kernel_size=5),  # 第二层卷积：6->16通道，5x5卷积核，进一步提取特征
            nn.MaxPool2d(kernel_size=2),  # 第二次最大池化：继续降低空间分辨率
            nn.ReLU(),  # 第二个ReLU激活
            nn.Conv2d(16, d_model, kernel_size=5, stride=5, padding=3)  # 最终卷积层：16->d_model通道，5x5卷积核，步长5，填充3，生成最终的patch embedding
        )

        self.reorder_dims = Rearrange('b c h w -> b (h w) c')  # 维度重排：将4D卷积输出(batch,channels,height,width)转换为3D序列格式(batch,seq_len,channels)
        # Position Encoding.
        # NOTE: Current setup for adding position encoding after patch Embedding.
        self.position_enc = PositionalEncoding(d_model, n_position=n_position, train_shape=train_shape)  # 初始化位置编码模块：为每个patch添加空间位置信息

        self.dropout = nn.Dropout(p=dropout)  # Dropout层：随机置零部分神经元，防止过拟合
        self.layer_stack = nn.ModuleList([  # 构建多层Transformer编码器堆栈
            EncoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)  # 创建单个编码器层：包含自注意力和前馈网络
            for _ in range(n_layers)  # 重复n_layers次，构建深层网络
        ])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化：标准化特征分布，稳定训练过程，eps防止除零错误
        

    def forward(self, input_map, returns_attns=False):
        """
        编码器前向传播函数
        
        【核心功能】
        将2D地图输入转换为高维特征表示，通过CNN特征提取、位置编码和多层Transformer
        处理，生成包含全局空间关系的特征向量。
        
        【处理流程】
        1. CNN特征提取：将原始地图转换为patch embeddings
        2. 维度重排：适配Transformer的序列输入格式
        3. 位置编码注入：为每个patch添加空间位置信息
        4. 正则化处理：应用dropout和layer normalization
        5. 多层Transformer：通过自注意力建模全局依赖关系
        
        Args:
            input_map (torch.Tensor): 输入地图张量，形状为(batch, channels, height, width)
                通道说明：通常包含障碍物信息、起点终点标记等
                尺寸要求：必须与训练时的尺寸兼容
                数据范围：通常归一化到[0,1]或[-1,1]
            returns_attns (bool, optional): 是否返回每层的自注意力权重
                默认False：仅返回编码特征，节省内存
                设为True：同时返回注意力权重，用于可视化分析
                用途：调试、可解释性分析、注意力可视化
        
        Returns:
            tuple: 包含编码输出和可选的注意力权重
                enc_output (torch.Tensor): 编码后的特征表示
                    形状：(batch, seq_len, d_model)
                    内容：每个位置的高维语义特征
                    用途：后续路径规划或分类任务的输入
                enc_slf_attn_list (list, optional): 各层自注意力权重列表
                    仅当returns_attns=True时返回
                    用途：分析模型关注的空间区域
        
        【技术实现细节】
        1. 特征提取阶段：
           - 卷积操作逐步降低空间分辨率，增加特征维度
           - 记录卷积输出的空间形状，用于位置编码
        
        2. 位置编码阶段：
           - 训练模式：使用随机位置采样增强泛化
           - 推理模式：使用精确位置映射保证一致性
        
        3. Transformer处理：
           - 层归一化在前：Pre-LN结构，提升训练稳定性
           - 无掩码注意力：允许全局信息交互
           - 残差连接：保证梯度流动和特征保持
        
        【性能考虑】
        - 内存使用：O(batch_size × seq_len × d_model)
        - 计算复杂度：O(seq_len² × d_model)，主要来自自注意力
        - 并行化：CNN和Transformer层都支持高效并行计算
        """
        enc_slf_attn_list = []  # 初始化自注意力权重列表：用于存储各层的注意力权重（当前实现中未使用）
        enc_output = self.to_patch_embedding(input_map)  # 卷积特征提取：将输入地图通过CNN转换为patch embeddings
        conv_map_shape = enc_output.shape[-2:]  # 记录卷积输出的空间形状：获取height和width维度，用于位置编码
        enc_output = self.reorder_dims(enc_output)  # 维度重排：将4D张量转换为3D序列格式，适配Transformer输入

        if self.training:  # 判断是否为训练模式
            enc_output = self.position_enc(enc_output)  # 训练模式：使用随机位置采样的位置编码
        else:  # 推理模式
            enc_output = self.position_enc(enc_output, conv_map_shape)  # 推理模式：使用精确位置映射的位置编码
    
        enc_output = self.dropout(enc_output)  # 应用Dropout：随机置零部分特征，防止过拟合
        enc_output = self.layer_norm(enc_output)  # 层归一化：标准化特征分布，稳定后续Transformer层的输入

        for enc_layer in self.layer_stack:  # 遍历所有编码器层
            enc_output = enc_layer(enc_output, slf_attn_mask=None)  # 通过编码器层：应用自注意力和前馈网络，slf_attn_mask=None表示无掩码
        
        if returns_attns:  # 如果需要返回注意力权重
            return enc_output, enc_slf_attn_list  # 返回编码输出和注意力权重列表
        return enc_output,  # 仅返回编码输出，逗号表示返回单元素元组

def wrap_to_pi(x):
    # x in radians, wrap to [-pi, pi]
    return (x + torch.pi) % (2*torch.pi) - torch.pi

# class PoseTokenInjector(nn.Module):
#     """
#     将 SE2 pose -> pose_token。
#     可选：从 feature_map 上插值取局部特征并与 pose embedding concat。
#     输入:
#       - start_goal: (B, 2, 4) [x(m), y(m), cos(yaw), sin(yaw)]
#       - feature_map: (B, C, Hf, Wf)  # 用于插值（可选）
#       - map_bounds: (xmin, xmax, ymin, ymax) in meters, e.g. (-5, 5, -5, 5)
#       - d_model: 输出维度
#     输出:
#       - pose_tokens: (B, 2, d_model)
#     """
#     def __init__(self, d_model, in_pose_dim=4, map_feat_dim=None,
#                  hidden=128, use_map_feature=False):
#         super().__init__()
#         self.use_map_feature = use_map_feature
#         self.map_feat_dim = map_feat_dim if use_map_feature else 0

#         # simple MLP for pose embedding from [x_norm, y_norm, cos, sin]
#         self.pose_mlp = nn.Sequential(
#             nn.Linear(in_pose_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, d_model - self.map_feat_dim)  # if concat with map feature
#         )
#         # if using sampled map features, project them to leftover dims and concat
#         if self.use_map_feature:
#             self.map_feat_proj = nn.Sequential(
#                 nn.Linear(self.map_feat_dim, self.map_feat_dim),
#                 nn.ReLU()
#             )
#         # optional learned tokens to add (start_token, goal_token) for trainable bias
#         self.start_token = nn.Parameter(torch.randn(1, d_model))
#         self.goal_token  = nn.Parameter(torch.randn(1, d_model))

#     def meters_to_grid_norm(self, x_m, y_m, map_bounds):
#         """
#         map_bounds: tuple (xmin, xmax, ymin, ymax)
#         return grid coords in range [-1, 1] (for grid_sample)
#         Note: grid_sample expects coords as (x_norm, y_norm) with x horizontal, y vertical,
#               and y positive downwards. You must adjust depending on your world->image convention.
#         """
#         xmin, xmax, ymin, ymax = map_bounds
#         # normalize to [0,1]
#         x01 = (x_m - xmin) / (xmax - xmin)
#         y01 = (y_m - ymin) / (ymax - ymin)
#         # convert to [-1,1]
#         x_norm = x01 * 2.0 - 1.0
#         # image y is usually top->bottom; if your world y is bottom->top you may flip:
#         y_norm = y01 * 2.0 - 1.0  # flip to match grid_sample convention (optional)
#         return x_norm, y_norm

#     def sample_map_features(self, feature_map, x_norm, y_norm):
#         """
#         feature_map: (B, C, Hf, Wf)
#         x_norm, y_norm: (B, 2) normalized to [-1,1] (2 poses per batch)
#         returns sampled_feats: (B, 2, C)
#         Implementation: use grid_sample with shape (B, 2, 1, 2) grid
#         """
#         B, C, Hf, Wf = feature_map.shape
#         # grid_sample expects grid shape (B, H_out, W_out, 2) - we'll do (B,2,1,2)
#         # Build grid: for each batch, 2 points -> grid shape (B, 2, 1, 2)
#         xg = x_norm.unsqueeze(-1)  # (B, 2, 1)
#         yg = y_norm.unsqueeze(-1)  # (B, 2, 1)
#         # grid_sample uses ordering (x, y) per point, create (B, 2, 1, 2)
#         grid = torch.stack([xg, yg], dim=-1)  # (B, 2, 1, 2)
#         # grid_sample will return (B, C, 2, 1) when sampling 2 points
#         sampled = F.grid_sample(feature_map, grid, mode='bilinear', align_corners=True)  # (B, C, 2, 1)
#         sampled = sampled.squeeze(-1).permute(0, 2, 1)  # -> (B, 2, C)
#         return sampled

#     def forward(self, start_goal, feature_map=None, map_bounds=(-5,5,-5,5)):
#         # start_goal: (B, 2, 4)  [x(m), y(m), cos(yaw), sin(yaw)]
#         B = start_goal.shape[0]
#         # split
#         xy = start_goal[..., :2]  # (B,2,2)
#         cs = start_goal[..., 2:]  # (B,2,2) cos,sin

#         # convert meters -> normalized grid coords for sampling
#         # xy[...,0] = x, xy[...,1] = y
#         x = xy[..., 0]  # (B,2)
#         y = xy[..., 1]
#         x_norm, y_norm = self.meters_to_grid_norm(x, y, map_bounds)  # each (B,2)

#         # basic pose input for mlp: we choose normalized x,y not raw meters
#         # stack as [x_norm, y_norm, cos, sin]
#         pose_in = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1), cs], dim=-1)  # (B,2,4)
#         # flatten to (B*2, 4) for mlp
#         pose_flat = pose_in.reshape(B*2, -1)
#         pose_emb = self.pose_mlp(pose_flat)  # (B*2, d_part)
#         pose_emb = pose_emb.reshape(B, 2, -1)  # (B,2,d_part)

#         if self.use_map_feature:
#             assert feature_map is not None, "feature_map required when use_map_feature=True"
#             sampled = self.sample_map_features(feature_map, x_norm, y_norm)  # (B,2,C)
#             sampled_proj = self.map_feat_proj(sampled.reshape(B*2, -1)).reshape(B,2,-1)  # (B,2,d_mapfeat)
#             fused = torch.cat([pose_emb, sampled_proj], dim=-1)  # (B,2,d_model_map)
#         else:
#             fused = pose_emb  # (B,2,d_model_part)

#         # if fused dim < d_model, pad with zeros or linear project; we'll add learned tokens
#         # final projection to exactly d_model
#         # Here we assume pose_mlp output + sampled_proj already equals desired final d_model
#         # add learned bias tokens
#         start_tok = fused[:,0,:] + self.start_token  # (B, d_model)
#         goal_tok  = fused[:,1,:] + self.goal_token
#         pose_tokens = torch.stack([start_tok, goal_tok], dim=1)  # (B,2,d_model)
#         return pose_tokens

class PoseTokenInjector(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.pose_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, start_goal, map_bounds=(-5,5,-5,5)):
        # start_goal: (B, 2, 4) [x, y, cos, sin]
        B = start_goal.shape[0]
        xy = start_goal[..., :2]
        cs = start_goal[..., 2:]
        
        # normalize
        xmin, xmax, ymin, ymax = map_bounds
        x_norm = (xy[..., 0] - xmin) / (xmax - xmin) * 2 - 1
        y_norm = (xy[..., 1] - ymin) / (ymax - ymin) * 2 - 1
        
        pose_in = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1), cs], dim=-1)  # (B, 2, 4)
        pose_flat = pose_in.reshape(B*2, -1)
        pose_emb = self.pose_mlp(pose_flat).reshape(B, 2, -1)  # (B, 2, d_model)
        
        return pose_emb


class UnevenEncoder(nn.Module):
    """    
    【架构设计】
    输入地图 -> CNN特征提取 -> Patch Embedding -> 位置编码 -> 多层Transformer -> 输出特征
       |                                     ^
       v                                     |
       ----> Pose Token Injector(delete) ---->
    """

    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape):
        super().__init__()  # 调用父类nn.Module的初始化方法
        
        self.map_fe = nn.Sequential(
            # Block 1
            nn.Conv2d(6, d_model//8, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model//8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 50×50
            
            # Block 2
            nn.Conv2d(d_model//8, d_model//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model//4),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25×25
            
            # Block 3
            nn.Conv2d(d_model//4, d_model//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model//2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12×12
            
            # Block 4
            nn.Conv2d(d_model//2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),             
        )

        self.reorder_dims = Rearrange('b c h w -> b (h w) c')  # 维度重排：将4D卷积输出(batch,channels,height,width)转换为3D序列格式(batch,seq_len,channels)
        
        # self.pose_injector = PoseTokenInjector(d_model, in_pose_dim=4, map_feat_dim=d_model, hidden=128, use_map_feature=True)  # 初始化姿态注入器：将SE2姿态转换为pose tokens，并结合局部特征
        # self.pose_injector = PoseTokenInjector(d_model)  # 初始化姿态注入器：将SE2姿态转换为pose tokens，并结合局部特征

        # Position Encoding.
        # NOTE: Current setup for adding position encoding after patch Embedding.
        self.position_enc = PositionalEncoding(d_model, n_position=n_position, train_shape=train_shape)  # 初始化位置编码模块：为每个patch添加空间位置信息

        self.dropout = nn.Dropout(p=dropout)  # Dropout层：随机置零部分神经元，防止过拟合
        self.layer_stack = nn.ModuleList([  # 构建多层Transformer编码器堆栈
            PoseWiseEncoderLayer(d_model, d_inner, n_heads, d_k, d_v, yaw_bins=36, dropout=dropout)  # 创建单个编码器层：包含自注意力和前馈网络
            # EncoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)  # 创建单个编码器层：包含自注意力和前馈网络
            for _ in range(n_layers)  # 重复n_layers次，构建深层网络
        ])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化：标准化特征分布，稳定训练过程，eps防止除零错误
        

    # def forward(self, input_map, input_pose, returns_attns=False):
    def forward(self, input_map, returns_attns=False):
        enc_slf_attn_list = []

        # 1) CNN -> map tokens
        map_feat = self.map_fe(input_map)                        # [B, D, Hf, Wf]
        conv_map_shape = map_feat.shape[-2:]
        map_tokens = self.reorder_dims(map_feat)                 # [B, N_map, D]
        map_tokens = self.position_enc(map_tokens, conv_shape=conv_map_shape if not self.training else None)

        # # 2) pose tokens (轻量 injector)
        # pose_tokens = self.pose_injector(input_pose, map_bounds=(-5,5,-5,5))  # [B, 2, D]

        # optional normalization/dropout (保留)
        map_tokens = self.dropout(self.layer_norm(map_tokens))
        # pose_tokens = self.dropout(self.layer_norm(pose_tokens))
        
        for enc_layer in self.layer_stack:
            # 逐层调用 encoder 层，收集自注意力权重
            map_tokens = enc_layer(map_tokens, slf_attn_mask=None)
            
        return map_tokens

        # # 3) 逐层调用 encoder 层，收集 aux（例如 yaw_logits）
        # yaw_logits_layers = []   # 用于聚合每层的 yaw logits（可选）
        # last_pose_ctx = None

        # for enc_layer in self.layer_stack:
        #     # 注意：轻量版 forward 返回 (out_map_tokens, aux_dict)
        #     map_tokens, aux = enc_layer(map_tokens, pose_tokens)   # map_tokens: [B, N_map, D]
        #     # aux 预期包含 'yaw_logits' : [B, N_map, K]，以及可能的 'pose_ctx'
        #     if aux is not None:
        #         if 'yaw_logits' in aux:
        #             yaw_logits_layers.append(aux['yaw_logits'])     # 收集以便后续聚合
        #         if 'pose_ctx' in aux:
        #             last_pose_ctx = aux['pose_ctx']                # 可选：保留最后一层的 pose 上下文

        # # 4) 聚合 yaw logits（两种常见方式：取平均或取最后一层）
        # if len(yaw_logits_layers) > 0:
        #     # yaw_logits_layers: list of tensors [B, N_map, K]
        #     # 方案 A：跨层求平均（更稳定）
        #     yaw_logits_stack = torch.stack(yaw_logits_layers, dim=0)  # [n_layers, B, N_map, K]
        #     yaw_logits_agg = yaw_logits_stack.mean(dim=0)            # [B, N_map, K]
        #     # 方案 B（可替换）：用最后一层： yaw_logits_agg = yaw_logits_layers[-1]
        # else:
        #     yaw_logits_agg = None

        # # 5) 最终编码输出：返回融合过 pose 的 map_tokens，以及聚合的 yaw_logits 供后续使用
        # if returns_attns:
        #     return map_tokens, last_pose_ctx, yaw_logits_agg, enc_slf_attn_list
        # return map_tokens, last_pose_ctx, yaw_logits_agg


class Decoder(nn.Module):
    """
    Decoder - Transformer解码器用于序列生成
    
    【系统概述】
    基于编码器输出和当前输入patch，逐步生成路径序列的解码器模块。
    采用标准的Transformer解码器架构，支持自回归的序列生成过程。
    
    【核心功能】
    1. 增量解码：基于已生成的路径片段预测下一个位置
    2. 交叉注意力：利用编码器的全局地图表示指导路径生成
    3. 位置感知：为当前patch注入序列位置信息
    4. 特征融合：结合局部patch特征和全局上下文信息
    
    【应用场景】
    1. 路径序列生成：逐步构建从起点到终点的完整路径
    2. 在线路径规划：根据当前位置和环境动态调整路径
    3. 路径优化：基于全局信息优化局部路径选择
    4. 多目标规划：支持多个中间目标点的路径规划
    
    【技术特点】
    - 自回归生成：每步预测都基于之前的生成结果
    - 注意力机制：同时关注编码器输出和解码器历史
    - 位置编码：维护序列中的时序和空间位置信息
    - 残差连接：保证深层网络的训练稳定性
    
    在MPT框架中的定位：
    - 序列生成器：将全局地图理解转换为具体路径序列
    - 决策模块：在每个时间步做出路径选择决策
    - 优化器：基于全局信息优化局部路径决策
    """

    def __init__(self, patch_size, n_layers, n_heads, d_k , d_v, d_model, d_inner, pad_idx, stride, n_position, dropout=0.1):
        """
        初始化Transformer解码器
        
        【核心功能】
        构建用于序列生成的解码器架构，包括patch embedding、位置编码、
        多层解码器和输出规范化组件。
        
        【架构设计】
        解码器采用标准的Transformer解码器结构，每层包含：
        1. 掩码自注意力：防止信息泄露，保证因果性
        2. 编码器-解码器注意力：利用编码器的全局信息
        3. 前馈网络：提供非线性变换能力
        4. 残差连接和层归一化：保证训练稳定性
        
        Args:
            patch_size (int): 每个patch的维度大小
                用途：定义输入patch的空间尺寸
                影响：决定解码器的输入分辨率
                示例：patch_size=16表示16×16的patch
            n_layers (int): 解码器层数
                范围：通常2-8层，平衡性能和计算成本
                作用：更多层提供更强的序列建模能力
                权衡：层数过多可能导致过拟合
            n_heads (int): 多头注意力的头数
                约束：d_model必须能被n_heads整除
                作用：不同头关注不同类型的依赖关系
                推荐：通常设为8或16
            d_k (int): Key向量的维度
                计算：通常等于d_model // n_heads
                作用：决定注意力计算的精度
                影响：影响模型的表达能力和计算复杂度
            d_v (int): Value向量的维度
                设置：通常与d_k相等
                用途：决定注意力输出的特征维度
                优化：可以与d_k不同以调节模型容量
            d_model (int): 模型的主要特征维度
                范围：通常256, 512, 1024等
                作用：决定模型的整体表达能力
                约束：必须与编码器的d_model一致
            d_inner (int): 前馈网络的隐藏层维度
                设置：通常为d_model的2-4倍
                作用：提供非线性变换能力
                示例：d_model=512时，d_inner=2048
            pad_idx (int): 填充标记的索引
                用途：处理变长序列的填充位置
                注意：在路径规划中可能用于无效位置标记
            stride (int): 卷积步长参数
                用途：控制patch提取的步长
                影响：决定输入序列的长度
                权衡：步长大减少计算量但可能丢失细节
            n_position (int): 支持的最大位置数
                用途：决定位置编码表的大小
                限制：限制可处理的最大序列长度
                设置：应大于预期的最大路径长度
            dropout (float): Dropout概率，默认0.1
                范围：0.0-0.5，通常0.1-0.3
                作用：防止过拟合，提升泛化能力
                调优：可根据数据量和模型复杂度调整
        
        【网络组件详解】
        1. Patch Embedding网络：
           - 卷积层1：1->6通道，4×4卷积核，提取基础特征
           - 卷积层2：6->16通道，4×4卷积核，增强特征表达
           - 维度重排：适配解码器的Key/Value兼容性
           - 线性层：将卷积特征映射到d_model维度
        
        2. 位置编码：
           - 类型：与编码器相同的2D空间位置编码
           - 参数：patch_size, stride, n_cols用于位置计算
           - 作用：为序列中的每个位置提供位置信息
        
        3. 多层解码器：
           - 结构：n_layers个相同的DecoderLayer
           - 功能：自注意力 + 交叉注意力 + 前馈网络
           - 连接：残差连接保证梯度流动
        
        4. 正则化组件：
           - Dropout：防止过拟合
           - LayerNorm：稳定训练过程
        """
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.to_patch_embedding = nn.Sequential(  # 构建patch embedding序列：将当前patch转换为特征表示
            nn.Conv2d(1, 6, kernel_size=4),  # 第一层卷积：1输入通道->6输出通道，4x4卷积核，提取patch基础特征
            nn.MaxPool2d(kernel_size=2),  # 最大池化：2x2池化核，降低空间分辨率
            nn.ReLU(),  # ReLU激活函数：引入非线性变换
            nn.Conv2d(6, 16, kernel_size=4),  # 第二层卷积：6->16通道，4x4卷积核，进一步提取特征
            nn.MaxPool2d(kernel_size=2),  # 第二次最大池化：继续降低分辨率
            nn.ReLU(),  # 第二个ReLU激活
            Rearrange('(b pad) k p1 p2 -> b pad (k p1 p2)', pad=1),  # 维度重排：确保与解码器Key/Value对的兼容性，将卷积输出重排为序列格式
            nn.Linear(25*16, d_model)  # 线性层：将卷积特征(25*16维)映射到模型维度d_model
        )

        self.position_enc = PositionalEncoding(  # 初始化位置编码模块
            d_model,  # 编码维度：与模型主维度一致
            n_position=n_position,  # 最大位置数：支持的序列长度上限
            patch_size=patch_size,  # patch尺寸：单个patch的空间大小
            stride=stride,  # 步长：patch提取的步长参数
            n_cols=int(np.sqrt(n_position))  # 网格列数：计算正方形网格的列数
            )
        
        self.dropout = nn.Dropout(p=dropout)  # Dropout层：防止过拟合的正则化技术
        self.layer_stack = nn.ModuleList(  # 构建多层解码器堆栈
            [
                DecoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)  # 创建单个解码器层：包含自注意力、交叉注意力和前馈网络
                for _ in range(n_layers)  # 重复n_layers次，构建深层解码器
            ]
        )
        self.layer_norm  = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化：标准化输入分布，稳定训练
        self.d_model = d_model  # 保存模型维度：用于后续计算和验证

    def forward(self, cur_patch, cur_patch_seq, enc_output):
        """
        解码器前向传播函数
        
        【核心功能】
        基于当前patch和编码器输出，生成下一步的路径预测。通过交叉注意力机制
        融合局部patch信息和全局地图上下文，实现序列化的路径生成。
        
        【处理流程】
        1. Patch特征提取：将当前patch转换为特征表示
        2. 位置编码注入：为当前patch添加序列位置信息
        3. 正则化处理：应用dropout和layer normalization
        4. 多层解码：通过自注意力和交叉注意力生成输出
        
        Args:
            cur_patch (torch.Tensor): 当前处理的地图patch
                形状：通常为(batch, channels, patch_height, patch_width)
                内容：包含当前位置周围的局部地图信息
                用途：提供局部上下文用于路径决策
            cur_patch_seq (int): 当前patch在序列中的位置索引
                范围：0到最大序列长度-1
                作用：用于位置编码，标识当前步骤
                重要性：保证序列生成的时序一致性
            enc_output (torch.Tensor): 编码器的输出特征
                形状：(batch, seq_len, d_model)
                内容：全局地图的高维特征表示
                用途：通过交叉注意力为解码提供全局上下文
        
        Returns:
            tuple: 包含解码输出和注意力权重
                dec_output (torch.Tensor): 解码器输出特征
                    形状：(batch, 1, d_model)
                    内容：当前步骤的预测特征
                    用途：后续分类或回归任务的输入
                dec_enc_attn (torch.Tensor): 解码器-编码器注意力权重
                    形状：(batch, n_heads, 1, seq_len)
                    内容：当前patch对全局地图各位置的注意力分布
                    用途：可视化分析和模型解释
        
        【技术实现】
        1. 特征提取阶段：
           - 卷积网络提取patch的局部特征
           - 维度变换适配解码器输入格式
           - 线性层映射到模型维度
        
        2. 位置编码阶段：
           - 根据cur_patch_seq确定序列位置
           - 注入时序位置信息
           - 保持序列生成的因果性
        
        3. 解码处理阶段：
           - 层归一化稳定训练
           - 多层解码器逐步精化特征
           - 交叉注意力融合全局信息
        
        【注意力机制】
        - 自注意力：建模解码序列内部依赖
        - 交叉注意力：利用编码器的全局信息
        - 掩码机制：防止未来信息泄露
        """
        dec_output = self.to_patch_embedding(cur_patch)  # Patch特征提取：将当前patch通过卷积网络转换为特征表示
        # Add position encoding !!!
        dec_output = self.position_enc(dec_output, cur_patch_seq)  # 位置编码注入：为当前patch添加序列位置信息，cur_patch_seq指定在序列中的位置

        dec_output = self.dropout(dec_output)  # 应用Dropout：随机置零部分特征，防止过拟合

        dec_output = self.layer_norm(dec_output)  # 层归一化：标准化特征分布，为后续解码器层提供稳定输入
        for dec_layer in self.layer_stack:  # 遍历所有解码器层
            dec_output, dec_enc_attn = dec_layer(dec_output, enc_output)  # 通过解码器层：应用自注意力和交叉注意力，返回解码输出和编码器-解码器注意力权重
        return dec_output,  # 返回解码器输出，逗号表示返回单元素元组


class Transformer(nn.Module):
    """
    Transformer - 完整的路径规划Transformer模型
    
    【系统概述】
    集成编码器和分类预测的完整Transformer架构，专门用于端到端的路径规划任务。
    该模型将2D地图作为输入，直接输出每个位置的路径概率分布。
    
    【核心功能】
    1. 地图编码：通过编码器将2D地图转换为高维特征表示
    2. 全局建模：利用自注意力机制建模空间位置间的依赖关系
    3. 路径预测：通过分类头输出每个位置的可通行性概率
    4. 端到端优化：支持从地图到路径的直接监督学习
    
    【技术特点】
    - 仅编码器架构：简化模型结构，专注于空间特征提取
    - 并行处理：所有位置同时预测，提升推理效率
    - 可解释性：注意力权重可视化模型决策过程
    - 多尺度感受野：结合CNN和Transformer的优势
    
    【应用场景】
    1. 静态路径规划：给定地图和起终点，规划最优路径
    2. 可达性分析：分析地图中各位置的可达性
    3. 地图理解：提取地图的语义和拓扑信息
    4. 路径质量评估：评估给定路径的合理性
    
    【模型优势】
    - 全局优化：考虑整个地图的全局信息
    - 并行推理：相比序列生成方法更高效
    - 端到端训练：无需手工特征工程
    - 泛化能力：支持不同尺寸和类型的地图
    
    在MPT框架中的定位：
    - 核心推理引擎：提供基于学习的路径规划能力
    - 基准模型：与传统算法进行性能对比
    - 特征提取器：为其他模块提供地图特征
    """
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape):
        """
        初始化完整的Transformer路径规划模型
        
        【核心功能】
        构建包含编码器和分类预测头的完整模型架构，实现从地图输入到路径概率输出的端到端映射。
        
        【设计理念】
        1. 模块化设计：编码器和分类头分离，便于复用和扩展
        2. 参数共享：编码器参数在所有位置间共享，提升泛化能力
        3. 端到端优化：整个模型可以通过路径监督信号进行联合训练
        4. 灵活配置：支持不同的模型规模和复杂度配置
        
        Args:
            n_layers (int): Transformer编码器层数
                范围：通常4-12层，根据任务复杂度调整
                影响：更多层提供更强的表达能力但增加计算成本
                推荐：路径规划任务通常6-8层效果较好
            n_heads (int): 多头注意力的头数
                约束：d_model必须能被n_heads整除
                作用：不同头关注不同类型的空间关系
                典型值：8, 16等
            d_k (int): Key向量的维度
                计算：通常等于d_model // n_heads
                作用：决定注意力计算的精度
                影响：影响模型的表达能力和计算复杂度
            d_v (int): Value向量的维度
                设置：通常与d_k相等
                用途：决定注意力输出的特征维度
            d_model (int): 模型的主要特征维度
                范围：通常256, 512, 1024等2的幂次
                作用：决定模型的整体表达能力
                注意：必须与分类头的输入维度匹配
            d_inner (int): 前馈网络的隐藏层维度
                设置：通常为d_model的2-4倍
                作用：在注意力层之间提供非线性变换
                示例：d_model=512时，d_inner=2048
            pad_idx (int): 填充标记的索引
                用途：处理变长序列时的填充位置
                当前状态：TODO - 需要完善填充处理逻辑
                未来：可用于处理不规则地图边界
            dropout (float): Dropout概率
                范围：0.0-0.5，通常0.1-0.3
                作用：防止过拟合，提升泛化能力
                调优：根据数据量和模型复杂度调整
            n_position (int): 支持的最大位置数
                计算：max_height × max_width
                限制：决定模型能处理的最大地图尺寸
                示例：10000支持100×100的地图
            train_shape (tuple): 训练时的地图形状
                格式：(height, width)
                用途：优化训练时的内存使用和计算效率
                约束：所有训练样本必须具有相同尺寸
        
        【模型架构】
        1. 编码器(encoder)：
           - 功能：将2D地图转换为高维特征表示
           - 结构：CNN特征提取 + 位置编码 + 多层Transformer
           - 输出：每个位置的d_model维特征向量
        
        2. 分类预测头(classPred)：
           - 功能：将编码特征转换为路径概率
           - 结构：维度重排 + 1×1卷积 + 维度重排
           - 输出：每个位置的2维概率分布(可通行/不可通行)
        
        【技术细节】
        - 硬编码维度：分类头中的512需要与d_model匹配
        - 输出类别：当前设计为2类(可通行/不可通行)
        - 维度变换：通过einops实现高效的张量重排
        
        【扩展性考虑】
        - 多类别支持：可修改输出维度支持更多路径类型
        - 回归任务：可替换分类头支持路径成本预测
        - 多任务学习：可添加多个预测头支持不同任务
        """
        super().__init__()  # 调用父类nn.Module的初始化方法

        self.encoder = Encoder(  # 初始化Transformer编码器：负责将地图转换为高维特征表示
            n_layers=n_layers,  # 编码器层数：控制模型深度和表达能力
            n_heads=n_heads,  # 多头注意力头数：并行处理不同类型的空间关系
            d_k=d_k,  # Key向量维度：决定注意力计算精度
            d_v=d_v,  # Value向量维度：决定注意力输出特征维度
            d_model=d_model,  # 模型主维度：整体特征表示的维度
            d_inner=d_inner,  # 前馈网络隐藏层维度：提供非线性变换能力
            pad_idx=pad_idx,  # 填充索引：处理变长序列的填充标记
            dropout=dropout,  # Dropout概率：防止过拟合的正则化参数
            n_position=n_position,  # 最大位置数：支持的地图尺寸上限
            train_shape=train_shape  # 训练形状：优化训练时的内存使用
        )

        # Last linear layer for prediction
        self.classPred = nn.Sequential(  # 构建分类预测头：将编码特征转换为路径概率
            Rearrange('b c d_model -> (b c) d_model 1 1'),  # 维度重排：将3D特征张量重排为4D格式，适配卷积层输入
            nn.Conv2d(512, 2, kernel_size=1),  # 1x1卷积：将512维特征映射为2类输出(可通行/不可通行)，实现逐位置分类
            Rearrange('bc d 1 1 -> bc d')  # 维度重排：将4D卷积输出重排回2D格式，移除空间维度
        )


    def forward(self, input_map):
        """
        Transformer模型前向传播函数
        
        【核心功能】
        执行完整的路径规划推理过程，从输入地图到输出每个位置的路径概率分布。
        整个过程包括特征编码、全局建模和路径预测三个主要阶段。
        
        【处理流程】
        1. 特征编码：通过编码器将2D地图转换为高维特征表示
        2. 全局建模：利用自注意力机制建模空间位置间的依赖关系
        3. 路径预测：通过分类头输出每个位置的可通行性概率
        4. 格式整理：将输出重新组织为批量格式
        
        Args:
            input_map (torch.Tensor): 输入地图张量
                形状：(batch_size, channels, height, width)
                通道含义：
                  - 通道0：障碍物信息(0=可通行, 1=障碍物)
                  - 通道1：起点终点标记(可选)
                数据范围：通常归一化到[0,1]
                示例：(4, 2, 64, 64)表示4个64×64的双通道地图
        
        Returns:
            torch.Tensor: 路径概率预测结果
                形状：(batch_size, seq_len, num_classes)
                内容：每个位置的类别概率分布
                  - seq_len：等于3×3=9，表示卷积后的空间位置数量
                  - num_classes：通常为2(可通行/不可通行)
                数值范围：经过softmax后的概率值[0,1]
                用途：可用于路径规划、可达性分析等下游任务
        
        【技术实现细节】
        1. 编码阶段：
           - 调用编码器处理输入地图
           - 获得每个位置的高维特征表示
           - 忽略注意力权重(使用*_语法)
        
        2. 预测阶段：
           - 通过分类头将特征映射为类别概率
           - 使用1×1卷积实现高效的逐位置分类
           - 维度变换适配输出格式要求
        
        3. 格式整理：
           - 记录原始批量大小
           - 使用einops重排输出维度
           - 确保输出格式的一致性
        
        【性能特点】
        - 并行处理：所有位置同时预测，无序列依赖
        - 内存效率：通过合理的维度变换减少内存占用
        - 计算效率：利用GPU并行能力加速推理
        
        【使用示例】
        ```python
        model = Transformer(...)
        input_map = torch.randn(4, 2, 64, 64)  # 4个64×64地图
        output = model(input_map)  # 形状：(4, 9, 2)
        probabilities = torch.softmax(output, dim=-1)  # 转换为概率
        ```
        
        【注意事项】
        - 输入地图尺寸必须与训练时一致(或在支持范围内)
        - 输出需要根据具体任务进行后处理(如softmax、argmax等)
        - 批量大小可以灵活调整，但会影响内存使用
        """
        enc_output, *_ = self.encoder(input_map)  # 编码阶段：通过编码器处理输入地图，获得特征表示，*_忽略可能的注意力权重返回值
        seq_logit = self.classPred(enc_output)  # 分类预测：通过分类头将编码特征转换为每个位置的类别logits
        batch_size = input_map.shape[0]  # 获取批量大小：从输入张量的第0维获取batch_size，用于后续维度重排
        return rearrange(seq_logit, '(b c) d -> b c d', b=batch_size)  # 输出重排：将展平的预测结果重新组织为(batch_size, seq_len, num_classes)格式

class SE2Transformer(Transformer):
    # Transformer模型的SE(2)变体，用于处理二维空间路径规划任务
    """
    该模型在原有的Transformer变体的基础上，拓展到SE(2)的任务。
    
    SE(2)表示二维空间中的平移和旋转变换，适用于需要考虑方向信息的路径规划任务。
    因此，在输入上，会增加两个通道（起终点朝向的sin和cos）来表示方向信息。
    因此输入将更改为4个通道：（障碍物信息，起点终点标记，sin，cos）
    
    “Orientation Prediction: For the SE(2) space, 
    each positive anchor point is assigned an orientation which is generated by this layer. 
    Like the classifier, this is also implemented efficiently using a 1 × 1 convolution layer.”
    
    在模型的最后一层分类头中，输出的类别数将从2变为4（可通行/不可通行 + 起点朝向sin/cos）。
    这使得模型能够同时预测路径的可通行性和方向信息。
    
    “For the point robot we sample uniformly across a square grid centered around each anchor point, 
    while for the SE(2) model we sample uniformly across an ellipse 
    whose major axis is oriented along the predicted orientation.”

    模型输出的结果，原本是每个位置的可通行性概率，经过阈值化后，得到可通行性掩膜，
    最后再将每个可通行锚点，映射回到各自的感受野区域，这样就可以获得一个缩小范围的采样区域。

    在SE(2)变体中，我们采样的不是一个正方形，而是一个椭圆，其长轴方向与预测的朝向一致。
    这也就意味着，从每个锚点开始，映射回到原图尺寸上的将是一个椭圆形状，而不是正方形。

    【改动】
    1. 输入通道数从2增加到4，新增sin和cos通道表示方向信息。相应的，卷积层也需要
       调整以适应新的输入维度。
    2. 分类头输出类别数从2增加到4，支持路径可通行性和方向预测。
    """
    
    
class UnevenTransformer(Transformer):
    """
    UnevenTransformer - 用于处理不平坦地面路径规划的Transformer变体
    """
    
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape, output_dim=10):
        """
        初始化不平坦地面路径规划的Transformer模型

        【核心功能】
        构建适用于不平坦地面的Transformer架构，支持变长输入和动态位置编码。
        该模型能够处理不同形状和尺寸的地图输入，适应性强，适用于复杂环境下的路径规划任务。
        
        【设计理念】
        1. 适应性编码：处理不同形状和尺寸的地图输入
        2. 动态位置编码：根据实际输入调整位置编码
        3. 模块化设计：与标准Transformer保持一致，便于复用
        
        Args:
            n_layers (int): Transformer编码器层数
            n_heads (int): 多头注意力的头数
            d_k (int): Key向量的维度
            d_v (int): Value向量的维度
            d_model (int): 模型的主要特征维度
            d_inner (int): 前馈网络的隐藏层维度
            pad_idx (int): 填充标记的索引
            dropout (float): Dropout概率
            n_position (int): 支持的最大位置数
            train_shape (tuple): 训练时的地图形状
            output_dim (int): 分类头输出的类别数, 默认为10, 进行n步的预测 
        """
        super().__init__(n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape)

        # # 重新定义编码器的CNN特征提取部分，以适应不平坦地面的4通道输入
        # self.encoder.to_patch_embedding = nn.Sequential(
        #     nn.Conv2d(6, 6, kernel_size=3),     # 第一层卷积：4输入通道->6输出通道，3x3卷积核，提取基础特征
        #     nn.MaxPool2d(kernel_size=2),        # 最大池化：2x2池化核，降低空间分辨率
        #     nn.ReLU(),                          # ReLU激活函数：引入非线性变换
        #     nn.Conv2d(6, 16, kernel_size=3),    # 第二层卷积：6->16通道，3x3卷积核，进一步提取特征
        #     nn.MaxPool2d(kernel_size=2),        # 第二次最大池化：继续降低分辨率
        #     nn.ReLU(),                          # 第二个ReLU激活
        #     nn.Conv2d(16, d_model,              # 第三层卷积：16->d_model通道，3x3卷积核，进一步提取特征并调整维度
        #               kernel_size=3, 
        #               stride=2, 
        #               padding=1),               
        # )
        
        # self.encoder.to_patch_embedding = nn.Sequential(
        #     # Block 1
        #     nn.Conv2d(3, d_model//8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(d_model//8),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),  # 50×50
            
        #     # Block 2
        #     nn.Conv2d(d_model//8, d_model//4, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(d_model//4),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),  # 25×25
            
        #     # Block 3
        #     nn.Conv2d(d_model//4, d_model//2, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(d_model//2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),  # 12×12
            
        #     # Block 4
        #     nn.Conv2d(d_model//2, d_model, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(d_model),
        #     nn.ReLU(),
        #     nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(d_model),
        #     nn.ReLU(),             
        # )
        
        # 重新定义编码器的CNN特征提取部分
        self.encoder = UnevenEncoder(  # 使用自定义的UnevenEncoder处理不平坦地面的输入
            n_layers=n_layers,  # 编码器层数：控制模型深度和表达能力
            n_heads=n_heads,  # 多头注意力头数：并行处理不同类型的空间关系
            d_k=d_k,  # Key向量维度：决定注意力计算精度
            d_v=d_v,  # Value向量维度：决定注意力输出特征维度
            d_model=d_model,  # 模型主维度：整体特征表示的维度
            d_inner=d_inner,  # 前馈网络隐藏层维度：提供非线性变换能力
            pad_idx=pad_idx,  # 填充索引：处理变长序列的填充标记
            dropout=dropout,  # Dropout概率：防止过拟合的正则化参数
            n_position=n_position,  # 最大位置数：支持的地图尺寸上限
            train_shape=train_shape  # 训练形状：优化训练时的内存使用
        )
        
        # 输出层归一化
        # self.layer_norm = nn.LayerNorm(d_model)
        
        # 更新分类头的输出维度，以适应不平坦地面的预测需求
        self.classPred = nn.Sequential(
            # 输入尺寸：(batch_size, seq_len, d_model)
            Rearrange('b c d_model -> (b c) d_model 1 1'),  # 维度重排：将3D特征张量重排为4D格式，适配卷积层输入
            
            nn.Conv2d(d_model, output_dim, kernel_size=1),  # 1x1卷积：将d_model维特征映射为(output_dim=)n步的预测输出
            
            # nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),  # 第一层：特征提取和降维
            # nn.BatchNorm2d(d_model // 2),
            # nn.ReLU(inplace=True),
            
            # nn.Conv2d(d_model // 2, d_model // 2, kernel_size=3, padding=1),  # 第二层：通道数保持，深层特征提取
            # nn.BatchNorm2d(d_model // 2),
            # nn.ReLU(inplace=True),
            
            # nn.Conv2d(d_model // 2, output_dim, kernel_size=1),  # 第三层：输出层，将特征映射到(output_dim=)n步的预测输出
            
            Rearrange('bc d 1 1 -> bc d'),   # 维度重排：将4D卷积输出重排回2D格式，移除空间维度
            # 这里需要特殊处理来对seq_len维度进行Softmax归一化
            # 输出尺寸：(batch_size * seq_len, output_dim)
        )
        
        # 增加一个预测头，用于预测位置的修正量和角度的生成
        # 输入为编码器输出和分类头输出的拼接结果
        self.correctionPred = nn.Sequential(
            # 输入尺寸：(batch_size, seq_len, d_model + output_dim)
            Rearrange('b c d_model -> (b c) d_model 1 1'),  # 维度重排：将3D特征张量重排为4D格式，适配卷积层输入
            
            nn.Conv2d(d_model + output_dim, 4*output_dim, kernel_size=1),  # 1x1卷积：将(d_model+output_dim)维特征映射为3*(output_dim=)n步的预测输出
            
            # # 第一层：特征提取和降维
            # nn.Conv2d(d_model + output_dim, (d_model + output_dim) // 2, kernel_size=3, padding=1),
            # nn.BatchNorm2d((d_model + output_dim) // 2),
            # nn.ReLU(inplace=True),
            
            # # 第二层：通道数保持，深层特征提取
            # nn.Conv2d((d_model + output_dim) // 2, (d_model + output_dim) // 2, kernel_size=3, padding=1),
            # nn.BatchNorm2d((d_model + output_dim) // 2),
            # nn.ReLU(inplace=True),
            
            # # 第三层：输出层
            # nn.Conv2d((d_model + output_dim) // 2, 3 * output_dim, kernel_size=1),  # 1x1卷积作为最终输出
            
            Rearrange('bc d 1 1 -> bc d'),  # 维度重排：将4D卷积输出重排为(batch_size, seq_len, 3*output_dim)格式
        )
        
    # def forward(self, input_map):
    #     # 模型前向传播函数，需要输出分类结果和修正结果
    #     enc_output, *_ = self.encoder(input_map)  # 编码阶段：通过编码器处理输入地图，获得特征表示，*_忽略可能的注意力权重返回值
        
    #     # enc_output = self.layer_norm(enc_output)  # 输出层归一化：对编码器输出进行层归一化，稳定训练过程
        
    #     seq_logit = self.classPred(enc_output)  # 分类预测：通过分类头将编码特征转换为每个位置的类别logits（未归一化）
    #     batch_size = input_map.shape[0]  # 获取批量大小：从输入张量的第0维获取batch_size，用于后续维度重排
    #     seq_logit_reshaped = rearrange(seq_logit, '(b c) d -> b c d', b=batch_size)  # 输出重排：将展平的预测结果重新组织为(batch_size, seq_len, num_classes)格式
        
    #     # 对seq_len维度进行Softmax归一化
    #     seq_logit_softmax = F.softmax(seq_logit_reshaped, dim=1)  # 在seq_len维度(dim=1)上进行Softmax归一化
    #     # return seq_logit_softmax  # 返回分类预测结果
        
    #     # 拼接编码器输出和分类预测结果
    #     combined_features = torch.cat([enc_output, seq_logit_softmax], dim=-1)  # 在最后一个维度上拼接特征：(batch, seq_len, d_model + output_dim)
    #     correction = self.correctionPred(combined_features)  # 结合地图特征和概率引导特征，通过修正头获得位置修正量和角度预测
        
    #     correction_sigmoid = F.sigmoid(correction)  # 对修正预测结果进行Sigmoid归一化，确保输出在[0, 1]范围内
        
    #     # 重排修正预测结果：将(batch, seq_len, 3*output_dim) -> (batch, seq_len, 3, output_dim)
    #     correction_reshaped = rearrange(correction_sigmoid, '(b c) (n d) -> b c n d', b=batch_size, n=3)
    #     return seq_logit_softmax, correction_reshaped  # 返回分类预测结果和位置修正预测结果
    
    # def forward(self, map_input, pose_input):
    def forward(self, map_input):
        # 模型前向传播函数，需要输出分类结果和修正结果
        # map_tokens, last_pose_ctx, yaw_logits_agg, *_ = self.encoder(map_input, pose_input)  # 编码阶段：通过编码器处理输入地图和位姿信息，获得特征表示，*_忽略可能的注意力权重返回值
        map_tokens = self.encoder(map_input)  # 编码阶段：通过编码器处理输入地图和位姿信息，获得特征表示，*_忽略可能的注意力权重返回值

        # enc_output = self.layer_norm(enc_output)  # 输出层归一化：对编码器输出进行层归一化，稳定训练过程

        seq_logit = self.classPred(map_tokens)  # 分类预测：通过分类头将编码特征转换为每个位置的类别logits（未归一化）
        batch_size = map_input.shape[0]  # 获取批量大小：从输入张量的第0维获取batch_size，用于后续维度重排
        seq_logit_reshaped = rearrange(seq_logit, '(b c) d -> b c d', b=batch_size)  # 输出重排：将展平的预测结果重新组织为(batch_size, seq_len, num_classes)格式
        
        # 对seq_len维度进行Softmax归一化
        seq_logit_softmax = F.softmax(seq_logit_reshaped, dim=1)  # 在seq_len维度(dim=1)上进行Softmax归一化
        # return seq_logit_softmax  # 返回分类预测结果
        
        # 拼接编码器输出和分类预测结果
        combined_features = torch.cat([map_tokens, seq_logit_softmax], dim=-1)  # 在最后一个维度上拼接特征：(batch, seq_len, d_model + output_dim)
        correction = self.correctionPred(combined_features)  # 结合地图特征和概率引导特征，通过修正头获得位置修正量和角度预测
        
        correction_sigmoid = F.sigmoid(correction)  # 对修正预测结果进行Sigmoid归一化，确保输出在[0, 1]范围内
        
        # 重排修正预测结果：将(batch, seq_len, 3*output_dim) -> (batch, seq_len, 3, output_dim)
        correction_reshaped = rearrange(correction_sigmoid, '(b c) (n d) -> b c n d', b=batch_size, n=4)
        # return seq_logit_softmax, correction_reshaped, yaw_logits_agg  # 返回分类预测结果和位置修正预测结果
        return seq_logit_softmax, correction_reshaped  # 返回分类预测结果和位置修正预测结果