"""
Models.py - Mamba模型的核心模块实现
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from functools import partial
from vision_mamba.Layers import EncoderLayer, DecoderLayer, PoseWiseEncoderLayer

from einops.layers.torch import Rearrange
from einops import rearrange

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position, train_shape):
        super(PositionalEncoding, self).__init__()  # 调用父类nn.Module的初始化方法
        self.n_pos_sqrt = int(np.sqrt(n_position))  # 计算网格边长：将总位置数开方得到正方形网格的边长
        self.train_shape = train_shape  # 保存训练时的地图形状，用于后续的位置编码选择
        # Not a parameter
        self.register_buffer('hashIndex', self._get_hash_table(n_position))  # 注册哈希索引表为缓冲区：不参与梯度更新但会随模型移动设备
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))  # 注册完整的正弦位置编码表
        self.register_buffer('pos_table_train', self._get_sinusoid_encoding_table_train(n_position, train_shape))  # 注册训练专用的位置编码表

    def _get_hash_table(self, n_position):
        return rearrange(torch.arange(n_position), '(h w) -> h w', h=int(np.sqrt(n_position)), w=int(np.sqrt(n_position)))  # 使用einops将1D索引序列重排为2D网格：创建从0到n_position-1的连续索引，按行优先顺序排列成正方形网格

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """
        生成正弦位置编码表
        - TODO优化：当前使用numpy实现，可改为纯torch提升GPU利用率
        """

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
        # 【步骤1】选择训练地图对应的坐标区域
        # 从完整的哈希索引表中切片出训练尺寸的矩形区域
        selectIndex = rearrange(
            self.hashIndex[:train_shape[0], :train_shape[1]],  # 从哈希表中切片选择训练尺寸的矩形区域：前train_shape[0]行，前train_shape[1]列
            'h w -> (h w)'  # 将2D坐标索引展平为1D序列索引：使用einops将2D索引矩阵重排为1D向量
        )
        
        # 【步骤2】根据索引提取对应的位置编码
        # dim=1表示沿位置维度进行索引选择
        # 结果形状：(1, train_height×train_width, d_hid)
        return torch.index_select(self.pos_table, dim=1, index=selectIndex)  # 根据选择的索引从完整位置编码表中提取对应的编码向量

    def forward(self, x, conv_shape=None):
        # Ensure all internal buffers on same device as input to avoid implicit transfers/syncs
        device = x.device
        if getattr(self, "pos_table", None) is not None and self.pos_table.device != device:
            # move once (will be cheap after first call); avoid moving every batch unnecessarily
            self.pos_table = self.pos_table.to(device)
        if getattr(self, "pos_table_train", None) is not None and self.pos_table_train.device != device:
            self.pos_table_train = self.pos_table_train.to(device)
        if getattr(self, "hashIndex", None) is not None and self.hashIndex.device != device:
            self.hashIndex = self.hashIndex.to(device)

        if conv_shape is None:
            # training mode: sample a random top-left offset
            # generate small Python ints to slice hashIndex (cheap)
            max_start = self.n_pos_sqrt - self.train_shape[0]
            if max_start <= 0:
                startH = 0
                startW = 0
            else:
                # use CPU RNG to get python ints (no heavy GPU sync)
                r = torch.randint(0, max_start, (2,))
                startH = int(r[0].item())
                startW = int(r[1].item())

            # slice (hashIndex already on correct device) and flatten indices
            selectIndex = rearrange(
                self.hashIndex[startH:startH + self.train_shape[0], startW:startW + self.train_shape[1]],
                'h w -> (h w)'
            )
            # make sure index is long and on same device
            if selectIndex.dtype != torch.long:
                selectIndex = selectIndex.long()
            if selectIndex.device != device:
                selectIndex = selectIndex.to(device)

            return x + torch.index_select(self.pos_table, dim=1, index=selectIndex).clone().detach()

        selectIndex = rearrange(self.hashIndex[:conv_shape[0], :conv_shape[1]], 'h w -> (h w)')
        if selectIndex.dtype != torch.long:
            selectIndex = selectIndex.long()
        if selectIndex.device != device:
            selectIndex = selectIndex.to(device)
        return x + torch.index_select(self.pos_table, dim=1, index=selectIndex)


class Encoder(nn.Module):
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape):
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

class Transformer(nn.Module):
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape):
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
        enc_output, *_ = self.encoder(input_map)  # 编码阶段：通过编码器处理输入地图，获得特征表示，*_忽略可能的注意力权重返回值
        seq_logit = self.classPred(enc_output)  # 分类预测：通过分类头将编码特征转换为每个位置的类别logits
        batch_size = input_map.shape[0]  # 获取批量大小：从输入张量的第0维获取batch_size，用于后续维度重排
        return rearrange(seq_logit, '(b c) d -> b c d', b=batch_size)  # 输出重排：将展平的预测结果重新组织为(batch_size, seq_len, num_classes)格式

    
class UnevenTransformer(nn.Module):
    """
    UnevenMamba - 用于处理不平坦地面路径规划的Mamba变体
    """
    
    def __init__(self, n_layers, d_state, dt_rank, d_model, pad_idx, dropout, drop_path, n_position, train_shape, output_dim=10):
        """
        初始化不平坦地面路径规划的Mamba模型

        【核心功能】
        构建适用于不平坦地面的Mamba架构，支持变长输入和动态位置编码。
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
        super().__init__()
        # super().__init__(n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape)

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
        
        # # 重新定义编码器的CNN特征提取部分
        # self.encoder = UnevenEncoder(  # 使用自定义的UnevenEncoder处理不平坦地面的输入
        #     n_layers=n_layers,  # 编码器层数：控制模型深度和表达能力
        #     n_heads=n_heads,  # 多头注意力头数：并行处理不同类型的空间关系
        #     d_k=d_k,  # Key向量维度：决定注意力计算精度
        #     d_v=d_v,  # Value向量维度：决定注意力输出特征维度
        #     d_model=d_model,  # 模型主维度：整体特征表示的维度
        #     d_inner=d_inner,  # 前馈网络隐藏层维度：提供非线性变换能力
        #     pad_idx=pad_idx,  # 填充索引：处理变长序列的填充标记
        #     dropout=dropout,  # Dropout概率：防止过拟合的正则化参数
        #     n_position=n_position,  # 最大位置数：支持的地图尺寸上限
        #     train_shape=train_shape  # 训练形状：优化训练时的内存使用
        # )
        
        self.encoder = VimEncoder(  # 使用自定义的VimEncoder处理不平坦地面的输入
            n_layers=n_layers,  # 编码器层数：控制模型深度和表达能力
            d_state=d_state,
            dt_rank=dt_rank,
            d_model=d_model,  # 模型主维度：整体特征表示的维度
            pad_idx=pad_idx,  # 填充索引：处理变长序列的填充标记
            dropout=dropout,  # Dropout概率：防止过拟合的正则化参数
            drop_path=drop_path,
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
            
            nn.Conv2d(d_model + output_dim, 3*output_dim, kernel_size=1),  # 1x1卷积：将(d_model+output_dim)维特征映射为3*(output_dim=)n步的预测输出
            
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
        correction_reshaped = rearrange(correction_sigmoid, '(b c) (n d) -> b c n d', b=batch_size, n=3)
        # return seq_logit_softmax, correction_reshaped, yaw_logits_agg  # 返回分类预测结果和位置修正预测结果
        return seq_logit_softmax, correction_reshaped  # 返回分类预测结果和位置修正预测结果

from vision_mamba.vmamba import (
    SS2D,           # VMamba 的核心 SS2D 模块（包含四向扫描）
    LayerNorm,      # VMamba 的 LayerNorm
    DropPath,       # DropPath 用于正则化
    VSSBlock,       # VMamba 的 VSSBlock 模块
)
class VimEncoder(nn.Module):
    """    
    基于 VMamba SS2D 的编码器（使用官方四向扫描实现）
    """
    def __init__(self, n_layers, d_state, dt_rank, d_model, pad_idx, dropout, drop_path, n_position, train_shape):
        super().__init__()
        
        # CNN 特征提取（保持不变）
        self.map_fe = nn.Sequential(
            nn.Conv2d(6, d_model//8, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model//8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 50×50
            
            nn.Conv2d(d_model//8, d_model//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model//4),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25×25
            
            nn.Conv2d(d_model//4, d_model//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model//2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12×12
            
            nn.Conv2d(d_model//2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
        )

        # 不需要 reorder_dims，SS2D 接受 BCHW 格式
        
        # 位置编码需要适配 channel_first 格式
        self.position_enc = PositionalEncoding(d_model, n_position=n_position, train_shape=train_shape)
        self.dropout = nn.Dropout(p=dropout)
        self.input_ln = LayerNorm(d_model, channel_first=True)  # 使用 VMamba 的 LayerNorm
        
        # DropPath rates
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_layers)]
        
        # 使用 VMamba 的 SS2D 模块（自带四向扫描）
        self.layers = nn.ModuleList([
            # SS2D(
            #     d_model=d_model,
            #     d_state=d_state,
            #     ssm_ratio=2.0,  # 可调整
            #     dt_rank=dt_rank if dt_rank != "auto" else "auto",
            #     act_layer=nn.SiLU,
            #     # dwconv 参数
            #     d_conv=3,
            #     conv_bias=True,
            #     # dropout
            #     dropout=dropout,
            #     bias=False,
            #     # dt init 参数
            #     dt_min=0.001,
            #     dt_max=0.1,
            #     dt_init="random",
            #     dt_scale=1.0,
            #     dt_init_floor=1e-4,
            #     initialize="v0",
            #     # forward_type：使用 v05 或 v2（支持四向扫描）
            #     forward_type="v05_noz",  # 推荐使用 v05_noz（无 z 分支，更轻量）
            #     channel_first=True,  # 使用 channel_first 模式
            # )

            VSSBlock(
                hidden_dim=d_model,
                drop_path=dpr[i],
                channel_first=True,
                # SSM 参数
                ssm_d_state=d_state,
                ssm_ratio=2.0,
                ssm_dt_rank=dt_rank,
                # MLP 参数
                mlp_ratio=4.0,  # 标准 Transformer 比例
                mlp_act_layer=nn.GELU,
                mlp_drop_rate=dropout,
                # 归一化
                norm_layer=partial(LayerNorm, channel_first=True),
                # 扫描类型
                forward_type="v05_noz",  # 或 "v2" "v05"
            )
            for i in range(n_layers)
        ])
        
        # 输出归一化
        self.norm_f = LayerNorm(d_model, channel_first=True)
        
    def forward(self, input_map, returns_attns=False):
        # CNN 特征提取
        map_feat = self.map_fe(input_map)  # [B, D, Hf, Wf]
        conv_map_shape = map_feat.shape[-2:]
        H, W = conv_map_shape
        
        # 位置编码（需要转换格式）
        # PositionalEncoding 期望 [B, N, D]，需要先转换
        map_tokens_pos = rearrange(map_feat, 'b c h w -> b (h w) c')
        map_tokens_pos = self.position_enc(
            map_tokens_pos, 
            conv_shape=conv_map_shape if not self.training else None
        )
        map_tokens_pos = self.dropout(map_tokens_pos)
        # 转换回 BCHW
        map_feat = rearrange(map_tokens_pos, 'b (h w) c -> b c h w', h=H, w=W)
        
        # 输入归一化
        map_feat = self.input_ln(map_feat)
        
        # 逐层处理（SS2D 接受 BCHW 格式）
        for layer in self.layers:
            map_feat = layer(map_feat)  # [B, D, H, W]
        
        # 最终归一化
        map_feat = self.norm_f(map_feat)
        
        # 转换回 [B, N, D] 格式用于后续处理
        map_tokens = rearrange(map_feat, 'b c h w -> b (h w) c')
        
        return map_tokens

# ----------------------------------------------- #
from vim.models_mamba import create_block
from timm.models.layers import DropPath

class VisionMambaLayer(nn.Module):
    """
    单个论文编码单元：x -> (forward conv1d -> forward SSM) & (backward conv1d -> backward SSM) + z path
    返回 (out, residual) 以兼容现有 Block 接口。
    """
    def __init__(self, d_model, d_state, layer_idx, drop_path, ssm_cfg, factory_kwargs, rms_norm, residual_in_fp32, fused_add_norm, if_bimamba, bimamba_type, if_divide_out, init_layer_scale):
        super().__init__()
        # pointwise projections (可视作 Conv1d kernel_size=1)
        self.x_proj_f = nn.Linear(d_model, d_model, bias=False)
        self.x_proj_b = nn.Linear(d_model, d_model, bias=False)
        self.z_proj   = nn.Linear(d_model, 1)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # 映射 fusion -> y（论文图中紫色投影）
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # 两个 SSM block（复用 create_block）
        self.forward_block = create_block(
            d_model=d_model,
            d_state=d_state,
            ssm_cfg=ssm_cfg,
            norm_epsilon=1e-5,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=layer_idx,
            if_bimamba=if_bimamba,
            bimamba_type=bimamba_type,
            drop_path=drop_path,
            if_divide_out=if_divide_out,
            init_layer_scale=init_layer_scale,
            **factory_kwargs,
        )
        self.backward_block = create_block(
            d_model=d_model,
            d_state=d_state,
            ssm_cfg=ssm_cfg,
            norm_epsilon=1e-5,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=layer_idx,
            if_bimamba=if_bimamba,
            bimamba_type=bimamba_type,
            drop_path=drop_path,
            if_divide_out=if_divide_out,
            init_layer_scale=init_layer_scale,
            **factory_kwargs,
        )

    def forward(self, x, residual=None):
        # x: [B, N, D]
        x_norm = self.norm(x)
        xf = self.x_proj_f(x_norm)         # forward path
        xb = self.x_proj_b(x_norm)         # backward path
        z_act = self.act(self.z_proj(x_norm))   # [B, N, 1]
        gate = torch.sigmoid(z_act)              # [B, N, 1]

        # forward SSM
        out_f, res_f = self.forward_block(xf, residual)

        # backward: run SSM on reversed seq and flip back
        xb_rev = xb.flip(1)
        res_rev = None if residual is None else residual.flip(1)
        out_b_rev, res_b_rev = self.backward_block(xb_rev, res_rev)
        out_b = out_b_rev.flip(1)

        # 用 gate 对 forward/backward 输出做加权混合（z 作为门控）
        fusion = gate * out_f + (1.0 - gate) * out_b   # fusion: [B,N,D]
        
        # --- 论文关键点：fusion -> 映射 -> 与输入残差相加（y 映射 + skip） ---
        y = self.out_proj(fusion)       # 映射回到特征维度（紫色块）
        out = x + y                      # 残差连接：out = x + y

        # # 合并残差（按 gate 加权，若某侧无 residual 则保留另一侧）
        # if res_f is None and res_b_rev is None:
        #     res = None
        # elif res_f is None:
        #     res = (1.0 - gate) * res_b_rev.flip(1)
        # elif res_b_rev is None:
        #     res = gate * res_f
        # else:
        #     res = gate * res_f + (1.0 - gate) * res_b_rev.flip(1)

        # return out, res
        return out

class MultiDirectionalMambaLayer(nn.Module):
    """
    多方向扫描的 Mamba 层
    
    【核心功能】
    实现 4 种扫描方式：
    1. 横向从左到右扫描（行优先，正向）
    2. 横向从右到左扫描（行优先，反向）
    3. 纵向从上到下扫描（列优先，正向）
    4. 纵向从下到上扫描（列优先，反向）
    
    【设计思路】
    - 通过不同的维度重排实现行扫描和列扫描
    - 使用序列翻转实现正向和反向扫描
    - 使用门控机制融合 4 个方向的特征
    """
    def __init__(self, d_model, d_state, layer_idx, drop_path, ssm_cfg, factory_kwargs, 
                 rms_norm, residual_in_fp32, fused_add_norm, if_bimamba, bimamba_type, 
                 if_divide_out, init_layer_scale):
        super().__init__()
        
        # 4 个方向的投影层
        self.x_proj_h_forward = nn.Linear(d_model, d_model, bias=False)   # 横向正向
        self.x_proj_h_backward = nn.Linear(d_model, d_model, bias=False)  # 横向反向
        self.x_proj_v_forward = nn.Linear(d_model, d_model, bias=False)   # 纵向正向
        self.x_proj_v_backward = nn.Linear(d_model, d_model, bias=False)  # 纵向反向
        
        # 门控投影（4 路加权融合）
        self.gate_proj = nn.Linear(d_model, 4)  # 输出 4 个权重
        
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # 最终输出投影
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        
        # 4 个 SSM 块（每个方向一个）
        self.h_forward_block = create_block(
            d_model=d_model, d_state=d_state, ssm_cfg=ssm_cfg,
            norm_epsilon=1e-5, rms_norm=rms_norm, residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm, layer_idx=layer_idx,
            if_bimamba=if_bimamba, bimamba_type=bimamba_type,
            drop_path=drop_path, if_divide_out=if_divide_out,
            init_layer_scale=init_layer_scale, **factory_kwargs,
        )
        
        self.h_backward_block = create_block(
            d_model=d_model, d_state=d_state, ssm_cfg=ssm_cfg,
            norm_epsilon=1e-5, rms_norm=rms_norm, residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm, layer_idx=layer_idx,
            if_bimamba=if_bimamba, bimamba_type=bimamba_type,
            drop_path=drop_path, if_divide_out=if_divide_out,
            init_layer_scale=init_layer_scale, **factory_kwargs,
        )
        
        self.v_forward_block = create_block(
            d_model=d_model, d_state=d_state, ssm_cfg=ssm_cfg,
            norm_epsilon=1e-5, rms_norm=rms_norm, residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm, layer_idx=layer_idx,
            if_bimamba=if_bimamba, bimamba_type=bimamba_type,
            drop_path=drop_path, if_divide_out=if_divide_out,
            init_layer_scale=init_layer_scale, **factory_kwargs,
        )
        
        self.v_backward_block = create_block(
            d_model=d_model, d_state=d_state, ssm_cfg=ssm_cfg,
            norm_epsilon=1e-5, rms_norm=rms_norm, residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm, layer_idx=layer_idx,
            if_bimamba=if_bimamba, bimamba_type=bimamba_type,
            drop_path=drop_path, if_divide_out=if_divide_out,
            init_layer_scale=init_layer_scale, **factory_kwargs,
        )
    
    def forward(self, x, residual=None, spatial_shape=None):
        """
        Args:
            x: [B, N, D] 输入特征，N = H * W
            residual: 残差（可选）
            spatial_shape: (H, W) 空间形状，用于重排
        
        Returns:
            out: [B, N, D] 输出特征
        """
        B, N, D = x.shape
        
        # 推断空间形状（如果未提供）
        if spatial_shape is None:
            H = W = int(N ** 0.5)
            assert H * W == N, f"N={N} 必须是完全平方数"
        else:
            H, W = spatial_shape
            assert H * W == N, f"H*W={H*W} 必须等于 N={N}"
        
        # 归一化
        x_norm = self.norm(x)
        
        # ===== 1. 横向正向扫描（行优先，从左到右）=====
        x_h_forward = self.x_proj_h_forward(x_norm)  # [B, N, D]
        out_h_f, _ = self.h_forward_block(x_h_forward, residual)  # [B, N, D]
        
        # ===== 2. 横向反向扫描（行优先，从右到左）=====
        x_h_backward = self.x_proj_h_backward(x_norm)
        # 重排为 [B, H, W, D]，然后在 W 维度翻转
        x_h_back_spatial = rearrange(x_h_backward, 'b (h w) d -> b h w d', h=H, w=W)
        x_h_back_flipped = x_h_back_spatial.flip(dims=[2])  # 翻转宽度维度
        x_h_back_seq = rearrange(x_h_back_flipped, 'b h w d -> b (h w) d')
        
        out_h_b, _ = self.h_backward_block(x_h_back_seq, residual)
        
        # 翻转回来
        out_h_b_spatial = rearrange(out_h_b, 'b (h w) d -> b h w d', h=H, w=W)
        out_h_b = rearrange(out_h_b_spatial.flip(dims=[2]), 'b h w d -> b (h w) d')
        
        # ===== 3. 纵向正向扫描（列优先，从上到下）=====
        x_v_forward = self.x_proj_v_forward(x_norm)
        # 转置：[B, H, W, D] -> [B, W, H, D]，即按列扫描
        x_v_forward_spatial = rearrange(x_v_forward, 'b (h w) d -> b h w d', h=H, w=W)
        x_v_forward_transposed = x_v_forward_spatial.permute(0, 2, 1, 3)  # [B, W, H, D]
        x_v_forward_seq = rearrange(x_v_forward_transposed, 'b w h d -> b (w h) d')
        
        out_v_f, _ = self.v_forward_block(x_v_forward_seq, residual)
        
        # 转置回来
        out_v_f_transposed = rearrange(out_v_f, 'b (w h) d -> b w h d', w=W, h=H)
        out_v_f = rearrange(out_v_f_transposed.permute(0, 2, 1, 3), 'b h w d -> b (h w) d')
        
        # ===== 4. 纵向反向扫描（列优先，从下到上）=====
        x_v_backward = self.x_proj_v_backward(x_norm)
        x_v_back_spatial = rearrange(x_v_backward, 'b (h w) d -> b h w d', h=H, w=W)
        # 先转置为列优先，再在列方向翻转
        x_v_back_transposed = x_v_back_spatial.permute(0, 2, 1, 3)  # [B, W, H, D]
        x_v_back_flipped = x_v_back_transposed.flip(dims=[2])  # 翻转高度维度
        x_v_back_seq = rearrange(x_v_back_flipped, 'b w h d -> b (w h) d')
        
        out_v_b, _ = self.v_backward_block(x_v_back_seq, residual)
        
        # 翻转并转置回来
        out_v_b_transposed = rearrange(out_v_b, 'b (w h) d -> b w h d', w=W, h=H)
        out_v_b_flipped_back = out_v_b_transposed.flip(dims=[2])
        out_v_b = rearrange(out_v_b_flipped_back.permute(0, 2, 1, 3), 'b h w d -> b (h w) d')
        
        # ===== 5. 门控融合 4 个方向 =====
        gate_logits = self.gate_proj(x_norm)  # [B, N, 4]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, N, 4]
        
        # 加权求和
        fusion = (
            gate_weights[..., 0:1] * out_h_f +
            gate_weights[..., 1:2] * out_h_b +
            gate_weights[..., 2:3] * out_v_f +
            gate_weights[..., 3:4] * out_v_b
        )  # [B, N, D]
        
        # 最终投影 + 残差
        y = self.out_proj(fusion)
        out = x + y
        
        return out


# # use the MultiDirectionalMambaLayer
# class VimEncoder(nn.Module):
#     """    
#     基于多方向扫描的 Vim 编码器
#     """
#     def __init__(self, n_layers, d_state, dt_rank, d_model, pad_idx, dropout, drop_path, n_position, train_shape):
#         super().__init__()
        
#         # ...existing code...（CNN 特征提取部分保持不变）
#         self.map_fe = nn.Sequential(
#             nn.Conv2d(6, d_model//8, kernel_size=3, padding=1),
#             nn.BatchNorm2d(d_model//8),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(d_model//8, d_model//4, kernel_size=3, padding=1),
#             nn.BatchNorm2d(d_model//4),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(d_model//4, d_model//2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(d_model//2),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(d_model//2, d_model, kernel_size=3, padding=1),
#             nn.BatchNorm2d(d_model),
#             nn.ReLU(),
#             nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
#             nn.BatchNorm2d(d_model),
#             nn.ReLU(),
#         )

#         self.reorder_dims = Rearrange('b c h w -> b (h w) c')
#         self.position_enc = PositionalEncoding(d_model, n_position=n_position, train_shape=train_shape)
#         self.dropout = nn.Dropout(p=dropout)
#         self.input_ln = nn.LayerNorm(d_model, eps=1e-6)
        
#         # ViM 参数配置
#         ssm_cfg = None
#         dpr = [x.item() for x in torch.linspace(0, drop_path, n_layers)]
#         inter_dpr = [0.0] + dpr

#         norm_epsilon = 1e-5
#         rms_norm = True
#         residual_in_fp32 = True
#         fused_add_norm = True
#         if_bimamba = True
#         bimamba_type = "v2"
#         if_divide_out = True
#         init_layer_scale = None
#         factory_kwargs = {"device": None, "dtype": None}
        
#         # 使用新的多方向扫描层
#         self.layers = nn.ModuleList(
#             [
#                 MultiDirectionalMambaLayer(
#                     d_model=d_model,
#                     d_state=d_state,
#                     layer_idx=i,
#                     drop_path=inter_dpr[i],
#                     ssm_cfg=ssm_cfg,
#                     factory_kwargs=factory_kwargs,
#                     rms_norm=rms_norm,
#                     residual_in_fp32=residual_in_fp32,
#                     fused_add_norm=fused_add_norm,
#                     if_bimamba=if_bimamba,
#                     bimamba_type=bimamba_type,
#                     if_divide_out=if_divide_out,
#                     init_layer_scale=init_layer_scale,
#                 )
#                 for i in range(n_layers)
#             ]
#         )
        
#         self.norm_f = nn.LayerNorm(d_model, eps=1e-5)
        
#     def forward(self, input_map, returns_attns=False):
#         # CNN特征提取
#         map_feat = self.map_fe(input_map)  # [B, D, Hf, Wf]
#         conv_map_shape = map_feat.shape[-2:]
#         H, W = conv_map_shape
#         map_tokens = self.reorder_dims(map_feat)  # [B, N_map, D]

#         # 位置编码
#         map_tokens = self.position_enc(map_tokens, conv_shape=conv_map_shape if not self.training else None)
#         map_tokens = self.dropout(map_tokens)
#         map_tokens = self.input_ln(map_tokens)
        
#         # 逐层调用多方向扫描
#         for layer in self.layers:
#             map_tokens = layer(map_tokens, residual=None, spatial_shape=(H, W))
        
#         # 最终归一化
#         map_tokens = self.norm_f(map_tokens.to(dtype=self.norm_f.weight.dtype))
            
#         return map_tokens

# class VimEncoder(nn.Module):
#     """    
#     基于 ViM / VisionMamba 思路的编码器：使用 model.py 中的 VisionEncoderMambaBlock（zeta.nn.SSM）
#     """
#     def __init__(self, n_layers, d_state, dt_rank, d_model, pad_idx, dropout, drop_path, n_position, train_shape):
#         super().__init__()
        
#         # CNN特征提取 (保持不变)
#         self.map_fe = nn.Sequential(
#             nn.Conv2d(6, d_model//8, kernel_size=3, padding=1),
#             nn.BatchNorm2d(d_model//8),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(d_model//8, d_model//4, kernel_size=3, padding=1),
#             nn.BatchNorm2d(d_model//4),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(d_model//4, d_model//2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(d_model//2),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(d_model//2, d_model, kernel_size=3, padding=1),
#             nn.BatchNorm2d(d_model),
#             nn.ReLU(),
#             nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
#             nn.BatchNorm2d(d_model),
#             nn.ReLU(),
#         )
        
#         # c1, c2, c3 = d_model//8, d_model//4, d_model//2
#         # self.map_fe = nn.Sequential(
#         #     # 100->50
#         #     nn.Conv2d(6, c1, kernel_size=3, padding=1, stride=1),
#         #     nn.BatchNorm2d(c1), nn.ReLU(),
#         #     nn.MaxPool2d(2),

#         #     # 50->25
#         #     nn.Conv2d(c1, c1, 3, padding=1, stride=1),
#         #     nn.BatchNorm2d(c1), nn.ReLU(),
#         #     nn.Conv2d(c1, c2, 3, padding=1, stride=2),
#         #     nn.BatchNorm2d(c2), nn.ReLU(),

#         #     # 25->12
#         #     nn.Conv2d(c2, c2, 3, padding=1, stride=1),
#         #     nn.BatchNorm2d(c2), nn.ReLU(),
#         #     nn.Conv2d(c2, c2, 3, padding=1, stride=1),
#         #     nn.BatchNorm2d(c2), nn.ReLU(),
#         #     nn.MaxPool2d(2),
            
#         #     nn.Conv2d(c2, d_model, 1),  # project to d_model
#         #     nn.BatchNorm2d(d_model), nn.ReLU()
#         # )

#         self.reorder_dims = Rearrange('b c h w -> b (h w) c')

#         # 位置编码 (保持不变)
#         self.position_enc = PositionalEncoding(d_model, n_position=n_position, train_shape=train_shape)
#         self.dropout = nn.Dropout(p=dropout)
        
#         self.input_ln = nn.LayerNorm(d_model, eps=1e-6)
        
#         # ViM 参数配置（使用 VisionEncoderMambaBlock）
#         ssm_cfg = None
#         dpr = [x.item() for x in torch.linspace(0, drop_path, n_layers)]  # drop path rate
#         inter_dpr = [0.0] + dpr
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         # 设置必要的参数
#         norm_epsilon = 1e-5
#         rms_norm = True
#         residual_in_fp32 = True
#         fused_add_norm = True
#         if_bimamba = True
#         bimamba_type = "v2"
#         if_divide_out = True
#         init_layer_scale = None
#         factory_kwargs = {"device": None, "dtype": None}
        
#         # # 使用 VisionEncoderMambaBlock 作为每一层的核心块
#         # self.layers = nn.ModuleList([
#         #     create_block(
#         #         d_model=d_model,
#         #         d_state=d_state,
#         #         ssm_cfg=ssm_cfg,
#         #         norm_epsilon=norm_epsilon,
#         #         rms_norm=rms_norm,
#         #         residual_in_fp32=residual_in_fp32,
#         #         fused_add_norm=fused_add_norm,
#         #         layer_idx=i,
#         #         if_bimamba=if_bimamba,
#         #         bimamba_type=bimamba_type,
#         #         drop_path=inter_dpr[i],
#         #         if_divide_out=if_divide_out,
#         #         init_layer_scale=init_layer_scale,
#         #         **factory_kwargs,
#         #     )
#         #     for i in range(n_layers)
#         # ])
        
#         # 每层使用论文编码单元封装（forward/backward SSM + activation path）
#         self.layers = nn.ModuleList(
#             [
#                 VisionMambaLayer(
#                     d_model=d_model,
#                     d_state=d_state,
#                     layer_idx=i,
#                     drop_path=inter_dpr[i],
#                     ssm_cfg=ssm_cfg,
#                     factory_kwargs=factory_kwargs,
#                     rms_norm=rms_norm,
#                     residual_in_fp32=residual_in_fp32,
#                     fused_add_norm=fused_add_norm,
#                     if_bimamba=if_bimamba,
#                     bimamba_type=bimamba_type,
#                     if_divide_out=if_divide_out,
#                     init_layer_scale=init_layer_scale,
#                 )
#                 for i in range(n_layers)
#             ]
#         )
        
#         # 输出归一化
#         self.norm_f = nn.LayerNorm(d_model, eps=1e-5)
        
#         # 添加双向扫描标志
#         self.if_bidirectional = True  # 控制是否启用双向扫描
        
#     def forward(self, input_map, returns_attns=False):
#         # CNN特征提取
#         map_feat = self.map_fe(input_map)  # [B, D, Hf, Wf]
#         conv_map_shape = map_feat.shape[-2:]
#         map_tokens = self.reorder_dims(map_feat)                 # [B, N_map, D]
        
#         # 简单断言：确保 token 维度与 block 期望一致，若不一致在早期报错以便定位
#         if len(self.layers) > 0:
#             expected_dim = getattr(self.layers[0], "dim", None)
#             if expected_dim is not None and map_tokens.shape[-1] != expected_dim:
#                 raise RuntimeError(f"Feature dim mismatch: map_tokens.dim={map_tokens.shape[-1]} but block.dim={expected_dim}. "
#                                    "请检查 d_model / block 配置。")

#         # 位置编码
#         map_tokens = self.position_enc(map_tokens, conv_shape=conv_map_shape if not self.training else None)
#         map_tokens = self.dropout(map_tokens)
        
#         map_tokens = self.input_ln(map_tokens)
        
#         # 使用 zeta.nn.SSM 的块逐层处理 (VisionEncoderMambaBlock 的 forward 已实现 SSM 两向处理)
#         # for layer in self.layers:
#         #     map_tokens = layer(map_tokens)
        
#         # for layer in self.layers:
#         #     out = layer(map_tokens)
#         #     # 兼容上游 layer 可能返回 (hidden_states, residual) 或类似 tuple
#         #     if isinstance(out, (tuple, list)):
#         #         if len(out) == 0:
#         #             raise RuntimeError("Encoder layer returned empty tuple/list")
#         #         map_tokens = out[0]
#         #     else:
#         #         map_tokens = out
        
#         # # 最终归一化（保持与原来接口一致）
#         # map_tokens = self.norm_f(map_tokens.to(dtype=self.norm_f.weight.dtype))
        
#         # 初始化残差
#         residual = None
        
#         # # 双向扫描实现
#         # if not self.if_bidirectional:
#         #     # 单向扫描处理
#         #     for layer in self.layers:
#         #         # 处理每个层
#         #         out = layer(map_tokens, residual)
#         #         # 兼容上游 layer 可能返回 (hidden_states, residual) 或类似 tuple
#         #         if isinstance(out, (tuple, list)):
#         #             if len(out) == 0:
#         #                 raise RuntimeError("Encoder layer returned empty tuple/list")
#         #             map_tokens, residual = out
#         #         else:
#         #             map_tokens = out
#         #             residual = None  # 重置残差
#         # else:
#         #     # 双向扫描处理 - 需要成对的层
#         #     if len(self.layers) % 2 != 0:
#         #         raise ValueError("For bidirectional scanning, the number of layers must be even")
            
#         #     for i in range(len(self.layers) // 2):
#         #         # 前向扫描
#         #         forward_layer = self.layers[i * 2]
#         #         out_forward = forward_layer(map_tokens, residual)
                
#         #         if isinstance(out_forward, (tuple, list)):
#         #             map_tokens_f, residual_f = out_forward
#         #         else:
#         #             map_tokens_f = out_forward
#         #             residual_f = None
                
#         #         # 后向扫描（翻转序列）
#         #         map_tokens_flipped = map_tokens.flip(1)
#         #         residual_flipped = None if residual is None else residual.flip(1)
                
#         #         backward_layer = self.layers[i * 2 + 1]
#         #         out_backward = backward_layer(map_tokens_flipped, residual_flipped)
                
#         #         if isinstance(out_backward, (tuple, list)):
#         #             map_tokens_b, residual_b = out_backward
#         #         else:
#         #             map_tokens_b = out_backward
#         #             residual_b = None
                
#         #         # 融合结果
#         #         map_tokens = map_tokens_f + map_tokens_b.flip(1)
#         #         if residual_f is not None and residual_b is not None:
#         #             residual = residual_f + residual_b.flip(1)
#         #         else:
#         #             residual = None
        
#         # 逐层调用：注意 VisionMambaLayer 内部已经实现了 forward/backward SSM + z 合并
#         for layer in self.layers:
#             out = layer(map_tokens, residual)
#             if isinstance(out, (tuple, list)):
#                 if len(out) == 0:
#                     raise RuntimeError("Encoder layer returned empty tuple/list")
#                 map_tokens = out[0]
#                 residual = out[1] if len(out) > 1 else residual
#             else:
#                 map_tokens = out
        
#         # 最终归一化
#         if residual is not None:
#             # 如果有残差，应用最终归一化
#             map_tokens = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
#         else:
#             map_tokens = self.norm_f(map_tokens.to(dtype=self.norm_f.weight.dtype))
            
#         return map_tokens
