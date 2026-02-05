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
- 自定义Transformer层（EncoderLayer, DecoderLayer, PoseWiseEncoderLayer）
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
from dit.DiTLayers import DiTBlock, TimestepEmbedder
from transformer.Layers import EncoderLayer, DecoderLayer, PoseWiseEncoderLayer

from einops.layers.torch import Rearrange
from einops import rearrange


import math


class DiTBlock(nn.Module):
    """标准DiT Block
    
    架构（来自DiT论文）：
        Input --> LayerNorm --> Self-Attention --> Residual
              --> LayerNorm --> FFN --> Residual
              --> AdaLN modulation (时间步和条件调制)
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )
        
        # 关键！AdaLN-Zero 初始化（DiT原论文）
        # 将gate参数初始化为0，使训练初期DiT block是恒等映射
        # 这样网络可以从简单到复杂逐步学习，避免梯度消失
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        
    def forward(self, x, c):
        """
        Args:
            x: (B, N, D) - 输入tokens
            c: (B, D) - 全局条件（时间步 + 其他条件）
            
        Returns:
            x: (B, N, D) - 输出tokens
        """
        # AdaLN调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Self-attention分支
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x_self, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * x_self
        
        # FFN分支
        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x_ffn = self.ffn(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * x_ffn
        
        return x


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


class UnevenEncoder(nn.Module):
    """    
    【架构设计】
    输入地图 -> CNN特征提取 -> Patch Embedding -> 位置编码 -> DiT Blocks -> 输出特征
    """

    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape):
        super().__init__()
        
        self.map_fe = nn.Sequential(
            # Block 1
            nn.Conv2d(6, d_model//8, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=d_model//8),  # 使用GroupNorm替代BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(2),  # 50×50
            
            # Block 2
            nn.Conv2d(d_model//8, d_model//4, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=d_model//4),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25×25
            
            # Block 3
            nn.Conv2d(d_model//4, d_model//2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=d_model//2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12×12
            
            # Block 4
            nn.Conv2d(d_model//2, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=d_model),
            nn.ReLU(),             
        )

        self.reorder_dims = Rearrange('b c h w -> b (h w) c')
        self.position_enc = PositionalEncoding(d_model, n_position=n_position, train_shape=train_shape)
        self.dropout = nn.Dropout(p=dropout)
        
        # 使用普通的Transformer EncoderLayer（不需要时间步调制）
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input_map, returns_attns=False):
        """
        Args:
            input_map: (B, 6, H, W) 输入地图
            returns_attns: 是否返回注意力（当前未使用）
        Returns:
            map_tokens: (B, N, D) 编码后的特征
        """
        enc_slf_attn_list = []

        # CNN -> map tokens
        map_feat = self.map_fe(input_map)  # [B, D, Hf, Wf]
        conv_map_shape = map_feat.shape[-2:]
        map_tokens = self.reorder_dims(map_feat)  # [B, N_map, D]
        map_tokens = self.position_enc(
            map_tokens, 
            conv_shape=conv_map_shape if not self.training else None
        )
        
        map_tokens = self.dropout(map_tokens)
        map_tokens = self.layer_norm(map_tokens)
        
        # 通过普通的Transformer EncoderLayer
        for enc_layer in self.layer_stack:
            map_tokens = enc_layer(map_tokens, slf_attn_mask=None)
            
        if returns_attns:
            return map_tokens, enc_slf_attn_list
        return map_tokens,  # 返回元组以保持接口一致


class Decoder(nn.Module):
    """
    Decoder - Transformer解码器用于序列生成
    """

    def __init__(self, patch_size, n_layers, n_heads, d_k , d_v, d_model, d_inner, pad_idx, stride, n_position, dropout=0.1):
        """
        初始化Transformer解码器
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
 
class UnevenTransformer(Transformer):
    """
    UnevenTransformer - 用于处理不平坦地面路径规划的Transformer变体
    """
    
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape, output_dim=10):
        nn.Module.__init__(self)
        
        # 重新定义编码器
        self.encoder = UnevenEncoder(
            n_layers=n_layers,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_inner=d_inner,
            pad_idx=pad_idx,
            dropout=dropout,
            n_position=n_position,
            train_shape=train_shape
        )
        
        # 分类头
        self.classPred = nn.Sequential(
            Rearrange('b c d_model -> (b c) d_model 1 1'),
            nn.Conv2d(d_model, output_dim, kernel_size=1),
            Rearrange('bc d 1 1 -> bc d'),
        )
        
        # 修正预测头
        self.correctionPred = nn.Sequential(
            Rearrange('b c d_model -> (b c) d_model 1 1'),
            nn.Conv2d(d_model + output_dim, 3*output_dim, kernel_size=1),
            Rearrange('bc d 1 1 -> bc d'),
        )

    def forward(self, map_input):
        """
        Args:
            map_input: (B, 6, H, W) 输入地图
        Returns:
            seq_logit_softmax: (B, N, output_dim) 分类预测
            correction_reshaped: (B, N, 3, output_dim) 修正预测
        """
        # 编码阶段
        map_tokens, *_ = self.encoder(map_input)
        
        # 分类预测
        seq_logit = self.classPred(map_tokens)
        batch_size = map_input.shape[0]
        seq_logit_reshaped = rearrange(seq_logit, '(b c) d -> b c d', b=batch_size)
        
        # Softmax归一化
        seq_logit_softmax = F.softmax(seq_logit_reshaped, dim=1)
        
        # 拼接特征
        combined_features = torch.cat([map_tokens, seq_logit_softmax], dim=-1)
        correction = self.correctionPred(combined_features)
        
        correction_sigmoid = F.sigmoid(correction)
        
        # 重排修正结果
        correction_reshaped = rearrange(correction_sigmoid, '(b c) (n d) -> b c n d', b=batch_size, n=3)
        
        return seq_logit_softmax, correction_reshaped


class PathDiffusionTransformer(nn.Module):
    """
    基于DiT架构的路径扩散模型（从UnevenTransformer改造）
    
    【架构设计】
    原始UnevenTransformer：地图 → CNN → Transformer blocks → 预测头
    改造为扩散模型：地图+噪声路径 → CNN特征提取 → token融合 → DiT blocks(时间步条件) → 噪声预测
    
    【核心改动】
    1. 输入增加：噪声路径 x_t
    2. Transformer blocks → DiT blocks（增加时间步条件）
    3. 输出改为：预测噪声 ε
    
    【训练目标】
    - 损失：MSE(pred_noise, true_noise)
    
    【采样】
    - DDIM采样
    """
    
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, 
                 pad_idx, dropout, n_position, train_shape, 
                 n_path_steps=20, diffusion_steps=50, prediction_type='epsilon', loss_type=None):
        """
        Args:
            prediction_type: 'epsilon' (预测噪声), 'x0' (预测原始数据), 或 'v' (预测velocity)
            loss_type: 'epsilon', 'x0', 或 'v' - 损失函数的目标类型，如果为None则与prediction_type相同
        """
        super().__init__()
        
        self.prediction_type = prediction_type
        self.loss_type = loss_type if loss_type is not None else prediction_type
        
        # ========== 地图CNN特征提取（保留原架构）==========
        self.map_fe = nn.Sequential(
            # Block 1
            nn.Conv2d(3, d_model//8, kernel_size=3, padding=1),
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
        
        self.reorder_dims = Rearrange('b c h w -> b (h w) c')
        self.position_enc = PositionalEncoding(d_model, n_position=n_position, train_shape=train_shape)
        self.dropout = nn.Dropout(p=dropout)
        
        # ========== 地图全局pooling ==========
        self.map_pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, D, H, W) -> (B, D, 1, 1)
        
        # ========== 条件融合MLP ==========
        # 将时间步、地图特征和起点终点融合
        self.condition_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 4),  # 3 = time + map + start_goal
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # ========== 起点终点编码（改进：分开处理）==========
        # 关键改进1：起点和终点独立编码，保持语义清晰
        self.start_embedder = nn.Sequential(
            nn.Linear(4, d_model),  # start(x,y,sin(θ),cos(θ)) - 4维角度编码
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.goal_embedder = nn.Sequential(
            nn.Linear(4, d_model),  # goal(x,y,sin(θ),cos(θ)) - 4维角度编码
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        # 关键改进2：学习起点终点的关系（方向、距离等）
        self.start_goal_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # ========== 路径Patchify ==========
        self.path_patchify = nn.Sequential(
            nn.Linear(4, d_model // 2),  # 4维输入: (x,y,sin(θ),cos(θ))
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 路径位置编码（可学习）
        self.path_pos_embed = nn.Parameter(
            torch.randn(1, n_path_steps, d_model) * 0.02
        )
        
        # ========== 时间步嵌入 ==========
        self.time_embedder = TimestepEmbedder(d_model)
        
        # ========== 改进的DiT Blocks ==========
        # 标准DiT架构：Self-Attention + FFN + AdaLN调制
        self.dit_blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model,
                n_heads=n_heads, 
                d_ff=d_model * 4,  # FFN hidden dim (DiT标准设置)
                dropout=dropout
            ) 
            for _ in range(n_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # ========== 解耦预测头设计 ==========
        # 注意：final_norm 使用 elementwise_affine=False 会阻断梯度流
        # main_pred 内部已有 LayerNorm，不需要额外的归一化层
        # self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        
        # 主预测头（dit_blocks前已有LayerNorm，无需重复）
        self.main_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 4)  # (x, y, sin, cos)
        )
        
        # # 细节预测头（用于精细化调整主预测）- 简化为两层
        # detail_in_dim = d_model + 4  # DiT特征 + 主预测的完整4维
        # self.detail_pred = nn.Sequential(
        #     nn.Linear(detail_in_dim, d_model // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model // 2, 4)  # 输出增量 (dx, dy, d_sin, d_cos)
        # )
        
        # # Zero初始化细节预测头
        # nn.init.zeros_(self.detail_pred[-1].weight)
        # nn.init.zeros_(self.detail_pred[-1].bias)
        
        # ========== 扩散参数（时间t参数化）==========
        self.n_path_steps = n_path_steps
        self.diffusion_steps = diffusion_steps
        
        # 时间步 t 的参数化（不再使用 beta/alpha）
        # 索引语义与标准DDPM一致：
        #   - timestep索引 t=0 → t_normalized=1.0 (纯数据)
        #   - timestep索引 t=T-1 → t_normalized=0.0 (纯噪声)
        # 加噪公式：z = t_normalized * x + (1 - t_normalized) * e
        #   - t=0: z = 1*x + 0*e = x (纯数据)
        #   - t=T-1: z = 0*x + 1*e = e (纯噪声)
        self.register_buffer('timesteps_normalized', torch.linspace(1, 0, diffusion_steps))
        
        # 防止除零的小常数（v-loss需要更大的值避免数值不稳定）
        # 参考文生图模型：当t接近1时，(1-t)可能非常小，导致v的计算梯度爆炸
        # t_eps=0.05确保分母不会太小，避免数值问题
        self.t_eps = 0.05
        
        # 时间步采样策略参数（用于训练时的智能采样）
        # P_mean=-0.8：偏向较小的时间步（更接近干净数据的区域）
        # P_std=0.8：控制采样的分散程度
        self.register_buffer('t_sample_mean', torch.tensor(-0.8))
        self.register_buffer('t_sample_std', torch.tensor(0.8))
    
    def sample_timesteps(self, n: int, device=None):
        """
        智能时间步采样策略（参考文生图模型）
        
        使用sigmoid正态分布采样，避免极端值：
        - 正态分布采样 → sigmoid映射到(0,1) → 转换为timestep索引
        - P_mean=-0.8偏向较小时间步（接近干净数据）
        - 避免t→0和t→1的极端情况
        
        Args:
            n: 采样数量
            device: 设备
        
        Returns:
            timestep_indices: (n,) 时间步索引
        """
        if device is None:
            device = self.timesteps_normalized.device
        
        # 正态分布采样
        z = torch.randn(n, device=device) * self.t_sample_std + self.t_sample_mean
        # sigmoid映射到(0,1)
        t_continuous = torch.sigmoid(z)
        # 转换为timestep索引（注意：t_normalized从1到0，所以需要反向映射）
        # t_continuous=0 → 索引T-1 (纯噪声)
        # t_continuous=1 → 索引0 (纯数据)
        timestep_indices = ((1 - t_continuous) * (self.diffusion_steps - 1)).long()
        timestep_indices = torch.clamp(timestep_indices, 0, self.diffusion_steps - 1)
        
        return timestep_indices
    
    def forward(self, map_input, noisy_path, timestep, start_pose, goal_pose):
        """
        扩散模型前向传播（改造自UnevenTransformer）
        
        Args:
            map_input: (B, 3, H, W) 输入地图
            noisy_path: (B, n_path_steps, 4) 加噪后的路径 x_t (x,y,sin(θ),cos(θ))
            timestep: (B,) 扩散时间步 t ∈ [0, diffusion_steps-1]
            start_pose: (B, 4) 起点坐标 (x,y,sin(θ),cos(θ)) - 已归一化
            goal_pose: (B, 4) 终点坐标 (x,y,sin(θ),cos(θ)) - 已归一化
            
        Returns:
            model_output: (B, n_path_steps, 4) 
                - 如果 prediction_type='epsilon': 预测的噪声 ε
                - 如果 prediction_type='x0': 预测的原始数据 x_0 (x,y,sin(θ),cos(θ))
        """
        B = map_input.shape[0]
        
        # ========== 步骤1：CNN提取地图特征 ==========
        map_feat = self.map_fe(map_input)  # (B, D, Hf, Wf)
        conv_map_shape = map_feat.shape[-2:]
        map_tokens = self.reorder_dims(map_feat)  # (B, N_map, D)
        
        # 添加位置编码
        map_tokens = self.position_enc(
            map_tokens, 
            conv_shape=conv_map_shape if not self.training else None
        )
        
        # ========== 步骤2：起点终点作为独立tokens（改进）==========
        # 关键改进：显式地将起点终点加入序列，而非仅作为条件
        c_start = self.start_embedder(start_pose).unsqueeze(1)  # (B, 1, D)
        c_goal = self.goal_embedder(goal_pose).unsqueeze(1)     # (B, 1, D)
        
        # ========== 步骤3：路径Patchify ==========
        path_tokens = self.path_patchify(noisy_path)  # (B, n_path_steps, D)
        path_tokens = path_tokens + self.path_pos_embed
        
        # ========== 步骤4：融合所有tokens（改进）==========
        # 关键改进：序列变为 [起点, 终点, 地图tokens, 路径tokens]
        # 这样路径可以直接attend到起点/终点，不仅仅是通过条件调制
        combined_tokens = torch.cat([
            c_start,      # [B, 1, D] - 起点token
            c_goal,       # [B, 1, D] - 终点token  
            map_tokens,   # [B, N_map, D] - 地图空间tokens
            path_tokens   # [B, n_path_steps, D] - 路径tokens
        ], dim=1)  # (B, 2+N_map+n_path_steps, D)
        
        # 正确顺序：LayerNorm → Dropout
        combined_tokens = self.layer_norm(combined_tokens)
        combined_tokens = self.dropout(combined_tokens)
        
        # ========== 步骤5：全局条件（用于DiT调制）==========
        c_time = self.time_embedder(timestep)  # (B, D) - 时间步条件
        map_global = self.map_pool(map_feat).squeeze(-1).squeeze(-1)  # (B, D) - 地图全局特征
        
        # 起点终点的关系特征（方向、距离等）
        c_start_flat = c_start.squeeze(1)  # (B, D)
        c_goal_flat = c_goal.squeeze(1)    # (B, D)
        c_start_goal = self.start_goal_fusion(torch.cat([c_start_flat, c_goal_flat], dim=-1))  # (B, D)
        
        # 【阶段1】基础条件：时间 + 地图 + 起点终点关系
        c = self.condition_mlp(torch.cat([c_time, map_global, c_start_goal], dim=-1))  # (B, D)
        
        # ========== 步骤6：DiT Blocks ==========
        # 标准DiT架构：所有tokens通过self-attention交互
        
        for dit_block in self.dit_blocks:
            combined_tokens = dit_block(x=combined_tokens, c=c)
        
        # ========== 步骤7：预测头 ==========
        # 序列结构：[起点(1), 终点(1), 地图(N_map), 路径(n_path_steps)]
        # 只取最后n_path_steps个tokens作为路径预测
        path_tokens_out = combined_tokens[:, -(self.n_path_steps):, :]  # (B, n_path_steps, D)
        
        # 主预测头：粗略预测
        main_output = self.main_pred(path_tokens_out)  # (B, n_path_steps, 4)
        
        # # 细节预测头：基于主预测和DiT特征的精细化增量
        # detail_input = torch.cat([path_tokens_out, main_output], dim=-1)  # (B, n_path_steps, D+4)
        # detail_output = self.detail_pred(detail_input)  # (B, n_path_steps, 4) - 增量
        
        # # 最终输出 = 主预测 + 细节增量
        # model_output = main_output + detail_output  # (B, n_path_steps, 4)
        
        model_output = main_output
        
        return model_output
    
    def q_sample(self, x_start, t, noise=None):
        """
        前向过程（训练时使用）
        
        新参数化：
            z = t * x + (1 - t) * e
            其中 t ∈ [0, 1]：
                t=0: z = e (纯噪声)
                t=1: z = x (纯数据)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 获取归一化的时间步 t (0到1)
        t_normalized = self.timesteps_normalized[t][:, None, None]  # (B, 1, 1)
        
        # 线性插值：z = t * x + (1 - t) * e
        z = t_normalized * x_start + (1 - t_normalized) * noise
        
        return z
    
    def compute_velocity(self, x_pred, z, t):
        """
        从预测的 x_pred 计算速度 v
        
        v = (x_pred - z) / (1 - t)
        
        Args:
            x_pred: (B, N, 4) - 网络预测的 x
            z: (B, N, 4) - 加噪后的轨迹
            t: (B,) - 时间步索引
        
        Returns:
            v_pred: (B, N, 4) - 预测的速度
        """
        # 获取归一化的时间步 t (0到1)
        t_normalized = self.timesteps_normalized[t][:, None, None]  # (B, 1, 1)
        
        # v = (x_pred - z) / (1 - t)
        v_pred = (x_pred - z) / torch.clamp(1 - t_normalized, min=self.t_eps)
        
        return v_pred
    
    @torch.no_grad()
    def p_sample(self, map_input, x_t, t, start_pose, goal_pose):
        """
        单步去噪/采样（新参数化）
        
        新参数化 (DDIM):
            z_t = t * x + (1-t) * e
            z_{t-1} = t_{prev} * x_pred + (1-t_{prev}) * e
        
        Args:
            map_input: (B, 3, H, W) 输入地图
            x_t: (B, n_path_steps, 4) 当前时刻的加噪轨迹
            t: (B,) 当前时间步
            start_pose: (B, 4) 起点
            goal_pose: (B, 4) 终点
        """
        # 获取模型输出
        model_output = self.forward(map_input, x_t, t, start_pose, goal_pose)
        
        # 获取归一化的时间步
        t_normalized = self.timesteps_normalized[t][:, None, None]
        
        # 根据prediction_type解析模型输出
        if self.prediction_type == 'epsilon':
            pred_epsilon = model_output
            pred_x0 = (x_t - (1 - t_normalized) * pred_epsilon) / torch.clamp(t_normalized, min=self.t_eps)
        elif self.prediction_type == 'x0':
            pred_x0 = model_output
            pred_epsilon = (x_t - t_normalized * pred_x0) / torch.clamp(1 - t_normalized, min=self.t_eps)
        elif self.prediction_type == 'v':
            pred_v = model_output
            pred_x0 = x_t + (1 - t_normalized) * pred_v
            pred_epsilon = (x_t - t_normalized * pred_x0) / torch.clamp(1 - t_normalized, min=self.t_eps)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
        
        # 裁剪位置坐标
        pred_x0[:, :, :2] = torch.clamp(pred_x0[:, :, :2], -1.0, 1.0)
        
        # 最后一步直接返回x_0
        if t[0] == 0:
            return pred_x0
        
        # DDIM公式：z_{t-1} = t_{prev} * x_pred + (1-t_{prev}) * e
        t_prev_normalized = self.timesteps_normalized[t-1][:, None, None]
        x_prev = t_prev_normalized * pred_x0 + (1 - t_prev_normalized) * pred_epsilon
        
        return x_prev
    
    def differentiable_sample(self, map_input, start_pose, goal_pose, ddim_steps=50, use_checkpoint=True):
        """
        可微分的DDIM采样（ODE方法，保持梯度连接）
        
        **原理**：
        DDIM本质上是概率流ODE (Probability Flow ODE)：
            dx/dt = -0.5 * β(t) * [x + (1-α(t)) * ∇_x log p(x)]
        
        在离散化后，ODE更新是确定性的且完全可微：
            x_{t-1} = √ᾱ_{t-1} * pred_x0 + √(1-ᾱ_{t-1}) * pred_ε
        
        与标准采样的区别：
        1. 不使用 @torch.no_grad()，保持梯度图
        2. 使用模型的forward()而非p_sample()（后者有@torch.no_grad装饰器）
        3. 手动实现DDIM更新逻辑，确保所有操作都可微
        
        **显存优化**：
        使用梯度检查点(gradient checkpointing)减少显存占用：
        - 前向传播时不保存中间激活值
        - 反向传播时重新计算需要的激活值
        - 显存消耗从O(n)降到O(1)，但计算量增加约2倍
        
        Args:
            map_input: (B, 3, H, W) 输入地图
            start_pose: (B, 4) 起点坐标 (x,y,sin(θ),cos(θ)) (已归一化)
            goal_pose: (B, 4) 终点坐标 (x,y,sin(θ),cos(θ)) (已归一化)
            ddim_steps: DDIM步数（默认50，与标准采样一致）
            use_checkpoint: 是否使用梯度检查点节省显存（默认True）
        
        Returns:
            x_0: (B, n_path_steps, 4) 采样得到的轨迹（保持梯度）
        """
        device = map_input.device
        B = map_input.shape[0]
        
        # 从标准正态分布初始化 (4维: x,y,sin(θ),cos(θ))
        # detach噪声以避免在反向传播时计算噪声的梯度（只需要模型参数的梯度）
        x_t = torch.randn(B, self.n_path_steps, 4, device=device).detach()
        
        # DDIM时间步调度
        step_size = max(1, self.diffusion_steps // ddim_steps)
        time_steps = list(range(self.diffusion_steps-1, 0, -step_size)) + [0]
        
        # 定义单步更新函数（用于梯度检查点）
        def ddim_step(x_in, t_val):
            """
            单步DDIM更新（ODE形式，新参数化）
            
            新参数化的DDIM更新：
                z_t = t * x + (1-t) * e
                预测 x_pred，然后更新 z_{t-1}
            
            Args:
                x_in: (B, N, 4) - 当前时刻的轨迹 z_t
                t_val: int - 当前时间步
            
            Returns:
                x_out: (B, N, 4) - 去噪后的轨迹 z_{t-1}
            """
            # 创建时间步tensor
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
            
            # 模型预测（保持梯度）
            model_output = self.forward(map_input, x_in, t_tensor, start_pose, goal_pose)
            
            # 获取归一化的时间步
            t_norm = self.timesteps_normalized[t_val]
            t_prev_norm = self.timesteps_normalized[t_val - 1] if t_val > 0 else torch.tensor(1.0, device=device)
            
            # 根据prediction_type解析输出
            if self.prediction_type == 'epsilon':
                # 网络预测噪声 e，反推 x_pred = (z - (1-t)*e) / t
                pred_epsilon = model_output
                pred_x0 = (x_in - (1 - t_norm) * pred_epsilon) / torch.clamp(t_norm, min=self.t_eps)
            elif self.prediction_type == 'x0':
                # 网络直接预测 x
                pred_x0 = model_output
                pred_epsilon = (x_in - t_norm * pred_x0) / torch.clamp(1 - t_norm, min=self.t_eps)
            elif self.prediction_type == 'v':
                # 网络预测速度 v = (x - z) / (1-t)
                pred_v = model_output
                pred_x0 = x_in + (1 - t_norm) * pred_v
                pred_epsilon = (x_in - t_norm * pred_x0) / torch.clamp(1 - t_norm, min=self.t_eps)
            else:
                raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
            
            # 裁剪位置坐标到归一化范围 [-1, 1]（避免in-place操作）
            pred_x0_pos_clamped = torch.clamp(pred_x0[:, :, :2], -1.0, 1.0)  # (B, N, 2)
            pred_x0_angle = pred_x0[:, :, 2:4]  # (B, N, 2)
            pred_x0 = torch.cat([pred_x0_pos_clamped, pred_x0_angle], dim=2)  # (B, N, 4)
            
            # DDIM去噪
            if t_val == 0:
                # 最后一步直接返回 pred_x0
                x_out = pred_x0
            else:
                # DDIM更新：z_{t-1} = t_{prev} * x_pred + (1 - t_{prev}) * e
                x_out = t_prev_norm * pred_x0 + (1 - t_prev_norm) * pred_epsilon
            
            return x_out
        
        # 迭代ODE采样（保持梯度连接）
        for i, t in enumerate(time_steps):
            if use_checkpoint:
                # 使用梯度检查点减少显存（以计算时间换空间）
                x_t = torch.utils.checkpoint.checkpoint(
                    ddim_step, x_t, t, use_reentrant=False
                )
            else:
                # 标准前向传播（占用更多显存）
                x_t = ddim_step(x_t, t)
        
        # 最终处理（避免in-place操作）
        # 裁剪位置坐标
        x_t_pos_clamped = torch.clamp(x_t[:, :, :2], -1.0, 1.0)  # (B, N, 2)
        
        # 归一化sin/cos部分，确保 sin²+cos² = 1
        sin_cos = x_t[:, :, 2:4]  # (B, n_path_steps, 2)
        norm = torch.sqrt(sin_cos[:, :, 0]**2 + sin_cos[:, :, 1]**2).unsqueeze(-1) + 1e-8
        sin_cos_normalized = sin_cos / norm  # (B, N, 2)
        
        # 拼接最终结果
        x_t = torch.cat([x_t_pos_clamped, sin_cos_normalized], dim=2)  # (B, N, 4)
        
        return x_t
    
    @torch.no_grad()
    def sample(self, map_input, start_pose, goal_pose, num_samples=5, ddim_steps=50):
        """
        DDIM加速采样（改进版：起点终点作为显式tokens）
        使用4维角度编码: (x, y, sin(θ), cos(θ))
        
        Args:
            map_input: (1, 3, H, W) 输入地图（3通道：normal_x, normal_y, normal_z）
            start_pose: (1, 4) 起点坐标 (x,y,sin(θ),cos(θ)) (已归一化)
            goal_pose: (1, 4) 终点坐标 (x,y,sin(θ),cos(θ)) (已归一化)
            num_samples: 采样数量
            ddim_steps: DDIM步数
        """
        device = map_input.device
        map_input = map_input.expand(num_samples, -1, -1, -1)
        start_pose = start_pose.expand(num_samples, -1)
        goal_pose = goal_pose.expand(num_samples, -1)
        
        # 从标准正态分布初始化 (4维: x,y,sin(θ),cos(θ))
        x_t = torch.randn(num_samples, self.n_path_steps, 4, device=device)
        
        # DDIM时间步调度
        step_size = max(1, self.diffusion_steps // ddim_steps)
        time_steps = list(range(self.diffusion_steps-1, 0, -step_size)) + [0]
        
        for t in time_steps:
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(map_input, x_t, t_batch, start_pose, goal_pose)
        
        # 最终裁剪位置坐标到归一化范围 [-1, 1]
        x_t[:, :, :2] = torch.clamp(x_t[:, :, :2], -1.0, 1.0)
        
        # 归一化sin/cos部分，确保 sin²+cos² = 1（只在推理/采样时，修正数值误差）
        sin_cos = x_t[:, :, 2:4]  # (num_samples, n_path_steps, 2)
        norm = torch.sqrt(sin_cos[:, :, 0]**2 + sin_cos[:, :, 1]**2).unsqueeze(-1) + 1e-8  # (B, N, 1)
        x_t[:, :, 2:4] = sin_cos / norm  # 投影到单位圆
        
        return x_t
    
    def guided_sample(self, map_input, start_pose, goal_pose, cost_map, map_info, 
                      num_samples=5, ddim_steps=50, guidance_scale=0.1, 
                      guidance_start_step=0.5):
        """
        带Capsize Cost引导的DDIM采样
        
        在采样过程中注入capsize loss的梯度，引导生成更安全的轨迹
        
        Args:
            map_input: (1, 3, H, W) 输入地图
            start_pose: (1, 4) 起点 (x,y,sin(θ),cos(θ)) 已归一化
            goal_pose: (1, 4) 终点 (x,y,sin(θ),cos(θ)) 已归一化
            cost_map: (1, num_layers, max_anchors) 稳定性代价地图
            map_info: dict 地图配置信息
            num_samples: 采样数量
            ddim_steps: DDIM步数
            guidance_scale: 引导强度 (建议0.05-0.2)
            guidance_start_step: 开始引导的时间点比例 (0.0-1.0，建议0.3-0.7)
        """
        from grad_optimizer import TrajectoryOptimizerSE2
        
        device = map_input.device
        map_input = map_input.expand(num_samples, -1, -1, -1)
        start_pose_expanded = start_pose.expand(num_samples, -1)
        goal_pose_expanded = goal_pose.expand(num_samples, -1)
        
        # 扩展cost_map到batch（保持4D形状）
        if cost_map.dim() == 4:
            cost_map = cost_map.expand(num_samples, -1, -1, -1)
        else:
            cost_map = cost_map.expand(num_samples, -1, -1)
        
        # 从标准正态分布初始化
        x_t = torch.randn(num_samples, self.n_path_steps, 4, device=device)
        
        # DDIM时间步调度
        step_size = max(1, self.diffusion_steps // ddim_steps)
        time_steps = list(range(self.diffusion_steps-1, 0, -step_size)) + [0]
        
        # 确定开始引导的步骤索引
        guidance_start_idx = int(len(time_steps) * guidance_start_step)
        
        # print(f"\n🎯 Guided Sampling: steps={ddim_steps}, guidance_scale={guidance_scale}, "
        #       f"start_step={guidance_start_step} (step {guidance_start_idx}/{len(time_steps)})")
        
        for step_idx, t in enumerate(time_steps):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # 是否应用引导（只在低噪声阶段）
            apply_guidance = (step_idx >= guidance_start_idx) and (guidance_scale > 0)
            
            if apply_guidance:
                # 需要梯度的采样步骤
                x_t_guided = x_t.clone().detach().requires_grad_(True)
                
                # 获取pred_x0（需要在梯度上下文中）
                with torch.enable_grad():
                    # 前向传播获取模型输出
                    model_output = self.forward(
                        map_input, x_t_guided, t_batch, 
                        start_pose_expanded, goal_pose_expanded
                    )
                    
                    # 计算pred_x0（假设prediction_type='x0'）
                    if self.prediction_type == 'x0':
                        pred_x0 = model_output
                    elif self.prediction_type == 'epsilon':
                        sqrt_alpha_t = torch.sqrt(self.alphas_cumprod[t_batch])[:, None, None]
                        sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alphas_cumprod[t_batch])[:, None, None]
                        pred_x0 = (x_t_guided - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t
                    else:
                        raise NotImplementedError(f"Guidance not implemented for {self.prediction_type}")
                    
                    # 反归一化pred_x0
                    pred_x0_denorm = torch.zeros_like(pred_x0)
                    pred_x0_denorm[:, :, :2] = pred_x0[:, :, :2] * 20.0
                    pred_x0_denorm[:, :, 2:] = pred_x0[:, :, 2:]
                    
                    # 计算capsize cost（batch处理）
                    costs = []
                    for i in range(num_samples):
                        try:
                            # 构建完整轨迹
                            start_fixed = start_pose[0, :2] * 20.0  # 反归一化起点
                            start_angle = torch.atan2(start_pose[0, 2], start_pose[0, 3])
                            start_3d = torch.cat([start_fixed, start_angle.unsqueeze(0)])
                            
                            goal_fixed = goal_pose[0, :2] * 20.0
                            goal_angle = torch.atan2(goal_pose[0, 2], goal_pose[0, 3])
                            goal_3d = torch.cat([goal_fixed, goal_angle.unsqueeze(0)])
                            
                            predicted_angles = torch.atan2(
                                pred_x0_denorm[i, :, 2], 
                                pred_x0_denorm[i, :, 3]
                            )
                            predicted_traj_3d = torch.stack([
                                pred_x0_denorm[i, :, 0],
                                pred_x0_denorm[i, :, 1],
                                predicted_angles
                            ], dim=1)
                            
                            full_traj = torch.cat([
                                start_3d.unsqueeze(0),
                                predicted_traj_3d,
                                goal_3d.unsqueeze(0)
                            ], dim=0)
                            
                            # 计算cost
                            # 检查cost_map的形状并正确处理
                            if cost_map.dim() == 4:
                                # 形状是 (B, H, W, D)，需要转为 (D, H, W)
                                stability_cost_map = cost_map[i].permute(2, 0, 1)
                            else:
                                # 形状是 (B, num_layers, max_anchors)
                                stability_cost_map = cost_map[i].permute(2, 0, 1)
                            
                            optimizer = TrajectoryOptimizerSE2(
                                full_traj.detach(),
                                stability_cost_map,
                                map_info,
                                device=device
                            )
                            cost = optimizer.cost_on_poses(full_traj)
                            
                            if not (torch.isnan(cost) or torch.isinf(cost)):
                                costs.append(cost)
                        except Exception as e:
                            pass
                    
                    if len(costs) > 0:
                        total_cost = torch.stack(costs).mean()
                        
                        # 计算梯度
                        grad = torch.autograd.grad(total_cost, x_t_guided, retain_graph=False)[0]
                        
                        # 梯度裁剪（避免过大的扰动）
                        grad_norm = torch.norm(grad)
                        if grad_norm > 1.0:
                            grad = grad / grad_norm
                        
                        # 应用引导：沿着降低cost的方向调整x_t
                        x_t = x_t - guidance_scale * grad.detach()
                        
                        # if step_idx % 10 == 0:
                        #     print(f"  Step {step_idx}/{len(time_steps)}: t={t}, cost={total_cost.item():.4e}, "
                        #           f"grad_norm={grad_norm.item():.4e}")
            
            # 执行标准DDIM步骤（使用可能被引导调整过的x_t）
            with torch.no_grad():
                x_t = self.p_sample(map_input, x_t, t_batch, start_pose_expanded, goal_pose_expanded)
        
        # 最终裁剪和归一化
        x_t[:, :, :2] = torch.clamp(x_t[:, :, :2], -1.0, 1.0)
        sin_cos = x_t[:, :, 2:4]
        norm = torch.sqrt(sin_cos[:, :, 0]**2 + sin_cos[:, :, 1]**2).unsqueeze(-1) + 1e-8
        x_t[:, :, 2:4] = sin_cos / norm
        
        # print("✓ Guided sampling completed\n")
        
        return x_t

    def freeze_for_stage2(self):
        """
        阶段2：简化训练策略（已移除cross-attention）
        
        现在所有参数都参与训练，阶段1和阶段2没有本质区别
        仅作为接口保留，方便后续扩展
        """
        print("\n" + "="*70)
        print("阶段2：继续全参数训练")
        print("="*70)
        
        # 全参数训练
        for param in self.parameters():
            param.requires_grad = True
        
        # 统计参数
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        print(f"\n参数统计：")
        print(f"  总参数: {total:,}")
        print(f"  训练参数: {trainable:,} ({100*trainable/total:.2f}%)")
        print(f"\n训练模式：全参数训练")
        print("="*70 + "\n")
    
    def unfreeze_for_stage1(self):
        """阶段1：全参数训练"""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ 阶段1：全参数训练模式")
    
    def get_trainable_parameters(self):
        """返回当前可训练的参数（用于优化器）"""
        return [p for p in self.parameters() if p.requires_grad]


