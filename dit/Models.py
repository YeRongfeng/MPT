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
        
        # 使用DiT Block
        self.timestep_embedder = TimestepEmbedder(d_model)
        self.layer_stack = nn.ModuleList([
            DiTBlock(
                hidden_size=d_model,
                num_heads=n_heads,
                mlp_ratio=d_inner / d_model,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

    def forward(self, input_map, timesteps=None, returns_attns=False):
        """
        Args:
            input_map: (B, 6, H, W) 输入地图
            timesteps: (B,) 时间步，可选。如果为None则使用0
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
        
        # 如果没有提供时间步，使用固定值（用于非扩散场景）
        if timesteps is None:
            timesteps = torch.zeros(input_map.shape[0], 
                                    device=input_map.device, 
                                    dtype=torch.long)
        
        # 获取时间步嵌入
        c = self.timestep_embedder(timesteps)  # [B, D]
        
        # 通过DiT Block
        for dit_block in self.layer_stack:
            map_tokens = dit_block(map_tokens, c)
            
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

    def forward(self, map_input, timesteps=None):
        """
        Args:
            map_input: (B, 6, H, W) 输入地图
            timesteps: (B,) 可选的时间步
        Returns:
            seq_logit_softmax: (B, N, output_dim) 分类预测
            correction_reshaped: (B, N, 3, output_dim) 修正预测
        """
        # 编码阶段 - 传递timesteps
        map_tokens, *_ = self.encoder(map_input, timesteps=timesteps)
        
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
    基于DiT的路径扩散模型
    
    【核心改变】
    不再预测锚点概率，而是直接生成完整路径：
    - 输入：noisy_path (B, 10, 3) + map (B, 6, H, W) + timestep
    - 输出：predicted_noise (B, 10, 3)
    
    【路径表示】
    每步路径为3D向量：[x坐标, y坐标, yaw角度]
    """
    
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, 
                 pad_idx, dropout, n_position, train_shape, 
                 n_path_steps=10, diffusion_steps=1000):
        super().__init__()
        
        # 地图编码器（复用DiT架构）
        self.map_encoder = UnevenEncoder(
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
        
        # 路径编码器：将noisy path编码为tokens
        self.path_encoder = nn.Sequential(
            nn.Linear(3, d_model // 2),  # [x, y, yaw] -> d_model//2
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 路径位置编码（区分10个步骤）
        self.path_pos_embed = nn.Parameter(
            torch.randn(1, n_path_steps, d_model) * 0.02
        )
        
        # 时间步嵌入
        self.time_embedder = TimestepEmbedder(d_model)
        
        # 交叉注意力DiT Blocks（路径attend to地图）
        self.cross_dit_blocks = nn.ModuleList([
            CrossAttentionDiTBlock(
                hidden_size=d_model,
                num_heads=n_heads,
                mlp_ratio=4.0,
                dropout=dropout
            )
            for _ in range(n_layers // 2)  # 用一半层做交叉注意力
        ])
        
        # 自注意力DiT Blocks（路径内部注意力）
        self.self_dit_blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=d_model,
                num_heads=n_heads,
                mlp_ratio=4.0,
                dropout=dropout
            )
            for _ in range(n_layers // 2)
        ])
        
        # 噪声预测头
        self.noise_pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),  # 预测 [x_noise, y_noise, yaw_noise]
            nn.Tanh()  # 限制输出范围在[-1, 1]
        )
        
        self.n_path_steps = n_path_steps
        self.diffusion_steps = diffusion_steps
        
        # 扩散调度器
        self.register_buffer('betas', self._cosine_beta_schedule(diffusion_steps))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1.0 - self.alphas_cumprod))
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """余弦调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, map_input, noisy_path, timestep):
        """
        训练时前向传播
        
        Args:
            map_input: (B, 6, H, W) 地图
            noisy_path: (B, 10, 3) 加噪的路径 [x, y, yaw]
            timestep: (B,) 扩散时间步
        Returns:
            pred_noise: (B, 10, 3) 预测的噪声
        """
        B = map_input.shape[0]
        
        # 1. 编码地图
        map_tokens, *_ = self.map_encoder(map_input, timesteps=timestep)  # (B, 144, D)
        
        # 2. 编码noisy path
        path_tokens = self.path_encoder(noisy_path)  # (B, 10, D)
        path_tokens = path_tokens + self.path_pos_embed  # 添加路径位置编码
        
        # 3. 时间步嵌入
        t_emb = self.time_embedder(timestep)  # (B, D)
        
        # 4. 交叉注意力：路径特征关注地图特征
        for cross_block in self.cross_dit_blocks:
            path_tokens = cross_block(path_tokens, map_tokens, t_emb)
        
        # 5. 自注意力：路径内部建模
        for self_block in self.self_dit_blocks:
            path_tokens = self_block(path_tokens, t_emb)
        
        # 6. 预测噪声
        pred_noise = self.noise_pred_head(path_tokens)  # (B, 10, 3)
        
        return pred_noise
    
    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散：x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(self, map_input, x_t, t):
        """
        逆向去噪一步
        """
        B = x_t.shape[0]
        
        # 预测噪声
        pred_noise = self.forward(map_input, x_t, t)
        
        # 计算系数
        alpha_t = self.alphas_cumprod[t][:, None, None]
        alpha_t_prev = self.alphas_cumprod[t-1][:, None, None] if t[0] > 0 else torch.ones_like(alpha_t)
        beta_t = self.betas[t][:, None, None]
        
        # 预测 x_0 并裁剪
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        pred_x0 = torch.clamp(pred_x0, min=-1.5, max=1.5)  # 稍微放宽范围
        
        # DDPM均值
        mean = torch.sqrt(alpha_t_prev) * beta_t / (1 - alpha_t) * pred_x0 + \
               torch.sqrt(1 - beta_t) * (1 - alpha_t_prev) / (1 - alpha_t) * x_t
        
        # 添加噪声
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = beta_t
            x_t_next = mean + torch.sqrt(variance) * noise
            # 每步都轻微裁剪
            x_t_next = torch.clamp(x_t_next, min=-2.0, max=2.0)
            return x_t_next
        else:
            # 最后一步严格裁剪到 [-1, 1]
            return torch.clamp(mean, min=-1.0, max=1.0)
    
    @torch.no_grad()
    def sample(self, map_input, num_samples=5):
        """
        DDPM采样：生成多条路径
        
        Args:
            map_input: (1, 6, H, W)
            num_samples: 生成路径数量
        Returns:
            paths: (num_samples, 10, 3)
        """
        device = map_input.device
        
        # 扩展map_input以支持批量采样
        map_input_expanded = map_input.expand(num_samples, -1, -1, -1)
        
        # 从纯噪声开始
        x_t = torch.randn(num_samples, self.n_path_steps, 3, device=device)
        x_t = torch.clamp(x_t, min=-2.0, max=2.0)
        
        # 逐步去噪
        for t in reversed(range(self.diffusion_steps)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(map_input_expanded, x_t, t_batch)
        
        # 最终确保在 [-1, 1] 范围
        x_t = torch.clamp(x_t, min=-1.0, max=1.0)
        
        return x_t  # (num_samples, 10, 3)


class CrossAttentionDiTBlock(nn.Module):
    """
    带交叉注意力的DiT Block
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        
        # adaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size)  # 3组 shift/scale/gate
        )
    
    def forward(self, x, context, c):
        """
        Args:
            x: (B, L_query, D) 路径特征
            context: (B, L_context, D) 地图特征
            c: (B, D) 时间步嵌入
        """
        # 生成调制参数
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(9, dim=1)
        
        # Self-attention
        x_normed = self.modulate(self.norm1(x), shift_sa, scale_sa)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(x_normed, x_normed, x_normed)[0]
        
        # Cross-attention
        x_normed = self.modulate(self.norm2(x), shift_ca, scale_ca)
        x = x + gate_ca.unsqueeze(1) * self.cross_attn(x_normed, context, context)[0]
        
        # MLP
        x_normed = self.modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_normed)
        
        return x
    
    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)