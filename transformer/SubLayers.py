"""
SubLayers.py - Transformer子层模块定义

【核心功能】
本模块定义了构成Transformer编码器和解码器层的基础子层组件，包括缩放点积注意力、
多头注意力机制和位置前馈网络。这些子层是Transformer架构的核心构建块。

【技术特点】
1. 注意力机制：实现高效的序列建模和长距离依赖捕获
2. 多头设计：通过多个注意力头捕获不同类型的依赖关系
3. 残差连接：保证深层网络的梯度流动和训练稳定性
4. 层归一化：加速训练收敛并提升模型性能
5. 位置无关：前馈网络对每个位置独立处理

【模块组成】
- ScaledDotProductAttention：缩放点积注意力的核心计算
- MultiHeadAttention：多头注意力机制的完整实现
- PositionwiseFeedForward：位置前馈网络

【数学基础】
1. 注意力机制：Attention(Q,K,V) = softmax(QK^T/√d_k)V
2. 多头注意力：MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
3. 前馈网络：FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

技术栈：
- PyTorch 深度学习框架
- 高效的矩阵运算和并行计算
- 标准化的神经网络组件

使用场景：
- 构建Transformer编码器和解码器
- 序列到序列建模任务
- 注意力机制的研究和应用
- 深度学习模型的模块化设计

参考文献：
- Attention Is All You Need (Vaswani et al., 2017)
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    ScaledDotProductAttention - 缩放点积注意力机制
    
    【系统概述】
    实现Transformer中的核心注意力计算：缩放点积注意力。这是所有注意力机制的基础，
    通过计算Query和Key的相似度来确定对Value的加权，实现序列中不同位置间的信息交互。
    
    【核心原理】
    缩放点积注意力的数学公式：
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    
    其中：
    - Q (Query): 查询矩阵，表示"我想要什么信息"
    - K (Key): 键矩阵，表示"我有什么信息"
    - V (Value): 值矩阵，表示"具体的信息内容"
    - √d_k: 缩放因子，防止softmax饱和
    
    【技术特点】
    1. 缩放机制：使用√d_k缩放避免梯度消失
    2. 并行计算：所有位置的注意力可以并行计算
    3. 掩码支持：支持掩码机制处理变长序列
    4. Dropout正则化：防止注意力权重过拟合
    
    【计算流程】
    1. 计算注意力分数：QK^T
    2. 缩放：除以√d_k
    3. 应用掩码：将无效位置设为负无穷
    4. Softmax归一化：得到注意力权重
    5. 应用Dropout：防止过拟合
    6. 加权求和：注意力权重乘以Value
    
    【应用场景】
    1. 自注意力：Q、K、V来自同一序列
    2. 交叉注意力：Q来自目标序列，K、V来自源序列
    3. 因果注意力：使用掩码防止未来信息泄露
    4. 全局注意力：每个位置都能关注所有位置
    
    在MPT框架中的定位：
    - 注意力计算核心：为多头注意力提供基础计算
    - 依赖关系建模：捕获序列中的长距离依赖
    - 信息聚合器：根据相关性聚合信息
    """

    def __init__(self, temperature, attn_dropout=0.1):
        """
        初始化缩放点积注意力模块
        
        【核心功能】
        设置注意力计算的缩放参数和dropout正则化。
        
        【设计原理】
        1. 温度参数（缩放因子）：防止softmax函数进入饱和区域
        2. Dropout正则化：防止注意力权重过拟合
        3. 简洁设计：只包含必要的参数和组件
        
        Args:
            temperature (float): 温度参数，通常为√d_k
                作用：缩放注意力分数，防止softmax饱和
                计算：通常设为Key向量维度的平方根
                影响：较大的值使注意力分布更平滑，较小的值使注意力更集中
                示例：当d_k=64时，temperature=8.0
            attn_dropout (float): 注意力权重的dropout概率，默认0.1
                范围：0.0-0.5，通常0.1-0.3
                作用：防止注意力权重过拟合，提升泛化能力
                时机：在softmax之后、加权求和之前应用
                调优：可根据数据量和模型复杂度调整
        
        【参数设置原理】
        1. 温度参数的重要性：
           - 没有缩放：QK^T的值可能很大，导致softmax饱和
           - 饱和问题：softmax输出接近one-hot，梯度接近0
           - 缩放效果：保持softmax在敏感区域，梯度流动良好
        
        2. Dropout的作用：
           - 训练时：随机将部分注意力权重置0
           - 效果：防止模型过度依赖特定的注意力模式
           - 泛化：提升模型对不同输入的适应能力
        """
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.temperature = temperature  # 保存温度参数：用于缩放注意力分数，通常为√d_k
        self.dropout = nn.Dropout(attn_dropout)  # 初始化Dropout层：对注意力权重进行正则化

    def forward(self, q, k, v, mask=None):
        """
        缩放点积注意力的前向传播
        
        【核心功能】
        执行完整的缩放点积注意力计算，包括分数计算、缩放、掩码、
        softmax归一化、dropout和加权求和。
        
        【计算步骤】
        1. 计算原始注意力分数：Q与K的点积
        2. 温度缩放：除以√d_k防止softmax饱和
        3. 应用掩码：将无效位置设为负无穷
        4. Softmax归一化：转换为概率分布
        5. Dropout正则化：防止过拟合
        6. 加权求和：用注意力权重对Value加权
        
        Args:
            q (torch.Tensor): Query张量
                形状：(batch_size, n_head, len_q, d_k)
                内容：查询向量，表示"想要什么信息"
                来源：通常由输入序列经过线性变换得到
            k (torch.Tensor): Key张量
                形状：(batch_size, n_head, len_k, d_k)
                内容：键向量，表示"有什么信息"
                约束：d_k维度必须与Query一致
            v (torch.Tensor): Value张量
                形状：(batch_size, n_head, len_v, d_v)
                内容：值向量，表示"具体的信息内容"
                注意：len_v通常等于len_k
            mask (torch.Tensor, optional): 注意力掩码
                形状：(batch_size, n_head, len_q, len_k)
                内容：0表示无效位置，1表示有效位置
                用途：处理变长序列、因果掩码等
        
        Returns:
            torch.Tensor: 注意力输出
                形状：(batch_size, n_head, len_q, d_v)
                内容：经过注意力加权的特征表示
                含义：每个Query位置聚合的相关信息
        
        【实现细节】
        1. 注意力分数计算：
           - 使用torch.matmul进行批量矩阵乘法
           - 同时除以temperature进行缩放
           - 利用广播机制处理多头和批次维度
        
        2. 掩码处理：
           - 将掩码为0的位置设为-1e9（负无穷的近似）
           - 确保这些位置在softmax后权重接近0
           - 使用masked_fill进行高效的条件填充
        
        3. Softmax和Dropout：
           - 在最后一个维度（len_k）上应用softmax
           - 得到每个Query对所有Key的注意力权重
           - 应用dropout防止过拟合
        
        4. 加权求和：
           - 用注意力权重对Value进行加权平均
           - 得到每个Query位置的聚合信息
        
        【数学解释】
        对于每个Query位置i和Key位置j：
        1. 相似度：score_ij = q_i · k_j / √d_k
        2. 注意力权重：α_ij = softmax(score_ij)
        3. 输出：output_i = Σ_j α_ij * v_j
        
        【性能优化】
        - 使用矩阵运算而非循环，充分利用GPU并行计算
        - 广播机制处理多维张量，避免显式的维度扩展
        - 内存高效的掩码操作，避免创建大型中间张量
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # 计算缩放注意力分数：Q与K转置的矩阵乘法，同时除以温度参数进行缩放

        if mask is not None:  # 检查是否需要应用掩码
            attn = attn.masked_fill(mask == 0, -1e9)  # 应用掩码：将无效位置（mask=0）的注意力分数设为负无穷，确保softmax后权重接近0

        attn = self.dropout(F.softmax(attn, dim=-1))  # Softmax归一化和Dropout：在最后一维上应用softmax得到注意力权重，然后应用dropout防止过拟合
        output = torch.matmul(attn, v)  # 加权求和：用注意力权重对Value进行加权平均，得到最终的注意力输出

        return output  # 返回注意力输出

        
class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention - 多头注意力机制
    
    【系统概述】
    多头注意力是Transformer的核心组件，通过并行运行多个注意力头来捕获不同类型的依赖关系。
    每个头关注输入的不同方面，最后将所有头的输出拼接并投影，形成丰富的表示。
    
    【核心思想】
    "多头"设计的动机：
    1. 不同的头可以关注不同类型的依赖关系
    2. 增加模型的表达能力而不显著增加参数量
    3. 提供更稳定的训练过程
    4. 允许模型同时关注不同位置的信息
    
    【技术架构】
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    【关键组件】
    1. 线性投影层：将输入投影到多个子空间
    2. 缩放点积注意力：每个头的核心计算
    3. 输出投影：将多头结果融合
    4. 残差连接：保证训练稳定性
    5. 层归一化：加速收敛
    
    【设计优势】
    1. 并行计算：所有头可以并行处理
    2. 参数效率：总参数量与单头相当
    3. 表达丰富：不同头捕获不同模式
    4. 训练稳定：残差连接和层归一化
    
    【应用模式】
    1. 自注意力：编码器中，Q=K=V=输入序列
    2. 交叉注意力：解码器中，Q=解码器状态，K=V=编码器输出
    3. 掩码注意力：解码器中，防止未来信息泄露
    
    在MPT框架中的定位：
    - 特征交互核心：实现序列内部和跨序列的信息交互
    - 依赖建模器：捕获长距离和复杂的依赖关系
    - 表示学习器：学习丰富的序列表示
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """
        初始化多头注意力模块
        
        【核心功能】
        构建包含多个注意力头的完整多头注意力机制，包括输入投影、
        注意力计算和输出投影的全部组件。
        
        【设计原理】
        1. 维度分解：将大的注意力空间分解为多个小的子空间
        2. 并行处理：多个头可以同时计算，提高效率
        3. 信息融合：通过输出投影层融合多头信息
        4. 残差连接：保证深层网络的训练稳定性
        
        Args:
            n_head (int): 注意力头的数量
                范围：通常为4, 8, 16等
                作用：决定模型能够关注的不同依赖类型数量
                约束：d_model应该能被n_head整除
                推荐：根据模型大小选择，大模型用更多头
            d_model (int): 输入/输出的特征维度
                作用：决定模型的整体特征维度
                约束：必须能被n_head整除
                范围：通常为256, 512, 1024等
            d_k (int): 每个Key向量的维度
                计算：通常等于d_model // n_head
                作用：决定每个头的Key空间大小
                影响：影响注意力计算的精度和复杂度
            d_v (int): 每个Value向量的维度
                设置：通常与d_k相等
                作用：决定每个头的Value空间大小
                灵活性：可以与d_k不同以调节模型容量
            dropout (float): Dropout概率，默认0.1
                作用：防止过拟合，提升泛化能力
                应用：在输出投影后应用
                调优：可根据任务特点调整
        
        【模块构建】
        1. 输入投影层：
           - w_qs: Query投影，d_model -> n_head * d_k
           - w_ks: Key投影，d_model -> n_head * d_k  
           - w_vs: Value投影，d_model -> n_head * d_v
           - 无偏置：简化模型，通常效果更好
        
        2. 注意力计算：
           - 使用ScaledDotProductAttention
           - 温度参数设为√d_k
           - 支持掩码机制
        
        3. 输出处理：
           - fc: 输出投影，n_head * d_v -> d_model
           - dropout: 正则化
           - layer_norm: 层归一化
        
        【参数量分析】
        总参数量 ≈ 4 * d_model^2 (忽略偏置)
        - Query投影: d_model * (n_head * d_k) = d_model^2
        - Key投影: d_model * (n_head * d_k) = d_model^2
        - Value投影: d_model * (n_head * d_v) = d_model^2
        - 输出投影: (n_head * d_v) * d_model = d_model^2
        
        【内存优化】
        - 使用view操作而非创建新张量
        - 合理的维度变换顺序
        - 避免不必要的中间结果存储
        """
        super().__init__()  # 调用父类nn.Module的初始化方法

        self.n_head = n_head  # 保存注意力头数量
        self.d_k = d_k  # 保存Key向量维度
        self.d_v = d_v  # 保存Value向量维度

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)  # Query投影层：将输入投影到n_head个d_k维的Query子空间
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)  # Key投影层：将输入投影到n_head个d_k维的Key子空间
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)  # Value投影层：将输入投影到n_head个d_v维的Value子空间
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)  # 输出投影层：将拼接的多头输出投影回d_model维度

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)  # 初始化缩放点积注意力：温度参数设为√d_k

        self.dropout = nn.Dropout(dropout)  # Dropout层：对输出进行正则化
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化：标准化输出分布，eps防止除零错误


    def forward(self, q, k, v, mask=None):
        """
        多头注意力的前向传播
        
        【核心功能】
        执行完整的多头注意力计算，包括输入投影、多头注意力计算、
        输出融合、残差连接和层归一化。
        
        【计算流程】
        1. 输入投影：将Q、K、V投影到多个子空间
        2. 维度重排：调整张量形状以支持多头并行计算
        3. 注意力计算：对每个头执行缩放点积注意力
        4. 输出融合：拼接多头结果并投影
        5. 残差连接：加上输入实现残差连接
        6. 层归一化：标准化最终输出
        
        Args:
            q (torch.Tensor): Query输入张量
                形状：(batch_size, len_q, d_model)
                内容：查询序列的特征表示
                来源：可能是输入embedding或上一层输出
            k (torch.Tensor): Key输入张量
                形状：(batch_size, len_k, d_model)
                内容：键序列的特征表示
                关系：在自注意力中与q相同，在交叉注意力中不同
            v (torch.Tensor): Value输入张量
                形状：(batch_size, len_v, d_model)
                内容：值序列的特征表示
                约束：len_v通常等于len_k
            mask (torch.Tensor, optional): 注意力掩码
                形状：(batch_size, len_q, len_k)
                作用：控制哪些位置可以被关注
                用途：处理变长序列、因果掩码等
        
        Returns:
            torch.Tensor: 多头注意力输出
                形状：(batch_size, len_q, d_model)
                内容：融合了多头注意力信息的特征表示
                特性：包含了丰富的依赖关系信息
        
        【实现细节】
        1. 输入投影和重塑：
           - 线性投影：d_model -> n_head * d_k/d_v
           - 视图重塑：分离头维度
           - 维度转置：调整为(batch, n_head, seq_len, d_k/d_v)
        
        2. 掩码处理：
           - 添加头维度：unsqueeze(1)用于广播
           - 确保掩码形状与注意力分数匹配
        
        3. 注意力计算：
           - 对所有头并行计算注意力
           - 每个头在独立的子空间中操作
           - 利用GPU的并行计算能力
        
        4. 输出处理：
           - 转置回原始维度顺序
           - 使用contiguous()确保内存连续性
           - 重塑为(batch, len_q, n_head * d_v)
           - 通过输出投影层融合多头信息
        
        5. 残差连接和归一化：
           - 加上原始Query输入（残差连接）
           - 应用层归一化稳定训练
        
        【多头机制的优势】
        1. 表达能力：不同头关注不同类型的依赖
        2. 并行计算：所有头可以同时计算
        3. 参数效率：总参数量与单头相当
        4. 训练稳定：多头提供更稳定的梯度
        
        【内存和计算优化】
        - 使用view和transpose进行高效的维度操作
        - 避免创建不必要的中间张量
        - 利用PyTorch的自动优化机制
        - contiguous()确保内存布局优化
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head  # 获取维度参数：Key维度、Value维度、头数
        sz_b_q = q.size(0)  # 获取Query的批次大小
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)  # 获取各张量的尺寸信息

        residual = q  # 保存残差连接的输入：用于后续的残差连接

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b_q, len_q, n_head, d_k)  # Query投影和重塑：线性投影后重塑为(batch, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)  # Key投影和重塑：线性投影后重塑为(batch, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)  # Value投影和重塑：线性投影后重塑为(batch, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # 维度转置：将头维度移到第二位，便于并行计算注意力

        if mask is not None:  # 检查是否需要处理掩码
            mask = mask.unsqueeze(1)   # For head axis broadcasting. # 添加头维度：在第1维插入维度，支持多头广播

        q = self.attention(q, k, v, mask=mask)  # 执行缩放点积注意力：对所有头并行计算注意力

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # 维度恢复和拼接：转置回原始顺序，然后将多头结果拼接
        q = self.dropout(self.fc(q))  # 输出投影和Dropout：通过线性层融合多头信息，然后应用dropout
        q += residual  # 残差连接：加上原始输入，保证梯度流动

        q = self.layer_norm(q)  # 层归一化：标准化输出分布，稳定训练过程

        return q  # 返回多头注意力的最终输出


class PositionwiseFeedForward(nn.Module):
    """
    PositionwiseFeedForward - 位置前馈网络
    
    【系统概述】
    位置前馈网络是Transformer中的重要组件，对序列的每个位置独立应用相同的前馈网络。
    它提供了非线性变换能力，增强了模型的表达能力，是注意力机制的重要补充。
    
    【核心特点】
    1. 位置独立：对每个位置独立应用相同的变换
    2. 非线性变换：通过ReLU激活函数引入非线性
    3. 维度变换：先升维再降维的"瓶颈"结构
    4. 残差连接：保证深层网络的训练稳定性
    5. 层归一化：加速训练收敛
    
    【网络结构】
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    其中：
    - W_1: 第一层权重矩阵 (d_model -> d_hid)
    - W_2: 第二层权重矩阵 (d_hid -> d_model)
    - max(0, ·): ReLU激活函数
    
    【设计原理】
    1. 升维-降维结构：
       - 先将特征维度从d_model升到d_hid
       - 在高维空间进行非线性变换
       - 再降维回d_model保持输入输出一致
    
    2. 位置独立性：
       - 每个位置使用相同的参数
       - 不同位置间没有直接交互
       - 专注于单个位置的特征变换
    
    3. 非线性能力：
       - ReLU激活函数提供非线性
       - 增强模型的表达能力
       - 补充注意力机制的线性特性
    
    【技术优势】
    1. 计算高效：位置独立，易于并行化
    2. 参数共享：所有位置共享参数，减少过拟合
    3. 表达丰富：高维中间层提供强大的变换能力
    4. 训练稳定：残差连接和层归一化保证稳定性
    
    【应用场景】
    1. Transformer编码器：每个编码器层的第二个子层
    2. Transformer解码器：每个解码器层的最后一个子层
    3. 特征变换：对序列特征进行非线性变换
    4. 维度调整：在不同维度间进行特征映射
    
    在MPT框架中的定位：
    - 非线性变换器：为线性的注意力机制提供非线性能力
    - 特征增强器：通过升维-降维结构增强特征表达
    - 位置处理器：对每个序列位置进行独立的特征变换
    """

    def __init__(self, d_in, d_hid, dropout=0.1):
        """
        初始化位置前馈网络
        
        【核心功能】
        构建包含两层全连接网络的位置前馈模块，实现升维-降维的特征变换。
        
        【设计原理】
        1. 升维变换：将输入特征从d_in升到d_hid，增加表达空间
        2. 非线性激活：使用ReLU引入非线性变换能力
        3. 降维变换：将特征从d_hid降回d_in，保持输入输出一致
        4. 残差连接：保证深层网络的训练稳定性
        5. 层归一化：标准化输出分布，加速收敛
        
        Args:
            d_in (int): 输入/输出的特征维度
                作用：决定网络的输入和输出维度
                约束：必须与上下层的特征维度一致
                范围：通常为256, 512, 1024等
                示例：在d_model=512的Transformer中，d_in=512
            d_hid (int): 隐藏层的特征维度
                设置：通常为d_in的2-4倍
                作用：决定中间层的表达能力
                影响：更大的d_hid提供更强的非线性变换能力
                推荐：d_hid = 4 * d_in是常见设置
                示例：当d_in=512时，d_hid通常设为2048
            dropout (float): Dropout概率，默认0.1
                范围：0.0-0.5，通常0.1-0.3
                作用：防止过拟合，提升泛化能力
                应用：在第二层线性变换后应用
                调优：可根据数据量和模型复杂度调整
        
        【模块构建】
        1. 第一层线性变换 (w_1)：
           - 功能：d_in -> d_hid的升维变换
           - 作用：将特征投影到高维空间
           - 激活：后接ReLU激活函数
        
        2. 第二层线性变换 (w_2)：
           - 功能：d_hid -> d_in的降维变换
           - 作用：将高维特征投影回原始维度
           - 输出：直接输出，无激活函数
        
        3. 层归一化 (layer_norm)：
           - 功能：标准化输出分布
           - 参数：eps=1e-6防止除零错误
           - 作用：稳定训练，加速收敛
        
        4. Dropout层：
           - 功能：随机置零部分神经元
           - 时机：在第二层线性变换后应用
           - 作用：防止过拟合
        
        【参数量分析】
        总参数量 ≈ d_in * d_hid + d_hid * d_in = 2 * d_in * d_hid
        - 第一层：d_in * d_hid + d_hid (权重 + 偏置)
        - 第二层：d_hid * d_in + d_in (权重 + 偏置)
        - 层归一化：2 * d_in (缩放 + 偏移参数)
        
        【内存优化】
        - 使用标准的nn.Linear层，PyTorch自动优化
        - 合理的dropout设置，平衡正则化和性能
        - 层归一化参数量相对较小
        """
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise  # 第一层线性变换：升维变换，将d_in维特征映射到d_hid维
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise  # 第二层线性变换：降维变换，将d_hid维特征映射回d_in维
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)  # 层归一化：标准化输出分布，eps防止除零错误
        self.dropout = nn.Dropout(dropout)  # Dropout层：随机置零部分神经元，防止过拟合

    def forward(self, x):
        """
        位置前馈网络的前向传播
        
        【核心功能】
        执行完整的位置前馈网络计算，包括升维变换、非线性激活、
        降维变换、dropout、残差连接和层归一化。
        
        【计算流程】
        1. 保存残差：保存输入用于残差连接
        2. 升维变换：通过第一层线性变换升维
        3. 非线性激活：应用ReLU激活函数
        4. 降维变换：通过第二层线性变换降维
        5. Dropout正则化：随机置零部分神经元
        6. 残差连接：加上原始输入
        7. 层归一化：标准化最终输出
        
        Args:
            x (torch.Tensor): 输入张量
                形状：(batch_size, seq_len, d_in)
                内容：序列的特征表示
                来源：通常来自注意力层的输出
                约束：最后一维必须等于d_in
        
        Returns:
            torch.Tensor: 前馈网络输出
                形状：与输入相同 (batch_size, seq_len, d_in)
                内容：经过非线性变换的特征表示
                特性：包含了增强的特征表达能力
        
        【实现细节】
        1. 残差连接：
           - 保存原始输入作为残差
           - 在网络输出后加上残差
           - 保证梯度流动和训练稳定性
        
        2. 升维-激活-降维：
           - 第一层：x -> ReLU(xW_1 + b_1)
           - 第二层：ReLU_output -> ReLU_output * W_2 + b_2
           - 形成完整的非线性变换
        
        3. 正则化：
           - Dropout：在降维后应用，防止过拟合
           - 层归一化：在残差连接后应用，稳定训练
        
        【数学表达】
        FFN(x) = LayerNorm(x + Dropout(W_2 * ReLU(W_1 * x + b_1) + b_2))
        
        【位置独立性】
        - 每个序列位置使用相同的参数
        - 不同位置间没有信息交互
        - 可以高效并行计算
        
        【非线性变换能力】
        - ReLU激活函数引入非线性
        - 高维中间层提供丰富的变换空间
        - 补充注意力机制的线性特性
        
        【训练稳定性】
        - 残差连接防止梯度消失
        - 层归一化稳定激活分布
        - Dropout防止过拟合
        """
        residual = x  # 保存残差连接的输入：用于后续的残差连接，保证梯度流动

        x = self.w_2(F.relu(self.w_1(x)))  # 前馈网络变换：先升维并应用ReLU激活，再降维回原始维度
        x = self.dropout(x)  # 应用Dropout：随机置零部分神经元，防止过拟合
        x += residual  # 残差连接：加上原始输入，保证深层网络的训练稳定性

        x = self.layer_norm(x)  # 层归一化：标准化输出分布，加速训练收敛并稳定训练过程

        return x  # 返回前馈网络的最终输出
