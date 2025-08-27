"""
Layers.py - Transformer层定义模块

【核心功能】
本模块定义了Transformer架构中的核心层组件，包括编码器层和解码器层。
这些层是构建完整Transformer模型的基础构建块。

【技术特点】
1. 模块化设计：每个层都是独立的nn.Module，便于组合和复用
2. 多头注意力：集成多头自注意力和交叉注意力机制
3. 残差连接：每个子层都包含残差连接，保证梯度流动
4. 位置前馈网络：提供非线性变换能力
5. 梯度检查点：支持内存优化的梯度计算

【架构设计】
- EncoderLayer：编码器层，包含自注意力和前馈网络
- DecoderLayer：解码器层，包含交叉注意力和前馈网络
- 每层都遵循"注意力->前馈网络"的标准Transformer结构

技术栈：
- PyTorch 深度学习框架
- 自定义SubLayers模块（MultiHeadAttention, PositionwiseFeedForward）
- 梯度检查点技术（torch.utils.checkpoint）

使用场景：
- 构建Transformer编码器和解码器
- 序列到序列建模
- 注意力机制的层级组合
- 深度神经网络的模块化设计

参考文献：
- Attention Is All You Need (Vaswani et al., 2017)
- https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""

import torch.nn as nn
import torch
import torch.utils.checkpoint
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward, CrossAttention

class EncoderLayer(nn.Module):
    """
    EncoderLayer - Transformer编码器层
    
    【系统概述】
    单个Transformer编码器层，由多头自注意力层和位置前馈网络层组成。
    这是构建深层Transformer编码器的基本单元，通过堆叠多个编码器层
    可以构建具有强大表达能力的序列编码模型。
    
    【核心组件】
    1. 多头自注意力(MultiHeadAttention)：捕获序列内部的依赖关系
    2. 位置前馈网络(PositionwiseFeedForward)：提供非线性变换
    3. 残差连接：每个子层都包含残差连接和层归一化
    
    【技术特点】
    - 自注意力机制：每个位置都能关注到序列中的所有位置
    - 并行计算：相比RNN，支持高效的并行训练
    - 梯度检查点：使用checkpoint技术节省内存
    - 可堆叠性：多个编码器层可以堆叠形成深层网络
    
    【应用场景】
    1. 序列编码：将输入序列转换为高维表示
    2. 特征提取：从序列数据中提取语义特征
    3. 表示学习：学习序列的分布式表示
    4. 预训练模型：作为大型预训练模型的组件
    
    【数学原理】
    编码器层的计算流程：
    1. 自注意力：Attention(Q,K,V) = softmax(QK^T/√d_k)V
    2. 残差连接：LayerNorm(x + Attention(x))
    3. 前馈网络：FFN(x) = max(0, xW1 + b1)W2 + b2
    4. 残差连接：LayerNorm(x + FFN(x))
    
    在MPT框架中的定位：
    - 序列编码器：将地图patch序列编码为高维特征
    - 特征提取器：提取空间位置间的依赖关系
    - 表示学习器：学习路径规划相关的语义表示
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        """
        初始化编码器层
        
        【核心功能】
        构建包含多头自注意力和位置前馈网络的完整编码器层。
        
        【设计原则】
        1. 模块化组合：将复杂的编码器层分解为独立的子模块
        2. 参数共享：在同一层内，注意力头之间共享参数结构
        3. 可配置性：支持不同的模型尺寸和复杂度配置
        4. 标准化接口：遵循PyTorch nn.Module的标准接口
        
        Args:
            d_model (int): 输入/输出的特征维度
                范围：通常为128, 256, 512, 1024等2的幂次
                作用：决定模型的整体表达能力
                约束：必须能被n_head整除
            d_inner (int): 位置前馈网络隐藏层的维度
                设置：通常为d_model的2-4倍
                作用：控制前馈网络的表达能力
                示例：d_model=512时，d_inner通常为2048
            n_head (int): 自注意力模块的头数
                范围：通常为4, 8, 16等
                作用：不同的头关注不同类型的依赖关系
                约束：d_model必须能被n_head整除
            d_k (int): 每个Key向量的维度
                计算：通常等于d_model // n_head
                作用：决定注意力计算的精度
                影响：影响模型的表达能力和计算复杂度
            d_v (int): 每个Value向量的维度
                设置：通常与d_k相等
                用途：决定注意力输出的特征维度
                优化：可以与d_k不同以调节模型容量
            dropout (float): Dropout概率，默认0.1
                范围：0.0-0.5，通常0.1-0.3
                作用：防止过拟合，提升泛化能力
                调优：可根据数据量和模型复杂度调整
        
        【模块构建】
        1. 多头自注意力模块：
           - 功能：捕获序列内部的长距离依赖关系
           - 参数：n_head个注意力头，每个头维度为d_k和d_v
           - 输出：加权后的特征表示
        
        2. 位置前馈网络：
           - 功能：提供非线性变换和特征融合
           - 结构：两层全连接网络，中间使用ReLU激活
           - 作用：增强模型的非线性表达能力
        
        【内存优化】
        - 使用相同的dropout概率保持一致性
        - 子模块独立初始化，便于参数管理
        - 支持梯度检查点技术节省内存
        """
        super(EncoderLayer, self).__init__()  # 调用父类nn.Module的初始化方法
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  # 初始化多头自注意力模块：用于捕获序列内部依赖关系
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)  # 初始化位置前馈网络：提供非线性变换能力

    def forward(self, enc_input, slf_attn_mask=None):
        """
        编码器层前向传播函数
        
        【核心功能】
        执行完整的编码器层计算，包括多头自注意力和位置前馈网络的处理。
        使用梯度检查点技术优化内存使用，适合深层网络训练。
        
        【计算流程】
        1. 多头自注意力计算：计算序列内部的注意力权重和加权特征
        2. 残差连接和层归一化：保证梯度流动和训练稳定性
        3. 位置前馈网络：应用非线性变换
        4. 再次残差连接和层归一化：最终输出处理
        
        Args:
            enc_input (torch.Tensor): 编码器输入张量
                形状：(batch_size, seq_len, d_model)
                内容：输入序列的特征表示
                来源：通常来自embedding层或上一个编码器层
            slf_attn_mask (torch.Tensor, optional): 自注意力掩码
                形状：(batch_size, seq_len, seq_len)
                作用：控制注意力计算中哪些位置可以被关注
                用途：处理变长序列或屏蔽特定位置
                当前：设为None表示不使用掩码
        
        Returns:
            torch.Tensor: 编码器层输出
                形状：与输入相同(batch_size, seq_len, d_model)
                内容：经过注意力和前馈网络处理的特征表示
                特性：包含了序列内部依赖关系的高级特征
        
        【技术实现】
        1. 梯度检查点技术：
           - 使用torch.utils.checkpoint.checkpoint
           - 在前向传播时不保存中间激活值
           - 反向传播时重新计算，节省内存
           - 适合深层网络和大批次训练
        
        2. 注意力计算：
           - Query, Key, Value都来自同一输入(自注意力)
           - 支持掩码机制处理特殊情况
           - 输出包含全局上下文信息
        
        3. 前馈网络处理：
           - 对每个位置独立应用相同的变换
           - 增加模型的非线性表达能力
           - 包含残差连接保证训练稳定性
        
        【内存优化策略】
        - 使用梯度检查点减少内存占用
        - 避免保存不必要的中间结果
        - 支持大模型和长序列的训练
        
        【注释说明】
        代码中包含了两种实现方式的注释：
        - 不使用梯度检查点的标准实现（已注释）
        - 使用梯度检查点的内存优化实现（当前使用）
        - 返回注意力权重的实现（已注释，可用于可视化）
        """
        # # Without gradient Checking
        # # 不使用梯度检查点的标准实现
        # enc_output = self.slf_attn(
        #     enc_input, enc_input, enc_input, mask=slf_attn_mask)

        # With Gradient Checking
        # 使用梯度检查点的内存优化实现
        enc_output = torch.utils.checkpoint.checkpoint(self.slf_attn,  # 使用梯度检查点技术：节省内存，在反向传播时重新计算前向结果
        enc_input, enc_input, enc_input, slf_attn_mask)  # 自注意力计算：Query、Key、Value都来自同一输入，实现序列内部的依赖建模

        # enc_output, enc_slf_attn = self.slf_attn(
        #     enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # 可选实现：同时返回注意力权重，用于可视化和分析（已注释）

        enc_output = self.pos_ffn(enc_output)  # 位置前馈网络：对注意力输出应用非线性变换，增强模型表达能力
        return enc_output  # 返回编码器层的最终输出
    
class PoseWiseEncoderLayer(nn.Module):
    """
    map 自注意力 + yaw-aware gating + yaw 稳定性分支
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, yaw_bins=36, dropout=0.1):
        super().__init__()
        
        # 1) 地图自注意力：保持上下文建模能力
        self.map_sa = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        
        # 2) 角度感知 gating (类似 attention bias)
        self.angle_proj = nn.Linear(d_model, yaw_bins, bias=False)  # 将 token 投影到角度分布
        self.gate_proj = nn.Linear(yaw_bins, d_model, bias=False)   # 将角度分布映射回调制向量
        self.norm = nn.LayerNorm(d_model)

        # 3) FFN
        self.ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)


    def forward(self, map_tokens, slf_attn_mask=None):
        """
        map_tokens: [B, N, D]    pose_tokens: [B, 2, D]
        返回:
          out: [B, N, D]  (融合后的地图特征)
          aux: dict(yaw_logits=[B, N, K], pose_ctx=[B, 2, D])
        """
        B, N, D = map_tokens.shape

        # 1) 地图自注意力
        map_feat = self.map_sa(map_tokens, map_tokens, map_tokens, mask=slf_attn_mask)  # [B,N,D]
        
        # 2) yaw-aware gating
        # 每个 token → soft angle 分布
        angle_logits = self.angle_proj(map_feat)  # [B, N, n_angle_bins]
        angle_weights = torch.softmax(angle_logits, dim=-1)  # 概率分布
        # 映射回调制向量
        gate = self.gate_proj(angle_weights)  # [B, N, D]
        fused_feat = self.norm(map_feat * (1 + torch.tanh(gate)))  # 残差调制 + LN

        # 3) FFN
        # out = self.ffn(fused_feat)  # [B,N,D]
        out = fused_feat  # [B,N,D]
        
        return out


class DecoderLayer(nn.Module):
    """
    DecoderLayer - Transformer解码器层
    
    【系统概述】
    Transformer解码器层，由三个主要组件组成：掩码自注意力、编码器-解码器注意力和位置前馈网络。
    在当前实现中，简化为只包含编码器-解码器注意力和位置前馈网络。
    
    【核心功能】
    1. 编码器-解码器注意力：利用编码器输出指导解码过程
    2. 位置前馈网络：提供非线性变换能力
    3. 残差连接：保证深层网络的训练稳定性
    
    【设计特点】
    - 交叉注意力：解码器可以关注编码器的所有位置
    - 信息融合：结合解码器状态和编码器上下文
    - 序列生成：支持自回归的序列生成过程
    - 模块化设计：便于与其他层组合使用
    
    【应用场景】
    1. 序列到序列生成：机器翻译、文本摘要等
    2. 条件生成：基于编码器上下文的条件生成
    3. 路径规划：基于地图编码生成路径序列
    4. 多模态融合：融合不同模态的信息
    
    【技术原理】
    解码器层的计算包含：
    1. 编码器-解码器注意力：Attention(Q_dec, K_enc, V_enc)
    2. 残差连接：LayerNorm(dec_input + Attention_output)
    3. 前馈网络：FFN(attention_output)
    4. 残差连接：LayerNorm(attention_output + FFN_output)
    
    【简化说明】
    当前实现省略了掩码自注意力部分，专注于编码器-解码器的信息交互。
    这种简化适合特定的应用场景，如基于全局上下文的单步预测。
    
    在MPT框架中的定位：
    - 路径生成器：基于地图编码生成路径点
    - 上下文融合器：融合局部和全局信息
    - 决策模块：在每个时间步做出路径选择
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        """
        初始化解码器层
        
        【核心功能】
        构建包含编码器-解码器注意力和位置前馈网络的解码器层。
        
        【设计考虑】
        1. 简化架构：省略掩码自注意力，专注于编码器-解码器交互
        2. 模块复用：使用与编码器相同的子模块设计
        3. 参数一致性：保持与编码器层相同的参数配置
        4. 灵活配置：支持不同的模型尺寸和复杂度
        
        Args:
            d_model (int): 输入/输出的特征维度
                作用：决定解码器层的整体特征维度
                约束：必须与编码器的d_model一致
                范围：通常为256, 512, 1024等
            d_inner (int): 位置前馈网络隐藏层维度
                设置：通常为d_model的2-4倍
                作用：控制前馈网络的表达能力
                影响：影响模型的非线性变换能力
            n_head (int): 多头注意力的头数
                作用：不同头关注编码器的不同方面
                约束：d_model必须能被n_head整除
                推荐：与编码器保持一致
            d_k (int): Key向量的维度
                计算：通常等于d_model // n_head
                作用：决定注意力计算的精度
                影响：影响编码器-解码器注意力的质量
            d_v (int): Value向量的维度
                设置：通常与d_k相等
                用途：决定注意力输出的特征维度
                优化：可以独立调节以优化性能
            dropout (float): Dropout概率，默认0.1
                作用：防止过拟合，提升泛化能力
                调优：可根据任务特点调整
                一致性：建议与编码器保持相同
        
        【模块构建】
        1. 编码器-解码器注意力：
           - 功能：让解码器关注编码器的输出
           - Query：来自解码器输入
           - Key和Value：来自编码器输出
           - 作用：实现跨序列的信息传递
        
        2. 位置前馈网络：
           - 功能：提供非线性变换
           - 结构：与编码器层相同
           - 作用：增强特征表达能力
        
        【架构简化说明】
        标准Transformer解码器层包含三个子层：
        1. 掩码自注意力（当前实现中省略）
        2. 编码器-解码器注意力（当前实现）
        3. 位置前馈网络（当前实现）
        
        当前简化版本专注于编码器-解码器的信息交互，
        适合不需要序列内部依赖建模的特定应用场景。
        """
        super(DecoderLayer, self).__init__()  # 调用父类nn.Module的初始化方法
        # self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # 掩码自注意力模块（已注释）：在当前简化实现中省略，标准解码器层会包含此模块
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)  # 编码器-解码器注意力模块：让解码器关注编码器的输出
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)  # 位置前馈网络：提供非线性变换能力

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        """
        解码器层前向传播函数
        
        【核心功能】
        执行解码器层的前向计算，包括编码器-解码器注意力和位置前馈网络处理。
        实现解码器状态与编码器上下文的信息融合。
        
        【计算流程】
        1. 编码器-解码器注意力：解码器查询编码器的信息
        2. 残差连接和层归一化：保证训练稳定性
        3. 位置前馈网络：应用非线性变换
        4. 残差连接和层归一化：最终输出处理
        
        Args:
            dec_input (torch.Tensor): 解码器输入张量
                形状：(batch_size, dec_seq_len, d_model)
                内容：解码器的输入特征，通常是目标序列的embedding
                来源：可能来自目标序列embedding或上一个解码器层
            enc_output (torch.Tensor): 编码器输出张量
                形状：(batch_size, enc_seq_len, d_model)
                内容：编码器的输出特征，包含源序列的完整信息
                作用：为解码器提供上下文信息
            slf_attn_mask (torch.Tensor, optional): 自注意力掩码
                形状：(batch_size, dec_seq_len, dec_seq_len)
                作用：控制解码器内部的注意力（当前实现中未使用）
                用途：防止未来信息泄露（因果掩码）
            dec_enc_attn_mask (torch.Tensor, optional): 解码器-编码器注意力掩码
                形状：(batch_size, dec_seq_len, enc_seq_len)
                作用：控制解码器对编码器的注意力
                用途：屏蔽编码器中的填充位置
        
        Returns:
            tuple: (dec_output, dec_enc_attn)
                dec_output (torch.Tensor): 解码器层输出
                    形状：(batch_size, dec_seq_len, d_model)
                    内容：融合了编码器信息的解码器特征
                    用途：传递给下一个解码器层或输出层
                dec_enc_attn (torch.Tensor): 编码器-解码器注意力权重
                    形状：(batch_size, n_head, dec_seq_len, enc_seq_len)
                    内容：解码器对编码器各位置的注意力分布
                    用途：可视化分析、调试、可解释性研究
        
        【技术实现】
        1. 编码器-解码器注意力：
           - Query：来自解码器输入
           - Key和Value：来自编码器输出
           - 实现跨序列的信息传递
           - 支持掩码机制处理特殊情况
        
        2. 位置前馈网络：
           - 对每个位置独立应用变换
           - 增加模型的非线性表达能力
           - 包含残差连接和层归一化
        
        【注意力机制详解】
        编码器-解码器注意力允许解码器的每个位置关注编码器的所有位置：
        - 这种机制使得解码器能够根据需要选择性地关注源序列的不同部分
        - 注意力权重反映了源序列和目标序列之间的对齐关系
        - 在机器翻译中，这对应于源语言和目标语言之间的词汇对齐
        
        【简化实现说明】
        当前实现省略了标准解码器层中的掩码自注意力部分：
        - 标准实现：自注意力 -> 编码器-解码器注意力 -> 前馈网络
        - 当前实现：编码器-解码器注意力 -> 前馈网络
        - 这种简化适合不需要序列内部依赖建模的特定场景
        """
        # dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        # 掩码自注意力计算（已注释）：标准解码器层的第一个子层，用于建模目标序列内部依赖
        dec_output, dec_enc_attn = self.enc_attn(dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)  # 编码器-解码器注意力：解码器查询编码器信息，实现跨序列信息传递
        dec_output = self.pos_ffn(dec_output)  # 位置前馈网络：对注意力输出应用非线性变换，增强特征表达能力
        return dec_output, dec_enc_attn  # 返回解码器输出和注意力权重：输出用于后续处理，注意力权重用于分析和可视化