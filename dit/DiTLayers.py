"""
DiT (Diffusion Transformer) 层实现
改编自 https://github.com/facebookresearch/DiT
"""

import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Attention, Mlp

def modulate(x, shift, scale):
    """应用仿射变换：scale * x + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    DiT Block with adaptive layer norm conditioning
    
    Args:
        hidden_size: Transformer隐藏层维度 (对应你的d_model)
        num_heads: 多头注意力头数
        mlp_ratio: MLP隐藏层维度相对于hidden_size的倍数
        dropout: Dropout概率
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, 
                       act_layer=nn.GELU, drop=dropout)
        
        # adaLN modulation - 6个参数用于两个层的shift, scale, gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        # 关键！AdaLN-Zero 初始化（DiT原论文）
        # 将adaLN最后一层的权重和偏置初始化为0
        # 这样训练初期，DiT block 是恒等映射（gate=0导致残差连接被bypass）
        # 网络可以从简单到复杂逐步学习，避免梯度消失和训练不稳定
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        """
        Args:
            x: 输入特征 (B, N, D) - N是序列长度(如12x12=144), D是特征维度
            c: 条件嵌入 (B, D) - 时间步或其他条件的嵌入
        Returns:
            输出特征 (B, N, D)
        """
        # 生成6个调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention with modulation
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        
        # FFN with modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        
        return x


class TimestepEmbedder(nn.Module):
    """
    时间步嵌入模块（用正弦位置编码）
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        创建正弦时间步嵌入
        Args:
            t: (N,) 时间步张量
            dim: 输出维度
        """
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        Args:
            t: (N,) 时间步
        Returns:
            (N, hidden_size) 时间步嵌入
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb