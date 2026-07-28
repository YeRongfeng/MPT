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

# B样条工具
from bspline_utils import (
    fit_bspline_least_squares,
    reconstruct_from_control_points,
    DifferentiableBSpline
)
from map_config import MAP_HALF_EXTENT

import math


class PhysicalScaledEdgeResidualRepresentation(nn.Module):
    """25× physical edge residuals with an optional radial feasibility map."""

    def __init__(
        self,
        num_control_points: int = 26,
        lower_bound: float = -1.0,
        upper_bound: float = 1.0,
        feasibility_eps: float = 1e-6,
    ):
        super().__init__()
        if num_control_points != 26:
            raise ValueError("Current implementation expects 26 control points.")
        if not lower_bound < upper_bound:
            raise ValueError("lower_bound must be smaller than upper_bound.")
        self.num_control_points = num_control_points
        self.num_edges = num_control_points - 1
        self.residual_scale = float(self.num_edges)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.feasibility_eps = float(feasibility_eps)

    def project_zero_sum(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.num_edges, 2):
            raise ValueError(
                f"Expected (B,{self.num_edges},2), got {tuple(x.shape)}"
            )
        projected = x - x.mean(dim=1, keepdim=True)
        return projected - projected.mean(dim=1, keepdim=True)

    def encode(
        self,
        control_points: torch.Tensor,
        start_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ) -> torch.Tensor:
        if control_points.ndim != 3 or control_points.shape[1:] != (
            self.num_control_points,
            2,
        ):
            raise ValueError(
                f"Expected control_points (B,{self.num_control_points},2), "
                f"got {tuple(control_points.shape)}"
            )
        if start_xy.shape != (control_points.shape[0], 2):
            raise ValueError(f"Expected start_xy (B,2), got {tuple(start_xy.shape)}")
        if goal_xy.shape != (control_points.shape[0], 2):
            raise ValueError(f"Expected goal_xy (B,2), got {tuple(goal_xy.shape)}")

        edges = control_points[:, 1:, :] - control_points[:, :-1, :]
        base_edge = (goal_xy - start_xy).unsqueeze(1) / self.num_edges
        return self.project_zero_sum(
            self.residual_scale * (edges - base_edge)
        )

    def decode(
        self,
        scaled_residuals: torch.Tensor,
        start_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ) -> torch.Tensor:
        scaled_residuals = self.project_zero_sum(scaled_residuals)
        if start_xy.shape != (scaled_residuals.shape[0], 2):
            raise ValueError(f"Expected start_xy (B,2), got {tuple(start_xy.shape)}")
        if goal_xy.shape != (scaled_residuals.shape[0], 2):
            raise ValueError(f"Expected goal_xy (B,2), got {tuple(goal_xy.shape)}")

        base_edge = (goal_xy - start_xy).unsqueeze(1) / self.num_edges
        edges = base_edge + scaled_residuals / self.residual_scale
        points_after_start = start_xy.unsqueeze(1) + torch.cumsum(edges, dim=1)
        return torch.cat([start_xy.unsqueeze(1), points_after_start], dim=1)

    def chord(self, start_xy: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
        if start_xy.ndim != 2 or start_xy.shape[-1] != 2:
            raise ValueError(f"Expected start_xy (B,2), got {tuple(start_xy.shape)}")
        if goal_xy.shape != start_xy.shape:
            raise ValueError(
                f"Expected goal_xy shaped {tuple(start_xy.shape)}, got "
                f"{tuple(goal_xy.shape)}"
            )
        tau = torch.linspace(
            0.0,
            1.0,
            self.num_control_points,
            device=start_xy.device,
            dtype=start_xy.dtype,
        ).view(1, self.num_control_points, 1)
        return (1.0 - tau) * start_xy.unsqueeze(1) + tau * goal_xy.unsqueeze(1)

    def compute_max_feasible_radius(
        self,
        direction: torch.Tensor,
        start_xy: torch.Tensor,
        goal_xy: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        """Maximum positive radius along a unit zero-sum residual direction."""
        if direction.ndim != 3 or direction.shape[1:] != (self.num_edges, 2):
            raise ValueError(
                f"Expected direction (B,{self.num_edges},2), got "
                f"{tuple(direction.shape)}"
            )
        chord = self.chord(start_xy, goal_xy)
        delta_after_start = torch.cumsum(direction, dim=1) / self.residual_scale
        delta = torch.cat(
            [torch.zeros_like(delta_after_start[:, :1]), delta_after_start],
            dim=1,
        )

        eps = self.feasibility_eps
        upper_slack = (self.upper_bound - chord).clamp_min(0.0)
        lower_slack = (chord - self.lower_bound).clamp_min(0.0)
        # Use a large finite sentinel instead of inf: the zero-direction branch
        # is masked later, and finite values avoid inf*0/NaN in backward.
        sentinel = torch.full_like(delta, 1.0 / eps)

        positive = delta > eps
        negative = delta < -eps
        safe_positive_delta = torch.where(positive, delta, torch.ones_like(delta))
        safe_negative_delta = torch.where(negative, -delta, torch.ones_like(delta))
        upper_limit = torch.where(
            positive,
            upper_slack / safe_positive_delta,
            sentinel,
        )
        lower_limit = torch.where(
            negative,
            lower_slack / safe_negative_delta,
            sentinel,
        )
        coordinate_limit = torch.minimum(upper_limit, lower_limit)
        flat_limit = coordinate_limit.flatten(start_dim=1)
        rho, active_flat_index = flat_limit.min(dim=1)
        rho = rho.clamp_min(0.0)

        if not return_diagnostics:
            return rho

        active_is_upper = (
            upper_limit.flatten(start_dim=1)
            .gather(1, active_flat_index.unsqueeze(1))
            .squeeze(1)
            <= lower_limit.flatten(start_dim=1)
            .gather(1, active_flat_index.unsqueeze(1))
            .squeeze(1)
        )
        diagnostics = {
            'active_flat_index': active_flat_index,
            'active_control_point': torch.div(
                active_flat_index, 2, rounding_mode='floor'
            ),
            'active_coordinate': active_flat_index.remainder(2),
            'active_is_upper': active_is_upper,
            'chord': chord,
            'delta': delta,
        }
        return rho, diagnostics

    def radial_feasible_residual(
        self,
        raw_v: torch.Tensor,
        raw_s: torch.Tensor,
        start_xy: torch.Tensor,
        goal_xy: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        """Map unconstrained network outputs to a feasible physical residual."""
        v = self.project_zero_sum(raw_v)
        batch_size = v.shape[0]
        if raw_s.ndim == 1:
            raw_s = raw_s.unsqueeze(1)
        if raw_s.shape != (batch_size, 1):
            raise ValueError(f"Expected raw_s (B,1), got {tuple(raw_s.shape)}")

        v_norm = torch.linalg.vector_norm(v.flatten(start_dim=1), dim=1)
        slack = F.softplus(raw_s.squeeze(1)) + 1e-4
        denominator = torch.sqrt(
            v_norm.square() + slack.square() + self.feasibility_eps
        )
        x = v / denominator.view(batch_size, 1, 1)
        x_norm = torch.linalg.vector_norm(x.flatten(start_dim=1), dim=1)
        nonzero = x_norm > self.feasibility_eps
        direction = x / x_norm.clamp_min(self.feasibility_eps).view(
            batch_size, 1, 1
        )
        rho, radius_diagnostics = self.compute_max_feasible_radius(
            direction,
            start_xy,
            goal_xy,
            return_diagnostics=True,
        )
        rho = torch.where(nonzero, rho, torch.zeros_like(rho))
        residual = rho.view(batch_size, 1, 1) * x
        residual = torch.where(
            nonzero.view(batch_size, 1, 1),
            residual,
            torch.zeros_like(residual),
        )

        if not return_diagnostics:
            return residual

        control_points = self.decode(residual, start_xy, goal_xy)
        hard_margin = torch.minimum(
            control_points - self.lower_bound,
            self.upper_bound - control_points,
        ).flatten(start_dim=1).amin(dim=1)
        diagnostics = {
            'x_norm': x_norm,
            'rho': rho,
            'slack': slack,
            'hard_boundary_margin': hard_margin,
            **radius_diagnostics,
        }
        return residual, diagnostics

    def radial_diagnostics_from_residual(
        self,
        residual: torch.Tensor,
        start_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ):
        """Recover radial utilization and active constraints from a residual."""
        residual = self.project_zero_sum(residual)
        residual_norm = torch.linalg.vector_norm(
            residual.flatten(start_dim=1), dim=1
        )
        nonzero = residual_norm > self.feasibility_eps
        direction = residual / residual_norm.clamp_min(
            self.feasibility_eps
        ).view(-1, 1, 1)
        rho, diagnostics = self.compute_max_feasible_radius(
            direction,
            start_xy,
            goal_xy,
            return_diagnostics=True,
        )
        rho = torch.where(nonzero, rho, torch.zeros_like(rho))
        x_norm = torch.where(
            nonzero,
            residual_norm / rho.clamp_min(self.feasibility_eps),
            torch.zeros_like(residual_norm),
        )
        control_points = self.decode(residual, start_xy, goal_xy)
        hard_margin = torch.minimum(
            control_points - self.lower_bound,
            self.upper_bound - control_points,
        ).flatten(start_dim=1).amin(dim=1)
        return {
            'x_norm': x_norm,
            'rho': rho,
            'hard_boundary_margin': hard_margin,
            **diagnostics,
        }

    def radial_project_residual(
        self,
        residual: torch.Tensor,
        start_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Project only excessive radial magnitude; preserve residual direction."""
        residual = self.project_zero_sum(residual)
        residual_norm = torch.linalg.vector_norm(
            residual.flatten(start_dim=1), dim=1
        )
        nonzero = residual_norm > self.feasibility_eps
        direction = residual / residual_norm.clamp_min(
            self.feasibility_eps
        ).view(-1, 1, 1)
        rho = self.compute_max_feasible_radius(direction, start_xy, goal_xy)
        scale = torch.minimum(
            torch.ones_like(residual_norm),
            rho / residual_norm.clamp_min(self.feasibility_eps),
        )
        scale = torch.where(nonzero, scale, torch.ones_like(scale))
        return residual * scale.view(-1, 1, 1)


class TrajDiTBlock(nn.Module):
    """
    路径规划专用DiT Block，支持Cross-Attention
    结构:
        Input (Path Tokens)
            |
        AdaLN (Time, Start, Goal) -> Norm
            |
        Self-Attention (路径点之间的平滑性/几何关系)
            |
        AdaLN -> Norm
            |
        Cross-Attention (路径点 查询 地图特征 -> 避障)
            |
        AdaLN -> Norm
            |
        Feed Forward
            |
        Output
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
            super().__init__()
            self.n_heads = n_heads
            self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.regime_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout)
            )
            self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(d_model, 9 * d_model, bias=True)
            )
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, regime_feat, c):
            params = self.adaLN_modulation(c).chunk(9, dim=-1)
            (shift_msa, scale_msa, gate_msa, 
                shift_mca, scale_mca, gate_mca, 
                shift_mlp, scale_mlp, gate_mlp) = params
            # 1. Self-Attention
            x_norm = self.norm1(x)
            x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
            x = x + gate_msa.unsqueeze(1) * attn_out
            # 2. Cross-Attention
            x_norm = self.norm2(x)
            x_norm = x_norm * (1 + scale_mca.unsqueeze(1)) + shift_mca.unsqueeze(1)
            attn_out, _ = self.regime_cross_attn(x_norm, regime_feat, regime_feat, need_weights=False)
            x = x + gate_mca.unsqueeze(1) * attn_out
            # 3. FFN
            x_norm = self.norm3(x)
            x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            ffn_out = self.ffn(x_norm)
            x = x + gate_mlp.unsqueeze(1) * ffn_out
            return x


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
        x_self, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
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


class MultiScalePositionalEncoding(nn.Module):
    """
    动态2D位置编码：支持任意(H, W)的特征图。
    用于多尺度KV场景（如25x25与12x12并存）。
    """
    def __init__(self, d_hid):
        super().__init__()
        self.d_hid = d_hid

    def _build_2d_sincos(self, h, w, device, dtype):
        # 2D sin-cos，优先按4路拆分；若d_hid非4倍数，尾部补零
        d_quarter = self.d_hid // 4
        d_used = d_quarter * 4

        y = torch.arange(h, device=device, dtype=dtype)
        x = torch.arange(w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        yy = yy.reshape(-1, 1)
        xx = xx.reshape(-1, 1)

        if d_quarter > 0:
            div_term = torch.exp(
                torch.arange(d_quarter, device=device, dtype=dtype) *
                (-math.log(10000.0) / max(d_quarter, 1))
            ).unsqueeze(0)  # (1, d_quarter)

            pos_x = xx * div_term
            pos_y = yy * div_term

            pe = torch.cat([
                torch.sin(pos_x), torch.cos(pos_x),
                torch.sin(pos_y), torch.cos(pos_y)
            ], dim=1)  # (H*W, 4*d_quarter)
        else:
            pe = torch.zeros((h * w, 0), device=device, dtype=dtype)

        if d_used < self.d_hid:
            pad = torch.zeros((h * w, self.d_hid - d_used), device=device, dtype=dtype)
            pe = torch.cat([pe, pad], dim=1)

        return pe.unsqueeze(0)  # (1, H*W, D)

    def forward(self, x, conv_shape):
        if conv_shape is None:
            raise ValueError("MultiScalePositionalEncoding requires conv_shape=(H,W).")
        h, w = int(conv_shape[0]), int(conv_shape[1])
        pos = self._build_2d_sincos(h, w, x.device, x.dtype)
        if x.shape[1] != h * w:
            raise ValueError(f"Token length mismatch: got {x.shape[1]}, expected {h*w} for conv_shape={conv_shape}.")
        return x + pos


class ImplicitGuidanceEncoder(nn.Module):
    """
    生成与轨迹进度对齐的隐式 guidance tokens。

    guidance queries 由 start/goal SE(2)、t/h 和可学习进度 token 构成，
    再由这些非逐点噪声 query 去读取 normal map tokens。
    """
    def __init__(self, d_model, n_heads, d_ff, n_path_steps=25, dropout=0.1, n_self_layers=1):
        super().__init__()
        self.n_path_steps = n_path_steps

        self.start_embedder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.goal_embedder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.task_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.time_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.progress_embed = nn.Parameter(
            torch.randn(1, n_path_steps, d_model) * 0.02
        )
        self.query_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * d_model, 2 * d_model)
        )

        self.map_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.query_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.map_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.guidance_self_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(n_self_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, map_tokens, start_pose, goal_pose, t_emb, h_emb):
        B = map_tokens.shape[0]

        start_token = self.start_embedder(start_pose)
        goal_token = self.goal_embedder(goal_pose)
        task_context = self.task_fusion(torch.cat([start_token, goal_token], dim=-1))
        time_context = self.time_fusion(torch.cat([t_emb, h_emb], dim=-1))

        context = torch.cat([task_context, time_context], dim=-1)
        shift, scale = self.query_film(context).chunk(2, dim=-1)

        queries = self.progress_embed.expand(B, -1, -1)
        queries = queries * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        q = self.query_norm(queries)
        kv = self.map_norm(map_tokens)
        guidance, _ = self.map_attn(q, kv, kv, need_weights=False)
        guidance = queries + guidance

        for layer in self.guidance_self_layers:
            guidance = layer(guidance)

        return self.output_norm(guidance)


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
    【控制点语义】
    - pMF状态：25个物理控制多边形边的25×缩放零和残差
    - 完整B样条：由起终点、均匀基准边和残差累加为26个控制点
    【架构设计】
    原始UnevenTransformer：地图 → CNN → Transformer blocks → 预测头
    改造为扩散模型：地图+噪声路径 → CNN特征提取 → token融合 → DiT blocks(时间步条件) → 噪声预测
    dit_blocks基于TrajDiTBlock实现，支持Cross-Attention融合地图特征
    【训练目标】
    - 损失：MSE(pred, true)
    - 预测空间：噪声ε、原始数据x0、velocity v
    依据prediction_type和loss_type设定
    当前主要以x0预测+v损失为主
    【采样】
    - ode采样(Euler或Heun)
    当前采样基于rectified flow方法实现，后续可拓展为meanflow，再进一步修改为pixelMeanFlow
    """
    
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, 
                 pad_idx, dropout, n_position, train_shape, 
                 n_path_steps=25, diffusion_steps=50, prediction_type='epsilon', loss_type=None,
                 coordinate_scale=MAP_HALF_EXTENT,
                 privileged_local_dim=0,
                 **deprecated_kwargs):
        """
        Args:
            n_path_steps: 物理边残差token数量，当前固定为25
            prediction_type: 'epsilon' (预测噪声), 'x0' (预测原始数据), 或 'v' (预测velocity)
            loss_type: 'epsilon', 'x0', 或 'v' - 损失函数的目标类型，如果为None则与prediction_type相同
            coordinate_scale: 归一化坐标1.0对应的物理米数
        """
        super().__init__()
        unexpected_kwargs = set(deprecated_kwargs) - {'num_regime_tokens'}
        if unexpected_kwargs:
            raise TypeError(f"Unexpected PathDiffusionTransformer kwargs: {sorted(unexpected_kwargs)}")
        if n_path_steps != 25:
            raise ValueError(
                "Physical edge residual representation requires n_path_steps=25."
            )
        
        self.prediction_type = prediction_type
        self.loss_type = loss_type if loss_type is not None else prediction_type
        self.coordinate_scale = float(coordinate_scale)
        self.privileged_local_dim = int(privileged_local_dim)
        if self.coordinate_scale <= 0.0:
            raise ValueError("coordinate_scale must be positive")
        if self.privileged_local_dim < 0:
            raise ValueError("privileged_local_dim must be non-negative")
        # 显存优化：训练时对DiT主干启用梯度检查点
        self.use_gradient_checkpoint = True
        
        # ========== 地图CNN特征提取（多尺度KV）==========
        self.map_fe_block1 = nn.Sequential(
            nn.Conv2d(3, d_model//8, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model//8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 50×50
        )

        self.map_fe_block2 = nn.Sequential(
            nn.Conv2d(d_model//8, d_model//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model//4),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25×25
        )

        self.map_fe_block3 = nn.Sequential(
            nn.Conv2d(d_model//4, d_model//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model//2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12×12
        )

        self.map_fe_block4 = nn.Sequential(
            nn.Conv2d(d_model//2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
        )

        # 25×25 分支通道对齐到 d_model，用于高分辨率KV
        self.map_proj_25 = nn.Sequential(
            nn.Conv2d(d_model//4, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
        )
        
        self.reorder_dims = Rearrange('b c h w -> b (h w) c')
        self.map_position_enc = MultiScalePositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        # ========== Implicit Guidance Encoder ==========
        # 用非逐点噪声的 progress/task/time/summary queries 读取 normal map，
        # 得到与25条控制多边形边对齐的隐式 guidance tokens。
        self.guidance_encoder = ImplicitGuidanceEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_inner,
            n_path_steps=n_path_steps,
            dropout=dropout,
            n_self_layers=1
        )

        # ========== 条件融合MLP ==========
        # 将 t、h=t-r、起点和终点融合为全局条件
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 5),  # 4 = time + time_h + start + goal
            nn.GELU(),
            nn.Linear(d_model * 5, d_model)
        )
        
        # ========== 起点终点编码（用于条件调制）==========
        # 将起点和终点编码为条件向量（不作为独立tokens）
        self.pose_embedder = nn.Sequential(
            nn.Linear(4, d_model),  # (x,y,cos(θ),sin(θ)) - 4维角度编码
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # ========== 路径Patchify ==========
        # 关键改进：输入只有2维(x,y)控制点，不包含theta
        self.path_patchify = nn.Sequential(
            nn.Linear(2, d_model // 2),  # 2维输入: (x,y) 控制点
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        if self.privileged_local_dim:
            self.privileged_local_embed = nn.Sequential(
                nn.Linear(self.privileged_local_dim, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
            )
            # Enabling the diagnostic branch must reproduce the frozen
            # Stage-1 output before any Stage-2 update.
            nn.init.zeros_(self.privileged_local_embed[-1].weight)
            nn.init.zeros_(self.privileged_local_embed[-1].bias)
        
        # 路径位置编码（可学习）
        self.path_pos_embed = nn.Parameter(
            torch.randn(1, n_path_steps, d_model) * 0.02
        )
        
        # ========== 时间步嵌入 ==========
        self.time_embedder = TimestepEmbedder(d_model)
        
        # ========== 改进的DiT Blocks（带Cross-Attention） ========== 
        self.dit_blocks = nn.ModuleList([
            TrajDiTBlock(
                d_model=d_model,
                n_heads=n_heads, 
                d_ff=d_inner,
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
        # self.main_pred = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model // 2, 2)  # (x, y)
        # )
        
        self.main_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # (x, y)
        )
        self.radial_slack_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # softplus makes slack positive with a non-zero derivative even when
        # raw_s is initialized at zero.
        nn.init.zeros_(self.radial_slack_pred[-1].weight)
        # A moderate initial slack keeps the untrained path near its chord
        # instead of placing random directions close to the hard boundary.
        initial_slack = 5.0
        initial_raw_s = math.log(math.expm1(initial_slack - 1e-4))
        nn.init.constant_(self.radial_slack_pred[-1].bias, initial_raw_s)
        
        # # 细节预测头（用于精细化调整主预测）
        # detail_in_dim = d_model + 2  # DiT特征 + 主预测的完整2维
        # self.detail_pred = nn.Sequential(
        #     nn.Linear(detail_in_dim, d_model // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model // 2, 3)  # 输出增量 (dx, dy， theta)
        # )
        
        # # Zero初始化细节预测头
        # nn.init.zeros_(self.detail_pred[-1].weight)
        # nn.init.zeros_(self.detail_pred[-1].bias)
        
        # ========== Rectified Flow 连续时间参数化 ==========
        self.n_path_steps = n_path_steps
        self.num_edges = n_path_steps
        self.num_control_points = 26
        self.trajectory_representation = PhysicalScaledEdgeResidualRepresentation(
            num_control_points=self.num_control_points
        )
        self.diffusion_steps = diffusion_steps  # 仅用于采样步数，不再作为离散索引
        
        # Rectified Flow 时间语义：
        #   t=0: z_0 = x_0 (纯数据)
        #   t=1: z_1 = x_1 (纯噪声)
        # 加噪公式：z_t = (1-t) * x_0 + t * x_1
        # 速度场：v_t = x_1 - x_0 (直线路径)
        # ODE: dz/dt = v_θ(z_t, t)
        
        # 时间步采样策略参数（用于训练时的智能采样）
        # 使用logit-normal分布，避免t=0和t=1的极端值
        self.register_buffer('t_sample_mean', torch.tensor(-0.8))  # logit空间的均值
        self.register_buffer('t_sample_std', torch.tensor(0.8))   # logit空间的标准差
    
    def sample_timesteps(self, n: int, device=None, generator=None):
        """
        连续时间步采样策略（Rectified Flow）
        
        使用logit-normal分布采样，避免t→0和t→1的极端值：
        - 正态分布采样 → sigmoid映射到(0,1) → 连续时间 t ∈ (0,1)
        - 避免t=0和t=1的数值不稳定问题
        
        Args:
            n: 采样数量
            device: 设备
        
        Returns:
            t_continuous: (n,) 连续时间 t ∈ (0, 1)
        """
        if device is None:
            device = self.t_sample_mean.device
        
        # logit-normal分布采样
        z = (
            torch.randn(n, device=device, generator=generator)
            * self.t_sample_std
            + self.t_sample_mean
        )
        # sigmoid映射到(0,1)，自然避免极端值
        t_continuous = torch.sigmoid(z)
        
        # 可选：进一步限制范围，避免数值问题
        t_continuous = torch.clamp(t_continuous, min=1e-5, max=1.0 - 1e-5)
        
        return t_continuous

    def project_zero_sum(self, residuals):
        return self.trajectory_representation.project_zero_sum(residuals)

    def decode_residual_edges(self, residuals, start_pose, goal_pose):
        return self.trajectory_representation.decode(
            residuals, start_pose[:, :2], goal_pose[:, :2]
        )

    def decode_feasible_residual_edges(self, residuals, start_pose, goal_pose):
        feasible_residuals = self.trajectory_representation.radial_project_residual(
            residuals,
            start_pose[:, :2],
            goal_pose[:, :2],
        )
        return self.decode_residual_edges(
            feasible_residuals, start_pose, goal_pose
        )

    def radial_feasible_x0(
        self,
        raw_v,
        raw_s,
        start_pose,
        goal_pose,
        return_diagnostics=False,
    ):
        return self.trajectory_representation.radial_feasible_residual(
            raw_v,
            raw_s,
            start_pose[:, :2],
            goal_pose[:, :2],
            return_diagnostics=return_diagnostics,
        )

    def forward(
        self,
        map_input,
        noisy_path,
        timestep,
        timestep_r,
        start_pose,
        goal_pose,
        privileged_local_features=None,
    ):
        """
        扩散模型前向传播（解耦数据流版本）- Rectified Flow
        
        【数据流架构】
        1. Path Tokens (x): 作为Query在DiT主干中流动并不断更新
        2. Map Tokens: 作为Key/Value在Cross-Attention中被查询（保持不变）
        3. Condition (c): Time + Start + Goal 融合后通过AdaLN调制每一层
        
        Args:
            map_input: (B, 3, H, W) 输入地图
            noisy_path: (B,25,2) 加噪后的缩放零和边残差
            timestep: (B,) 连续时间步 t ∈ [0, 1]
                     t=0: 纯数据, t=1: 纯噪声
            start_pose: (B, 4) 起点坐标 (x,y,cos(θ),sin(θ)) - 已归一化
            goal_pose: (B, 4) 终点坐标 (x,y,cos(θ),sin(θ)) - 已归一化
            
        Returns:
            model_output: (B,25,2) 缩放零和边残差预测
        """
        B = map_input.shape[0]
        
        # ========== 1. 编码地图 (Context, 单尺度25x25 KV) ==========
        feat_50 = self.map_fe_block1(map_input)           # (B, d_model//8, 50, 50)
        feat_25 = self.map_fe_block2(feat_50)             # (B, d_model//4, 25, 25)
        feat_12 = self.map_fe_block3(feat_25)             # (B, d_model//2, 12, 12)
        feat_12 = self.map_fe_block4(feat_12)             # (B, d_model, 12, 12)
        # feat_25_proj = self.map_proj_25(feat_25)          # (B, d_model, 25, 25)

        # map_tokens = self.reorder_dims(feat_25_proj)      # (B, 625, D)
        # map_tokens = self.map_position_enc(map_tokens, conv_shape=feat_25_proj.shape[-2:])
        map_tokens = self.reorder_dims(feat_12)           # (B, 144, D)
        map_tokens = self.map_position_enc(map_tokens, conv_shape=feat_12.shape[-2:])
        
        # ========== 2. 编码25条物理边残差 (Query Sequence) ==========
        if noisy_path.ndim != 3 or noisy_path.shape[1:] != (self.num_edges, 2):
            raise ValueError(
                f"Expected noisy_path (B,{self.num_edges},2), "
                f"got {tuple(noisy_path.shape)}"
            )

        path_tokens = self.path_patchify(noisy_path)  # (B, 25, D)
        combined_tokens = path_tokens + self.path_pos_embed
        if privileged_local_features is not None:
            if not self.privileged_local_dim:
                raise ValueError(
                    "privileged_local_features were provided, but "
                    "privileged_local_dim=0"
                )
            if (
                privileged_local_features.ndim != 3
                or privileged_local_features.shape[0] != B
                or privileged_local_features.shape[2] != self.privileged_local_dim
            ):
                raise ValueError(
                    "Expected privileged_local_features "
                    f"(B,N,{self.privileged_local_dim}), got "
                    f"{tuple(privileged_local_features.shape)}"
                )
            local_features = F.adaptive_avg_pool1d(
                privileged_local_features.transpose(1, 2),
                self.num_edges,
            ).transpose(1, 2)
            combined_tokens = (
                combined_tokens
                + self.privileged_local_embed(local_features)
            )
        
        # 2.3 预处理
        # 建议先 Norm 再进入 Block，保证分布一致性
        combined_tokens = self.layer_norm(combined_tokens)
        combined_tokens = self.dropout(combined_tokens)
        
        # ========== 3. 编码条件 (Condition) ==========
        t_emb = self.time_embedder(timestep)      # (B, D) - 时间步
        h_emb = self.time_embedder(torch.clamp(timestep - timestep_r, min=0.0, max=1.0))  # (B, D) - pMF步长
        s_emb = self.pose_embedder(start_pose)    # (B, D) - 起点
        g_emb = self.pose_embedder(goal_pose)     # (B, D) - 终点
        
        # 融合所有全局条件 -> (B, D)
        # cond = self.cond_mlp(torch.cat([t_emb, s_emb, g_emb], dim=-1))
        cond = self.cond_mlp(torch.cat([t_emb, h_emb, s_emb, g_emb], dim=-1))
        # cond = t_emb  # 仅使用时间步作为条件
        
        # ========== 4. Implicit Guidance Tokens ==========
        guidance_tokens = self.guidance_encoder(
            map_tokens=map_tokens,
            start_pose=start_pose,
            goal_pose=goal_pose,
            t_emb=t_emb,
            h_emb=h_emb
        )

        # ========== 5. DiT Blocks（解耦数据流） ==========
        # x (Query): combined tokens，在主干中不断更新
        # guidance_tokens (Key/Value): 轨迹进度对齐的隐式地图引导
        # cond: 全局条件，通过AdaLN调制每一层
        x = combined_tokens
        for block in self.dit_blocks:
            if self.training and self.use_gradient_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, guidance_tokens, cond, use_reentrant=False
                )
            else:
                x = block(x, guidance_tokens, cond)
            
        x = self.layer_norm(x)  # 最后LayerNorm
        
        # ========== 6. 条件径向可行域 x0 输出 ==========
        raw_v = self.main_pred(x)
        raw_s = self.radial_slack_pred(x.mean(dim=1))
        return self.radial_feasible_x0(
            raw_v,
            raw_s,
            start_pose,
            goal_pose,
        )
    
    def q_sample(self, x_start, t, noise=None):
        """
        Rectified Flow 前向加噪过程
        
        直线插值（Rectified Flow核心）：
            z_t = (1-t) * x_0 + t * x_1
            
        其中：
            x_0: 真实数据（t=0时为纯数据）
            x_1: 噪声（t=1时为纯噪声）
            t ∈ [0, 1]：连续时间
        
        速度场（常数，沿直线）：
            v = x_1 - x_0
        
        Args:
            x_start: (B, N, D) 真实数据 x_0
            t: (B,) 连续时间 t ∈ [0, 1]
            noise: (B, N, D) 噪声 x_1，如果为None则随机生成
        
        Returns:
            z_t: (B, N, D) 加噪数据
            noise: (B, N, D) 使用的噪声（用于计算损失）
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_start = self.project_zero_sum(x_start)
        noise = self.project_zero_sum(noise)
        
        # 扩展时间维度以匹配数据形状
        t = t.view(-1, 1, 1)  # (B, 1, 1)
        
        # Rectified Flow: z_t = (1-t) * x_0 + t * x_1
        z_t = self.project_zero_sum((1 - t) * x_start + t * noise)
        
        return z_t, noise
    
    def get_velocity(self, z_in, t_scalar, r_scalar, map_input, start_pose, goal_pose):
        """pMF 专用速度获取函数"""
        B = z_in.shape[0]
        device = z_in.device
        
        t_batch = torch.full((B,), t_scalar, device=device)
        r_batch = torch.full((B,), r_scalar, device=device)
        
        # 模型前向传播，现在需要传入 t 和 r
        model_out = self.forward(map_input, z_in, t_batch, r_batch, start_pose, goal_pose)
        
        # 统一逻辑：先求 pred_x0
        if self.prediction_type == 'v':
            pred_x0 = z_in - t_scalar * model_out
        elif self.prediction_type == 'x0':
            pred_x0 = model_out
        elif self.prediction_type == 'epsilon':
            pred_x0 = (z_in - t_scalar * model_out) # 基于 Rectified Flow: z = (1-t)x0 + t*eps -> x0 = z - t*eps 是不对的，应该是标准 RF 转换
            # 建议训练和采样统一使用 x0 prediction 
        
        # pMF 采样逻辑：在 ODE 中，我们依然需要瞬时速度场 u = (z - x0) / t
        if t_scalar < 1e-5:
            return torch.zeros_like(z_in)
            
        return self.project_zero_sum((z_in - pred_x0) / t_scalar)

    def get_velocity_tensor(self, z_in, t_scalar, r_scalar, map_input, start_pose, goal_pose):
        """可微分的速度获取函数（t/r 使用 0-d tensor）"""
        B = z_in.shape[0]
        device = z_in.device

        if not torch.is_tensor(t_scalar):
            t_scalar = torch.tensor(t_scalar, device=device, dtype=z_in.dtype)
        if not torch.is_tensor(r_scalar):
            r_scalar = torch.tensor(r_scalar, device=device, dtype=z_in.dtype)

        t_batch = t_scalar.expand(B)
        r_batch = r_scalar.expand(B)

        model_out = self.forward(map_input, z_in, t_batch, r_batch, start_pose, goal_pose)

        if self.prediction_type == 'v':
            pred_x0 = z_in - t_scalar * model_out
        elif self.prediction_type == 'x0':
            pred_x0 = model_out
        else:  # 'epsilon'
            pred_x0 = (z_in - t_scalar * model_out)

        eps = 1e-5
        return self.project_zero_sum((z_in - pred_x0) / (t_scalar + eps))

    def sample_differentiable(self, map_input, start_pose, goal_pose, num_steps=3,
                              solver='pmf_refined', reconstruct_trajectory=True, num_traj_points=100):
        """
        可微分采样（用于训练期的梯度回传）。零和物理边残差在解码时
        结构性保证真实起终点，不需要事后覆盖。

        Args:
            map_input: (B, 3, H, W)
            start_pose: (B, 4) 归一化起点 (x,y,cos,sin)
            goal_pose: (B, 4) 归一化终点 (x,y,cos,sin)
        Returns:
            reconstructed_traj: (B, num_traj_points, 2) 或控制点 (B, 26, 2)
        """
        device = map_input.device
        B = map_input.shape[0]

        z = self.project_zero_sum(
            torch.randn(B, self.num_edges, 2, device=device)
        )

        if solver == 'pmf_refined':
            t_start = torch.tensor(1.0, device=device)
            t_mid = torch.tensor(0.1, device=device)

            v_coarse = self.get_velocity_tensor(z, t_start, t_mid, map_input, start_pose, goal_pose)
            z = self.project_zero_sum(z + (t_mid - t_start) * v_coarse)

            timesteps = torch.linspace(t_mid, torch.tensor(0.0, device=device), num_steps + 1, device=device)
            for i in range(num_steps):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_curr
                v_fine = self.get_velocity_tensor(z, t_curr, t_next, map_input, start_pose, goal_pose)
                z = self.project_zero_sum(z + dt * v_fine)
        elif solver == 'euler':
            timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
            for i in range(num_steps):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_curr
                v_pred = self.get_velocity_tensor(z, t_curr, t_next, map_input, start_pose, goal_pose)
                z = self.project_zero_sum(z + dt * v_pred)
        elif solver == 'heun':
            timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
            for i in range(num_steps):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_curr
                v1 = self.get_velocity_tensor(z, t_curr, t_next, map_input, start_pose, goal_pose)
                z_pred = z + dt * v1
                v2 = self.get_velocity_tensor(z_pred, t_next, t_next, map_input, start_pose, goal_pose)
                z = self.project_zero_sum(z + dt * (v1 + v2) / 2)
        else:
            raise ValueError(f"Unknown solver: {solver}. Choose 'pmf_refined', 'euler' or 'heun'.")

        control_points_normalized = self.decode_feasible_residual_edges(
            z, start_pose, goal_pose
        )
        control_points_denorm = control_points_normalized * self.coordinate_scale

        if not reconstruct_trajectory:
            return control_points_denorm

        bspline_layer = DifferentiableBSpline(
            num_control_points=self.num_control_points,
            num_output_points=num_traj_points,
            degree=3
        ).to(device)
        reconstructed_traj = bspline_layer(control_points_denorm)
        return reconstructed_traj
    
    @torch.no_grad()
    def sample_pmf_onestep(self, map_input, start_pose, goal_pose):
        """pMF 核心：单步生成"""
        device = map_input.device
        B = map_input.shape[0]
        
        # 从纯噪声开始 t=1
        z = self.project_zero_sum(
            torch.randn(B, self.num_edges, 2, device=device)
        )
        
        t_batch = torch.ones(B, device=device)
        r_batch = torch.zeros(B, device=device) # 目标是 0 
        
        # 模型在 pMF 训练下，t=1, r=0 的输出即为修正后的高质量 x0
        pred_x0 = self.forward(map_input, z, t_batch, r_batch, start_pose, goal_pose)
        
        return self.decode_feasible_residual_edges(pred_x0, start_pose, goal_pose)
    
    @torch.no_grad()
    def sample_pmf_refined(self, map_input, start_pose, goal_pose, 
                                    refine_steps=3, t_refine_start=0.1):
        """
        1. 阶段一：大步长跳跃。从 t=1 跨越到 t=coarse_step_r。
        2. 阶段二：小步长精修。从 t=coarse_step_r 细化积分到 t=0。
        这样保证了所有 z 都在同一个连续的 Probability Flow 轨迹上。
        """
        device = map_input.device
        B = map_input.shape[0]

        # --- 初始化 ---
        z = self.project_zero_sum(
            torch.randn(B, self.num_edges, 2, device=device)
        )  # t=1
        
        # --- 阶段一：Coarse Jump (利用 pMF 的跨步能力) ---
        t_start = 1.0
        t_mid = t_refine_start
        
        # 调用模型预测从 1.0 到 0.1 的平均速度
        # 根据 pMF 定义：z_mid = z_1 + (t_mid - t_start) * V(z_1, t=1, r=t_mid)
        v_coarse = self.get_velocity(z, t_start, t_mid, map_input, start_pose, goal_pose)
        z = self.project_zero_sum(
            z + (t_mid - t_start) * v_coarse
        )  # 注意此时 dt = -0.9
        
        # --- 阶段二：Fine-grained Integration (局部细化) ---
        # 此时 z 已经严格处于 Probability Flow 上的 t=0.1 位置
        timesteps = torch.linspace(t_mid, 0.0, refine_steps + 1, device=device)
        
        for i in range(refine_steps):
            t_curr = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t_curr
            
            # 保持 r = t_next，每一小步都利用 pMF 修正局部截断误差
            v_fine = self.get_velocity(z, t_curr, t_next, map_input, start_pose, goal_pose)
            z = self.project_zero_sum(z + dt * v_fine)
            
        return self.decode_feasible_residual_edges(z, start_pose, goal_pose)
    
    @torch.no_grad()
    def sample_euler(self, map_input, start_pose, goal_pose, num_steps=None):
        """
        一阶Euler ODE求解器（Rectified Flow采样）
        
        ODE求解：
            dz/dt = v_θ(z_t, t)
            
        Euler方法（一阶）：
            z_{t+dt} = z_t + dt * v_θ(z_t, t)
        
        在Rectified Flow中：
            - 模型直接预测速度场 v = x_1 - x_0
            - ODE从 t=1 (纯噪声) 积分到 t=0 (纯数据)
        
        Args:
            map_input: (B, 3, H, W) 输入地图
            start_pose: (B, 4) 起点 (x,y,cos,sin)
            goal_pose: (B, 4) 终点 (x,y,cos,sin)
            num_steps: ODE求解步数（默认使用self.diffusion_steps）
        
        Returns:
            control_points: (B, 26, 2) 控制点（含起终点）
        """
        if num_steps is None:
            num_steps = self.diffusion_steps
        
        device = map_input.device
        B = map_input.shape[0]
        
        # 初始化：t=1时为纯噪声
        z = self.project_zero_sum(
            torch.randn(B, self.num_edges, 2, device=device)
        )
        
        # 时间步划分：从t=1到t=0
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        
        # ODE积分（从t=1到t=0）
        for i in range(num_steps):
            t_curr = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t_curr
            
            # pMF 迭代建议：r 等于下一步的时间
            v_pred = self.get_velocity(z, t_curr, t_next, map_input, start_pose, goal_pose)
            z = self.project_zero_sum(z + dt * v_pred)
        
        return self.decode_feasible_residual_edges(z, start_pose, goal_pose)
    
    @torch.no_grad()
    def sample_heun(self, map_input, start_pose, goal_pose, num_steps=None):
        if num_steps is None:
            num_steps = self.diffusion_steps
        
        device = map_input.device
        B = map_input.shape[0]
        
        # 初始化：t=1时为纯噪声
        z = self.project_zero_sum(
            torch.randn(B, self.num_edges, 2, device=device)
        )
        
        # 时间步划分：从t=1到t=0
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        
        # ODE积分（从t=1到t=0）
        for i in range(num_steps):
            t_curr = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t_curr
            
            # 1. 评估 v1 (r 取 t_next)
            v1 = self.get_velocity(z, t_curr, t_next, map_input, start_pose, goal_pose)
            
            # 2. 预测步
            z_pred = z + dt * v1
            
            # 3. 评估 v2
            v2 = self.get_velocity(z_pred, t_next, t_next, map_input, start_pose, goal_pose)
            
            # 4. 校正步
            z = self.project_zero_sum(z + dt * (v1 + v2) / 2)
        
        return self.decode_feasible_residual_edges(z, start_pose, goal_pose)
    
    
    @torch.no_grad()
    def sample(self, map_input, start_pose, goal_pose, num_samples=5, num_steps=50, 
               solver='heun', reconstruct_trajectory=True, num_traj_points=100):
        """
        Rectified Flow采样（使用ODE求解器）
        
        【完整B样条】25个缩放零和物理边残差解码为26个控制点
        
        Args:
            map_input: (1, 3, H, W) 输入地图（3通道：normal_x, normal_y, normal_z）
            start_pose: (1, 4) 起点坐标 (x,y,cos(θ),sin(θ)) (已归一化)
            goal_pose: (1, 4) 终点坐标 (x,y,cos(θ),sin(θ)) (已归一化)
            num_samples: 采样数量
            num_steps: ODE求解步数
            solver: ODE求解器类型 ('euler' 或 'heun')
            reconstruct_trajectory: 是否从控制点重建轨迹（默认True）
            num_traj_points: 重建后的轨迹点数（默认100）
            
        Returns:
            如果reconstruct_trajectory=True: 
                (num_samples, num_traj_points, 2) - 重建的轨迹(x,y)
            如果reconstruct_trajectory=False: 
                (num_samples, 26, 2) - 完整控制点(包含起终点)
        """
        device = map_input.device
        
        # 扩展batch维度（处理任意批大小）
        # 如果输入批大小为1，使用expand；否则使用repeat然后提取第一个样本
        batch_size = map_input.shape[0]
        
        # 【修复】对每个batch样本生成num_samples个采样
        # 使用repeat_interleave保证样本顺序：[s1, s1, ..., s1, s2, s2, ..., s2, ...]
        if num_samples == 1:
            # 特殊优化：只需1个采样
            map_input_batch = map_input  # (B, C, H, W)
            start_pose_batch = start_pose  # (B, 4)
            goal_pose_batch = goal_pose  # (B, 4)
        else:
            # 多采样：为每个样本复制num_samples次（保持顺序）
            map_input_batch = map_input.repeat_interleave(num_samples, dim=0)  # (B*num_samples, C, H, W)
            start_pose_batch = start_pose.repeat_interleave(num_samples, dim=0)  # (B*num_samples, 4)
            goal_pose_batch = goal_pose.repeat_interleave(num_samples, dim=0)  # (B*num_samples, 4)
        
        # 选择ODE求解器
        if solver == 'pmf_onestep':
            result = self.sample_pmf_onestep(map_input_batch, 
                                             start_pose_batch, 
                                             goal_pose_batch)
        elif solver == 'pmf_refined':
            result = self.sample_pmf_refined(map_input_batch, 
                                             start_pose_batch, 
                                             goal_pose_batch, 
                                             refine_steps=num_steps, 
                                             t_refine_start=0.1)
        elif solver == 'euler':
            result = self.sample_euler(map_input_batch, 
                                       start_pose_batch, 
                                       goal_pose_batch, 
                                       num_steps=num_steps)
        elif solver == 'heun':
            result = self.sample_heun(map_input_batch, 
                                      start_pose_batch, 
                                      goal_pose_batch, 
                                      num_steps=num_steps)
        else:
            raise ValueError(f"Unknown solver: {solver}. Choose 'pmf_onestep', 'euler' or 'heun'.")
        
        # result: (batch_size, 26, 2) 或 (batch_size*num_samples, 26, 2)
        control_points_denorm = result * self.coordinate_scale
        
        if not reconstruct_trajectory:
            return control_points_denorm
        
        # 使用B样条重建轨迹
        bspline_layer = DifferentiableBSpline(
            num_control_points=self.num_control_points,
            num_output_points=num_traj_points,
            degree=3
        ).to(device)
        reconstructed_traj = bspline_layer(control_points_denorm)
        
        return reconstructed_traj  # (batch_size, num_traj_points, 2) 或 (batch_size*num_samples, num_traj_points, 2)

    

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


class CostConditionedPathDiffusionTransformer(PathDiffusionTransformer):
    """
    成本条件化路径扩散变换器
    
    基于PathDiffusionTransformer，扩展功能支持成本标量作为额外的条件输入。
    
    【核心改进】
    1. 新增成本输入：cost 是一个标量，表示路径的某种成本度量（如长度、曲率、碰撞风险等）
    2. 成本嵌入：使用与时间步相同的正弦位置编码方法进行成本的Embedding
    3. 条件融合：将成本条件与时间步和起终点条件一并融合到AdaLN中
    
    【数据流】
    地图 → CNN特征提取 → Token融合 → 
    [路径tokens + 时间步embedding + 成本embedding + 起终点embedding] →
    DiT Blocks (带Cross-Attention) → 输出预测
    
    【使用场景】
    - 条件路径规划：根据成本约束生成不同的路径
    - 成本感知的轨迹优化：融合多种成本维度
    - 可控生成：通过调整成本条件控制生成结果
    """
    
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, 
                 pad_idx, dropout, n_position, train_shape, 
                 n_path_steps=25, diffusion_steps=50, prediction_type='epsilon', loss_type=None,
                 coordinate_scale=MAP_HALF_EXTENT):
        """
        初始化成本条件化路径扩散变换器
        
        Args:
            n_layers: Transformer层数
            n_heads: 多头注意力头数
            d_k: Key的维度
            d_v: Value的维度
            d_model: 模型隐层维度
            d_inner: FFN内层维度
            pad_idx: 填充索引
            dropout: Dropout比率
            n_position: 最大位置数
            train_shape: 训练输入形状
            n_path_steps: 缩放物理边残差token数量，当前固定为25
            diffusion_steps: 扩散步数
            prediction_type: 预测类型 ('epsilon', 'x0', 'v')
            loss_type: 损失类型
        """
        super().__init__(
            n_layers=n_layers,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_inner=d_inner,
            pad_idx=pad_idx,
            dropout=dropout,
            n_position=n_position,
            train_shape=train_shape,
            n_path_steps=n_path_steps,
            diffusion_steps=diffusion_steps,
            prediction_type=prediction_type,
            loss_type=loss_type,
            coordinate_scale=coordinate_scale,
        )
        
        # ========== 成本嵌入模块 ==========
        # 连续标量cost使用MLP嵌入（比sin/cos对标量更稳定）
        self.cost_embedder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        # 时间步嵌入缩放：将 t,r 从[0,1]放大后再做sin/cos编码，增强区分度
        self.time_embed_scale = 1000.0
        # learnable null token：用于CFG中的无条件分支
        self.null_cost_embed = nn.Parameter(torch.zeros(1, d_model))
        
        # ========== 更新条件融合MLP ==========
        # 原始：4个d_model (time + time_r + start + goal)
        # 新增：1个d_model (cost)
        # 总计：5个d_model
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 6),  # 5 = time + time_r + start + goal + cost
            nn.GELU(),
            nn.Linear(d_model * 6, d_model)
        )
    
    def forward(self, map_input, noisy_path, timestep, timestep_r, start_pose, goal_pose, cost,
                cost_drop_mask=None, force_null_cost=False):
        """
        前向传播 - 支持成本条件
        
        【数据流架构】
        1. Path Tokens (x): 作为Query在DiT主干中流动并不断更新
        2. Map Tokens: 作为Key/Value在Cross-Attention中被查询（保持不变）
        3. Condition (c): Time + Start + Goal + Cost 融合后通过AdaLN调制每一层
        
        Args:
            map_input: (B, 3, H, W) 输入地图
            noisy_path: (B,25,2) 加噪后的缩放零和边残差
            timestep: (B,) 连续时间步 t ∈ [0, 1]
                     t=0: 纯数据, t=1: 纯噪声
            timestep_r: (B,) 反向时间步
            start_pose: (B, 4) 起点坐标 (x,y,cos(θ),sin(θ)) - 已归一化
            goal_pose: (B, 4) 终点坐标 (x,y,cos(θ),sin(θ)) - 已归一化
            cost: (B,) 成本标量，可以是单个数值或批量成本
                  范围可以是任意，建议归一化到 [0, 1]
            
        Returns:
            model_output: (B,25,2) 缩放零和边残差预测
        """
        B = map_input.shape[0]
        
        # ========== 1. 编码地图 (Context, 单尺度12×12 KV) ==========
        feat_50 = self.map_fe_block1(map_input)           # (B, d_model//8, 50, 50)
        feat_25 = self.map_fe_block2(feat_50)             # (B, d_model//4, 25, 25)
        feat_12 = self.map_fe_block3(feat_25)             # (B, d_model//2, 12, 12)
        feat_12 = self.map_fe_block4(feat_12)             # (B, d_model, 12, 12)
        
        map_tokens = self.reorder_dims(feat_12)           # (B, 144, D)
        map_tokens = self.map_position_enc(map_tokens, conv_shape=feat_12.shape[-2:])
        
        # ========== 2. 编码25条局部边残差 (Query Sequence) ==========
        if noisy_path.ndim != 3 or noisy_path.shape[1:] != (self.num_edges, 2):
            raise ValueError(
                f"Expected noisy_path (B,{self.num_edges},2), "
                f"got {tuple(noisy_path.shape)}"
            )
        path_tokens = self.path_patchify(noisy_path)
        path_tokens = path_tokens + self.path_pos_embed  # (B, N, D)
        combined_tokens = path_tokens
        
        # 2.3 预处理
        combined_tokens = self.layer_norm(combined_tokens)
        combined_tokens = self.dropout(combined_tokens)
        
        # ========== 3. 编码条件（新增成本） ==========
        t_emb = self.time_embedder(timestep * self.time_embed_scale)              # (B, D) - 时间步
        r_emb = self.time_embedder(timestep_r * self.time_embed_scale)            # (B, D) - 反向时间步
        s_emb = self.pose_embedder(start_pose)            # (B, D) - 起点
        g_emb = self.pose_embedder(goal_pose)             # (B, D) - 终点
        
        # 【关键改进】成本条件化嵌入
        # 方式1：如果cost是标量或单个值，需要扩展为批次大小
        if cost.dim() == 0:
            # 单个标量，扩展为批量
            cost_batch = cost.unsqueeze(0).expand(B)
        else:
            # 已经是批量形式 (B,)
            cost_batch = cost
        
        # 连续cost标量通过MLP直接映射到特征空间
        cost_embed_input = cost_batch.float().reshape(B, 1)
        cost_emb = self.cost_embedder(cost_embed_input)   # (B, D)
        null_emb = self.null_cost_embed.expand(B, -1)     # (B, D)

        # CFG训练：按mask替换为learnable null token；
        # 推理CFG无条件分支：force_null_cost=True时全量替换
        if force_null_cost:
            cost_emb = null_emb
        elif cost_drop_mask is not None:
            mask = cost_drop_mask
            if mask.dim() > 1:
                mask = mask.reshape(B)
            mask = mask.to(device=cost_emb.device, dtype=torch.bool)
            cost_emb = torch.where(mask.unsqueeze(-1), null_emb, cost_emb)
        
        # 融合所有全局条件 -> (B, D)
        cond = self.cond_mlp(torch.cat([t_emb, r_emb, s_emb, g_emb, cost_emb], dim=-1))
        
        # ========== 4. DiT Blocks（解耦数据流） ==========
        # x (Query): combined tokens，在主干中不断更新
        # map_tokens (Key/Value): 地图tokens，保持不变
        # cond: 全局条件，包含时间步、起终点和成本
        x = combined_tokens
        for block in self.dit_blocks:
            if self.training and self.use_gradient_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, map_tokens, cond, use_reentrant=False
                )
            else:
                x = block(x, map_tokens, cond)
        
        x = self.layer_norm(x)  # 最后LayerNorm
        
        # ========== 5. 条件径向可行域 x0 输出 ==========
        raw_v = self.main_pred(x)
        raw_s = self.radial_slack_pred(x.mean(dim=1))
        return self.radial_feasible_x0(
            raw_v,
            raw_s,
            start_pose,
            goal_pose,
        )
    
    def get_velocity(self, z_in, t_scalar, r_scalar, cost_scalar, map_input, start_pose, goal_pose, w=1.0):
        """
        成本条件化速度获取函数（用于pMF采样）
        
        Args:
            z_in: (B, N, 2) 当前噪声状态
            t_scalar: 时间步标量或张量
            r_scalar: 反向时间步标量或张量
            cost_scalar: 成本标量或张量
            map_input: (B, 3, H, W) 地图
            start_pose: (B, 4) 起点
            goal_pose: (B, 4) 终点
            
        Returns:
            v_field: (B, N, 2) 速度场
        """
        B = z_in.shape[0]
        device = z_in.device
        
        # 确保所有输入都是批量形式
        if not isinstance(t_scalar, torch.Tensor):
            t_batch = torch.full((B,), t_scalar, device=device, dtype=z_in.dtype)
        else:
            t_batch = t_scalar if t_scalar.shape[0] == B else torch.full((B,), t_scalar.item(), device=device, dtype=z_in.dtype)
        
        if not isinstance(r_scalar, torch.Tensor):
            r_batch = torch.full((B,), r_scalar, device=device, dtype=z_in.dtype)
        else:
            r_batch = r_scalar if r_scalar.shape[0] == B else torch.full((B,), r_scalar.item(), device=device, dtype=z_in.dtype)
        
        if not isinstance(cost_scalar, torch.Tensor):
            cost_batch = torch.full((B,), cost_scalar, device=device, dtype=z_in.dtype)
        else:
            cost_batch = cost_scalar if cost_scalar.shape[0] == B else torch.full((B,), cost_scalar.item(), device=device, dtype=z_in.dtype)
        
        # 模型前向传播（支持CFG guidance）
        model_out = self._forward_with_cfg(
            map_input, z_in, t_batch, r_batch, start_pose, goal_pose, cost_batch, w=w
        )
        
        # 统一逻辑：先求 pred_x0
        if self.prediction_type == 'v':
            pred_x0 = z_in - t_scalar * model_out
        elif self.prediction_type == 'x0':
            pred_x0 = model_out
        else:  # 'epsilon'
            pred_x0 = (z_in - t_scalar * model_out)
        
        # 计算速度场
        eps = 1e-5
        t_val = t_scalar if isinstance(t_scalar, (int, float)) else t_scalar.item()
        if t_val < 1e-5:
            return torch.zeros_like(z_in)
        
        return self.project_zero_sum((z_in - pred_x0) / t_val)
    
    def get_velocity_tensor(self, z_in, t_scalar, r_scalar, cost_scalar, map_input, start_pose, goal_pose, w=1.0):
        """
        可微分的成本条件化速度获取函数（t/r/cost 使用 0-d tensor）
        
        Args:
            z_in: (B, N, 2) 当前噪声状态
            t_scalar: 时间步标量或张量
            r_scalar: 反向时间步标量或张量
            cost_scalar: 成本标量或张量
            map_input: (B, 3, H, W) 地图
            start_pose: (B, 4) 起点
            goal_pose: (B, 4) 终点
            
        Returns:
            v_field: (B, N, 2) 速度场
        """
        B = z_in.shape[0]
        device = z_in.device

        # 确保为张量
        if not torch.is_tensor(t_scalar):
            t_scalar = torch.tensor(t_scalar, device=device, dtype=z_in.dtype)
        if not torch.is_tensor(r_scalar):
            r_scalar = torch.tensor(r_scalar, device=device, dtype=z_in.dtype)
        if not torch.is_tensor(cost_scalar):
            cost_scalar = torch.tensor(cost_scalar, device=device, dtype=z_in.dtype)

        # 扩展为批量
        t_batch = t_scalar.expand(B)
        r_batch = r_scalar.expand(B)
        cost_batch = cost_scalar.expand(B)

        # 前向传播（支持CFG guidance）
        model_out = self._forward_with_cfg(
            map_input, z_in, t_batch, r_batch, start_pose, goal_pose, cost_batch, w=w
        )

        # 预测x0
        if self.prediction_type == 'v':
            pred_x0 = z_in - t_scalar * model_out
        elif self.prediction_type == 'x0':
            pred_x0 = model_out
        else:  # 'epsilon'
            pred_x0 = (z_in - t_scalar * model_out)

        # 计算速度场（可微分）
        eps = 1e-5
        return self.project_zero_sum((z_in - pred_x0) / (t_scalar + eps))

    def _expand_cost_to_batch(self, cost_scalar, batch_size, device, dtype):
        """将 cost 标量或张量统一扩展为 batch 形式。"""
        if not torch.is_tensor(cost_scalar):
            return torch.full((batch_size,), float(cost_scalar), device=device, dtype=dtype)

        cost_scalar = cost_scalar.to(device=device, dtype=dtype)
        if cost_scalar.dim() == 0:
            return cost_scalar.expand(batch_size)
        if cost_scalar.shape[0] == batch_size:
            return cost_scalar
        return torch.full((batch_size,), float(cost_scalar.reshape(-1)[0].item()), device=device, dtype=dtype)

    def _forward_with_cfg(self, map_input, z_in, t_batch, r_batch, start_pose, goal_pose, cost_batch, w=1.0):
        """CFG前向：pred = pred_uncond + w * (pred_cond - pred_uncond)。

        这里的“uncond”采用 learnable null cost token 分支。
        """
        if abs(float(w) - 1.0) < 1e-8:
            return self.forward(map_input, z_in, t_batch, r_batch, start_pose, goal_pose, cost_batch)

        pred_cond = self.forward(map_input, z_in, t_batch, r_batch, start_pose, goal_pose, cost_batch)
        pred_uncond = self.forward(
            map_input, z_in, t_batch, r_batch, start_pose, goal_pose, cost_batch,
            force_null_cost=True
        )
        return pred_uncond + float(w) * (pred_cond - pred_uncond)

    def sample_differentiable(self, map_input, start_pose, goal_pose, cost_scalar=0.0, num_steps=3,
                              solver='pmf_refined', reconstruct_trajectory=True, num_traj_points=100, w=1.0):
        """
        可微分采样（用于训练期的梯度回传），支持 cost 条件。
        """
        device = map_input.device
        B = map_input.shape[0]

        z = self.project_zero_sum(
            torch.randn(B, self.num_edges, 2, device=device)
        )
        cost_batch = self._expand_cost_to_batch(cost_scalar, B, device, z.dtype)

        if solver == 'pmf_refined':
            t_start = torch.tensor(1.0, device=device, dtype=z.dtype)
            t_mid = torch.tensor(0.1, device=device, dtype=z.dtype)

            v_coarse = self.get_velocity_tensor(z, t_start, t_mid, cost_batch, map_input, start_pose, goal_pose, w=w)
            z = self.project_zero_sum(z + (t_mid - t_start) * v_coarse)

            timesteps = torch.linspace(t_mid, torch.tensor(0.0, device=device, dtype=z.dtype), num_steps + 1, device=device)
            for i in range(num_steps):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_curr
                v_fine = self.get_velocity_tensor(z, t_curr, t_next, cost_batch, map_input, start_pose, goal_pose, w=w)
                z = self.project_zero_sum(z + dt * v_fine)
        elif solver == 'euler':
            timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=z.dtype)
            for i in range(num_steps):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_curr
                v_pred = self.get_velocity_tensor(z, t_curr, t_next, cost_batch, map_input, start_pose, goal_pose, w=w)
                z = self.project_zero_sum(z + dt * v_pred)
        elif solver == 'heun':
            timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=z.dtype)
            for i in range(num_steps):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_curr
                v1 = self.get_velocity_tensor(z, t_curr, t_next, cost_batch, map_input, start_pose, goal_pose, w=w)
                z_pred = z + dt * v1
                v2 = self.get_velocity_tensor(z_pred, t_next, t_next, cost_batch, map_input, start_pose, goal_pose, w=w)
                z = self.project_zero_sum(z + dt * (v1 + v2) / 2)
        else:
            raise ValueError(f"Unknown solver: {solver}. Choose 'pmf_refined', 'euler' or 'heun'.")

        control_points_normalized = self.decode_feasible_residual_edges(
            z, start_pose, goal_pose
        )
        control_points_denorm = control_points_normalized * self.coordinate_scale

        if not reconstruct_trajectory:
            return control_points_denorm

        bspline_layer = DifferentiableBSpline(
            num_control_points=self.num_control_points,
            num_output_points=num_traj_points,
            degree=3
        ).to(device)
        reconstructed_traj = bspline_layer(control_points_denorm)
        return reconstructed_traj

    @torch.no_grad()
    def sample_pmf_onestep(self, map_input, start_pose, goal_pose, cost_scalar=0.0, w=1.0):
        """pMF 核心：单步生成，支持 cost 条件。"""
        device = map_input.device
        B = map_input.shape[0]
        z = self.project_zero_sum(
            torch.randn(B, self.num_edges, 2, device=device)
        )
        cost_batch = self._expand_cost_to_batch(cost_scalar, B, device, z.dtype)

        t_batch = torch.ones(B, device=device, dtype=z.dtype)
        r_batch = torch.zeros(B, device=device, dtype=z.dtype)
        pred_x0 = self._forward_with_cfg(
            map_input, z, t_batch, r_batch, start_pose, goal_pose, cost_batch, w=w
        )

        return self.decode_feasible_residual_edges(pred_x0, start_pose, goal_pose)

    @torch.no_grad()
    def sample_pmf_refined(self, map_input, start_pose, goal_pose, cost_scalar=0.0,
                           refine_steps=3, t_refine_start=0.1, w=1.0):
        """pMF 两阶段采样，支持 cost 条件。"""
        device = map_input.device
        B = map_input.shape[0]
        z = self.project_zero_sum(
            torch.randn(B, self.num_edges, 2, device=device)
        )
        cost_batch = self._expand_cost_to_batch(cost_scalar, B, device, z.dtype)

        t_start = 1.0
        t_mid = t_refine_start
        v_coarse = self.get_velocity(z, t_start, t_mid, cost_batch, map_input, start_pose, goal_pose, w=w)
        z = self.project_zero_sum(z + (t_mid - t_start) * v_coarse)

        timesteps = torch.linspace(t_mid, 0.0, refine_steps + 1, device=device, dtype=z.dtype)
        for i in range(refine_steps):
            t_curr = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t_curr
            v_fine = self.get_velocity(z, t_curr, t_next, cost_batch, map_input, start_pose, goal_pose, w=w)
            z = self.project_zero_sum(z + dt * v_fine)

        return self.decode_feasible_residual_edges(z, start_pose, goal_pose)

    @torch.no_grad()
    def sample_euler(self, map_input, start_pose, goal_pose, cost_scalar=0.0, num_steps=None, w=1.0):
        """Euler ODE 采样，支持 cost 条件。"""
        if num_steps is None:
            num_steps = self.diffusion_steps

        device = map_input.device
        B = map_input.shape[0]
        z = self.project_zero_sum(
            torch.randn(B, self.num_edges, 2, device=device)
        )
        cost_batch = self._expand_cost_to_batch(cost_scalar, B, device, z.dtype)

        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=z.dtype)
        for i in range(num_steps):
            t_curr = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t_curr
            v_pred = self.get_velocity(z, t_curr, t_next, cost_batch, map_input, start_pose, goal_pose, w=w)
            z = self.project_zero_sum(z + dt * v_pred)

        return self.decode_feasible_residual_edges(z, start_pose, goal_pose)

    @torch.no_grad()
    def sample_heun(self, map_input, start_pose, goal_pose, cost_scalar=0.0, num_steps=None, w=1.0):
        """Heun ODE 采样，支持 cost 条件。"""
        if num_steps is None:
            num_steps = self.diffusion_steps

        device = map_input.device
        B = map_input.shape[0]
        z = self.project_zero_sum(
            torch.randn(B, self.num_edges, 2, device=device)
        )
        cost_batch = self._expand_cost_to_batch(cost_scalar, B, device, z.dtype)

        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=z.dtype)
        for i in range(num_steps):
            t_curr = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t_curr

            v1 = self.get_velocity(z, t_curr, t_next, cost_batch, map_input, start_pose, goal_pose, w=w)
            z_pred = z + dt * v1
            v2 = self.get_velocity(z_pred, t_next, t_next, cost_batch, map_input, start_pose, goal_pose, w=w)
            z = self.project_zero_sum(z + dt * (v1 + v2) / 2)

        return self.decode_feasible_residual_edges(z, start_pose, goal_pose)

    @torch.no_grad()
    def sample(self, map_input, start_pose, goal_pose, cost_scalar=0.0, num_samples=5, num_steps=50,
               solver='heun', reconstruct_trajectory=True, num_traj_points=100, w=1.0):
        """
        成本条件化采样接口。

        这里重定义 sample 的原因是：基类 sample 会调用不带 cost 参数的 get_velocity，
        对新网络类会丢失 cost 条件。
        """
        device = map_input.device
        batch_size = map_input.shape[0]

        cost_batch = self._expand_cost_to_batch(cost_scalar, batch_size, device, map_input.dtype)

        if num_samples == 1:
            map_input_batch = map_input
            start_pose_batch = start_pose
            goal_pose_batch = goal_pose
            cost_input_batch = cost_batch
        else:
            map_input_batch = map_input.repeat_interleave(num_samples, dim=0)
            start_pose_batch = start_pose.repeat_interleave(num_samples, dim=0)
            goal_pose_batch = goal_pose.repeat_interleave(num_samples, dim=0)
            cost_input_batch = cost_batch.repeat_interleave(num_samples, dim=0)

        if solver == 'pmf_onestep':
            result = self.sample_pmf_onestep(
                map_input_batch, start_pose_batch, goal_pose_batch, cost_input_batch, w=w
            )
        elif solver == 'pmf_refined':
            result = self.sample_pmf_refined(map_input_batch, start_pose_batch, goal_pose_batch, cost_input_batch,
                                             refine_steps=num_steps, t_refine_start=0.1, w=w)
        elif solver == 'euler':
            result = self.sample_euler(map_input_batch, start_pose_batch, goal_pose_batch, cost_input_batch,
                                       num_steps=num_steps, w=w)
        elif solver == 'heun':
            result = self.sample_heun(map_input_batch, start_pose_batch, goal_pose_batch, cost_input_batch,
                                     num_steps=num_steps, w=w)
        else:
            raise ValueError(f"Unknown solver: {solver}. Choose 'pmf_onestep', 'euler' or 'heun'.")

        control_points_denorm = result * self.coordinate_scale

        if not reconstruct_trajectory:
            return control_points_denorm

        bspline_layer = DifferentiableBSpline(
            num_control_points=self.num_control_points,
            num_output_points=num_traj_points,
            degree=3
        ).to(device)
        reconstructed_traj = bspline_layer(control_points_denorm)
        return reconstructed_traj

