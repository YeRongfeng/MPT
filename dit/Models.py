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
from boundary_constrained_path import (
    BoundaryConstrainedPathRepresentation,
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
    """Conditional Path MeanFlow 主干网络。

    路径状态为 22×2 whitened free path coordinates。网络始终预测目标
    路径坐标 ``y_1``，训练时使用 Path MeanFlow 目标，部署时只计算
    ``t=1, r=0`` 的单步映射。端点位置和航向由一阶边界约束 B 样条
    的仿射解码解析保证。
    """

    # 22 个二维自由路径坐标即 44D 无约束生成空间。
    NUM_EDGES = 22
    NUM_CONTROL_POINTS = 26

    # 只有这些字段会真正改变模型结构或输入语义。读取旧配置时，其余字段会被过滤。
    CONFIG_KEYS = frozenset({
        'n_layers',
        'n_heads',
        'd_model',
        'd_inner',
        'dropout',
        'coordinate_scale',
        'map_channels',
        'use_radial_output',
    })

    def __init__(
        self,
        n_layers=6,
        n_heads=8,
        d_model=512,
        d_inner=1024,
        dropout=0.1,
        coordinate_scale=MAP_HALF_EXTENT,
        map_channels=4,
        use_radial_output=False,
    ):
        """初始化轨迹 MeanFlow。

        Args:
            n_layers: DiT block 数量。
            n_heads: 自注意力和交叉注意力头数。
            d_model: token 特征维度。
            d_inner: 前馈网络隐藏维度。
            dropout: dropout 比例。
            coordinate_scale: 归一化坐标 1.0 对应的物理距离。
            map_channels: 4 表示三通道法向量加显式可通行 mask；mask 同时
                合并不完整观测和实体障碍，3 通道仅用于旧基线。
            use_radial_output: 是否启用旧 radial 可行化映射；主方法必须为 False。
        """
        super().__init__()
        self.coordinate_scale = float(coordinate_scale)
        self.map_channels = int(map_channels)
        self.use_radial_output = bool(use_radial_output)
        if self.coordinate_scale <= 0.0:
            raise ValueError("coordinate_scale must be positive")
        if self.map_channels not in (3, 4):
            raise ValueError("map_channels must be 3 (legacy full map) or 4 (map + mask)")
        if self.use_radial_output:
            raise ValueError(
                "The first-order boundary-constrained path representation "
                "does not support the legacy radial output map."
            )
        # 显存优化：训练时对DiT主干启用梯度检查点
        self.use_gradient_checkpoint = True
        
        # ========== 地图CNN特征提取（多尺度KV）==========
        self.map_fe_block1 = nn.Sequential(
            nn.Conv2d(self.map_channels, d_model//8, kernel_size=3, padding=1),
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

        self.reorder_dims = Rearrange('b c h w -> b (h w) c')
        self.map_position_enc = MultiScalePositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        # ========== Implicit Guidance Encoder ==========
        # 用非逐点噪声的 progress/task/time/summary queries 读取 normal map，
        # 得到与 22 个 44D state tokens 对齐的隐式 guidance tokens。
        self.guidance_encoder = ImplicitGuidanceEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_inner,
            n_path_steps=self.NUM_EDGES,
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
        # 路径位置编码（可学习）
        self.path_pos_embed = nn.Parameter(
            torch.randn(1, self.NUM_EDGES, d_model) * 0.02
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
        if self.use_radial_output:
            self.radial_slack_pred = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            # Legacy baseline: analytically maps every raw output inside the box.
            nn.init.zeros_(self.radial_slack_pred[-1].weight)
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
        self.num_edges = self.NUM_EDGES
        self.num_control_points = self.NUM_CONTROL_POINTS
        self.trajectory_representation = (
            BoundaryConstrainedPathRepresentation()
        )
        
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

    def load_compatible_state_dict(self, state_dict):
        """Only the current boundary-constrained representation is accepted."""
        self.load_state_dict(state_dict, strict=True)
        return [], [], []

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

    def validate_path_coordinates(self, path_coordinates):
        """Validate 44-D free path coordinates; no projection is required."""
        if path_coordinates.ndim != 3 or path_coordinates.shape[1:] != (
            self.num_edges,
            2,
        ):
            raise ValueError(
                f"Expected state (B,{self.num_edges},2), got "
                f"{tuple(path_coordinates.shape)}"
            )
        return path_coordinates

    def project_zero_sum(self, path_coordinates):
        """Compatibility alias; free path coordinates are unconstrained."""
        return self.validate_path_coordinates(path_coordinates)

    @staticmethod
    def _condition_pose_to_xy_yaw(condition_pose):
        """Convert normalized ``(x,y,cos,sin)`` model conditions to poses."""
        if condition_pose.ndim != 2 or condition_pose.shape[1] != 4:
            raise ValueError(
                "condition_pose must have shape (B,4), got "
                f"{tuple(condition_pose.shape)}"
            )
        yaw = torch.atan2(condition_pose[:, 3], condition_pose[:, 2])
        return torch.cat([condition_pose[:, :2], yaw[:, None]], dim=1)

    def decode_path_coordinates(
        self,
        path_coordinates,
        start_pose,
        goal_pose,
    ):
        """Decode 44-D free path coordinates to B-spline control points."""
        path_coordinates = self.validate_path_coordinates(path_coordinates)
        return self.trajectory_representation.decode_control_points(
            path_coordinates,
            self._condition_pose_to_xy_yaw(start_pose),
            self._condition_pose_to_xy_yaw(goal_pose),
        )

    def decode_residual_edges(self, residuals, start_pose, goal_pose):
        """Compatibility alias for :meth:`decode_path_coordinates`."""
        return self.decode_path_coordinates(
            residuals, start_pose, goal_pose
        )

    def evaluate_trajectory_state(self, state, start_pose, goal_pose):
        """Return physical p, p', p'', yaw and curvature analytically."""
        state = self.project_zero_sum(state)
        geometry = self.trajectory_representation.evaluate(
            state,
            self._condition_pose_to_xy_yaw(start_pose),
            self._condition_pose_to_xy_yaw(goal_pose),
        )
        scale = float(self.coordinate_scale)
        for key in (
            "control_points",
            "position",
            "first_derivative",
            "second_derivative",
            "speed",
        ):
            geometry[key] = geometry[key] * scale
        geometry["curvature"] = geometry["curvature"] / scale
        return geometry

    def audit_trajectory_state_curvature(self, state, start_pose, goal_pose):
        curvature = self.trajectory_representation.audit_curvature(
            self.validate_path_coordinates(state),
            self._condition_pose_to_xy_yaw(start_pose),
            self._condition_pose_to_xy_yaw(goal_pose),
        )
        return curvature / float(self.coordinate_scale)

    def sample_timesteps(self, n, device=None, generator=None):
        if device is None:
            device = self.t_sample_mean.device
        logits = (
            torch.randn(n, device=device, generator=generator)
            * self.t_sample_std
            + self.t_sample_mean
        )
        return torch.sigmoid(logits).clamp(1e-5, 1.0 - 1e-5)

    def audit_trajectory_state_curvature(self, state, start_pose, goal_pose):
        """Return physical curvature on the representation's 1001-point grid."""
        state = self.project_zero_sum(state)
        curvature = self.trajectory_representation.audit_curvature(
            state,
            self._condition_pose_to_xy_yaw(start_pose),
            self._condition_pose_to_xy_yaw(goal_pose),
        )
        return curvature / float(self.coordinate_scale)

    def decode_feasible_residual_edges(self, residuals, start_pose, goal_pose):
        """Compatibility alias for decoding free path coordinates."""
        return self.decode_path_coordinates(
            residuals, start_pose, goal_pose
        )

    def radial_feasible_x0(
        self,
        raw_v,
        raw_s,
        start_pose,
        goal_pose,
        return_diagnostics=False,
    ):
        raise RuntimeError(
            "Legacy radial output is incompatible with the first-order "
            "boundary-constrained path representation."
        )

    def forward(
        self,
        map_input,
        noisy_path,
        timestep,
        timestep_r,
        start_pose,
        goal_pose,
        return_features=False,
    ):
        """
        扩散模型前向传播（解耦数据流版本）- Rectified Flow
        
        【数据流架构】
        1. Path Tokens (x): 作为Query在DiT主干中流动并不断更新
        2. Map Tokens: 作为Key/Value在Cross-Attention中被查询（保持不变）
        3. Condition (c): Time + Start + Goal 融合后通过AdaLN调制每一层
        
        Args:
            map_input: (B, 4, H, W)，前三通道为法向量/遮挡高斯噪声，
                第四通道为 1=允许进入、0=禁止进入的统一 mask。
            noisy_path: (B,22,2) 加噪后的自由路径坐标
            timestep: (B,) 连续时间步 t ∈ [0, 1]
                     t=0: 纯数据, t=1: 纯噪声
            start_pose: (B, 4) 起点坐标 (x,y,cos(θ),sin(θ)) - 已归一化
            goal_pose: (B, 4) 终点坐标 (x,y,cos(θ),sin(θ)) - 已归一化
            
        Returns:
            model_output: (B,22,2) 自由路径坐标预测
        """
        B = map_input.shape[0]
        if map_input.ndim != 4 or map_input.shape[1] != self.map_channels:
            raise ValueError(
                f"Expected map_input (B,{self.map_channels},H,W), got "
                f"{tuple(map_input.shape)}"
            )
        
        # ========== 1. 编码地图 (Context, 单尺度25x25 KV) ==========
        feat_50 = self.map_fe_block1(map_input)           # (B, d_model//8, 50, 50)
        feat_25 = self.map_fe_block2(feat_50)             # (B, d_model//4, 25, 25)
        feat_12 = self.map_fe_block3(feat_25)             # (B, d_model//2, 12, 12)
        feat_12 = self.map_fe_block4(feat_12)             # (B, d_model, 12, 12)

        map_tokens = self.reorder_dims(feat_12)           # (B, 144, D)
        map_tokens = self.map_position_enc(map_tokens, conv_shape=feat_12.shape[-2:])
        
        # ========== 2. 编码 22×2 自由路径坐标 (Query Sequence) ==========
        if noisy_path.ndim != 3 or noisy_path.shape[1:] != (self.num_edges, 2):
            raise ValueError(
                f"Expected noisy_path (B,{self.num_edges},2), "
                f"got {tuple(noisy_path.shape)}"
            )

        path_tokens = self.path_patchify(noisy_path)  # (B, 22, D)
        combined_tokens = path_tokens + self.path_pos_embed
        
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
        
        # ========== 6. 无约束 44D x0 输出 ==========
        raw_v = self.main_pred(x)
        state = self.project_zero_sum(raw_v)
        if return_features:
            return state, {
                "path_tokens": x,
                "guidance_tokens": guidance_tokens,
                "map_tokens": map_tokens,
            }
        return state
    
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
        
        # pMF 采样逻辑：在 ODE 中，我们依然需要瞬时速度场 u = (z - x0) / t
        if t_scalar < 1e-5:
            return torch.zeros_like(z_in)

        return self.project_zero_sum((z_in - model_out) / t_scalar)

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

        eps = 1e-5
        return self.project_zero_sum((z_in - model_out) / (t_scalar + eps))

    def sample_differentiable(self, map_input, start_pose, goal_pose, num_steps=3,
                              solver='pmf_refined', reconstruct_trajectory=True, num_traj_points=200):
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

        geometry = self.evaluate_trajectory_state(z, start_pose, goal_pose)
        control_points_denorm = geometry["control_points"]

        if not reconstruct_trajectory:
            return control_points_denorm

        if int(num_traj_points) != self.trajectory_representation.dense_points:
            raise ValueError(
                "Analytic production sampling requires "
                f"{self.trajectory_representation.dense_points} points."
            )
        return geometry["position"]
    
    @torch.no_grad()
    def sample_pmf_onestep(
        self,
        map_input,
        start_pose,
        goal_pose,
        source_noise=None,
        return_residual=False,
    ):
        """pMF 核心：从固定源噪声单步生成目标残差。

        显式传入 ``source_noise`` 可以保持 source identity：构造后验粒子时，
        每个优化结果能够追溯到产生它的 Stage 1 proposal。
        """
        device = map_input.device
        B = map_input.shape[0]

        # 从纯噪声开始 t=1
        if source_noise is None:
            source_noise = torch.randn(
                B, self.num_edges, 2, device=device, dtype=map_input.dtype
            )
        else:
            source_noise = torch.as_tensor(
                source_noise, device=device, dtype=map_input.dtype
            )
            if source_noise.shape != (B, self.num_edges, 2):
                raise ValueError(
                    f"Expected source_noise {(B, self.num_edges, 2)}, got "
                    f"{tuple(source_noise.shape)}"
                )
        z = self.project_zero_sum(source_noise)
        
        t_batch = torch.ones(B, device=device)
        r_batch = torch.zeros(B, device=device) # 目标是 0 
        
        # 模型在 pMF 训练下，t=1, r=0 的输出即为修正后的高质量 x0
        pred_x0 = self.forward(map_input, z, t_batch, r_batch, start_pose, goal_pose)
        pred_x0 = self.project_zero_sum(pred_x0)
        if return_residual:
            return pred_x0
        return self.decode_feasible_residual_edges(pred_x0, start_pose, goal_pose)
    
    @torch.no_grad()
    def sample_pmf_refined(
        self,
        map_input,
        start_pose,
        goal_pose,
        refine_steps=3,
        t_refine_start=0.1,
        return_state=False,
    ):
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
            
        if return_state:
            return z
        return self.decode_feasible_residual_edges(z, start_pose, goal_pose)
    
    @torch.no_grad()
    def sample_euler(
        self,
        map_input,
        start_pose,
        goal_pose,
        num_steps=50,
        return_state=False,
    ):
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
            num_steps: ODE求解步数
        
        Returns:
            control_points: (B, 26, 2) 控制点（含起终点）
        """
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

        if return_state:
            return z
        return self.decode_feasible_residual_edges(z, start_pose, goal_pose)
    
    @torch.no_grad()
    def sample_heun(
        self,
        map_input,
        start_pose,
        goal_pose,
        num_steps=50,
        return_state=False,
    ):
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
        
        if return_state:
            return z
        return self.decode_feasible_residual_edges(z, start_pose, goal_pose)
    
    
    @torch.no_grad()
    def sample(self, map_input, start_pose, goal_pose, num_samples=5, num_steps=50,
               solver='heun', reconstruct_trajectory=True, num_traj_points=200,
               source_noise=None, return_residual=False):
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

        if source_noise is not None:
            source_noise = torch.as_tensor(
                source_noise, device=device, dtype=map_input.dtype
            )
            expected_batch = map_input_batch.shape[0]
            if source_noise.shape != (expected_batch, self.num_edges, 2):
                raise ValueError(
                    f"Expected source_noise {(expected_batch, self.num_edges, 2)}, "
                    f"got {tuple(source_noise.shape)}"
                )
        
        # 选择ODE求解器
        if solver == 'pmf_onestep':
            state = self.sample_pmf_onestep(
                map_input_batch,
                start_pose_batch,
                goal_pose_batch,
                source_noise=source_noise,
                return_residual=True,
            )
        elif solver == 'pmf_refined':
            state = self.sample_pmf_refined(
                map_input_batch,
                start_pose_batch,
                goal_pose_batch,
                refine_steps=num_steps,
                t_refine_start=0.1,
                return_state=True,
            )
        elif solver == 'euler':
            state = self.sample_euler(
                map_input_batch,
                start_pose_batch,
                goal_pose_batch,
                num_steps=num_steps,
                return_state=True,
            )
        elif solver == 'heun':
            state = self.sample_heun(
                map_input_batch,
                start_pose_batch,
                goal_pose_batch,
                num_steps=num_steps,
                return_state=True,
            )
        else:
            raise ValueError(f"Unknown solver: {solver}. Choose 'pmf_onestep', 'euler' or 'heun'.")

        if return_residual:
            return state

        geometry = self.evaluate_trajectory_state(
            state, start_pose_batch, goal_pose_batch
        )
        control_points_denorm = geometry["control_points"]
        
        if not reconstruct_trajectory:
            return control_points_denorm
        
        if int(num_traj_points) != self.trajectory_representation.dense_points:
            raise ValueError(
                "Analytic production sampling uses "
                f"{self.trajectory_representation.dense_points} points; got "
                f"{num_traj_points}."
            )
        return geometry["position"]


# Canonical production model name. The subclass does not add parameters or
# change computation; it only separates the current Path MeanFlow method from
# unrelated diffusion-era baselines that still import the legacy class name.
class CompactAttention(nn.Module):
    """Explicit multi-head attention compatible with torch.func.jvp."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch, query_length, width = query.shape
        key_length = key.shape[1]

        def split_heads(tensor):
            return tensor.view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(query))
        k = split_heads(self.k_proj(key))
        v = split_heads(self.v_proj(value))
        weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = self.dropout(torch.softmax(weights, dim=-1))
        output = torch.matmul(weights, v).transpose(1, 2).contiguous()
        return self.out_proj(output.view(batch, query_length, width))


class CompactPathBlock(nn.Module):
    """Small path-only DiT block used by the compact architecture."""

    def __init__(self, d_model, n_heads, d_inner, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = CompactAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Start as an identity map, as in AdaLN-Zero.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, condition):
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            self.adaLN_modulation(condition).chunk(6, dim=-1)
        )
        normalized = self.norm1(x)
        normalized = normalized * (1.0 + scale_attn[:, None]) + shift_attn[:, None]
        attended = self.self_attn(normalized, normalized, normalized)
        x = x + gate_attn[:, None] * attended

        normalized = self.norm2(x)
        normalized = normalized * (1.0 + scale_ffn[:, None]) + shift_ffn[:, None]
        x = x + gate_ffn[:, None] * self.ffn(normalized)
        return x


class CompactPathMeanFlowTransformer(nn.Module):
    """Compact conditional Path MeanFlow generator.

    This class keeps the established 44-D path interface and deployment endpoint,
    while using one spatial map read followed by a small path-only DiT backbone.
    It is intentionally separate from ``PathMeanFlowTransformer`` so existing
    checkpoints remain immutable and cannot be loaded with incompatible weights.
    """

    NUM_EDGES = PathDiffusionTransformer.NUM_EDGES
    NUM_CONTROL_POINTS = PathDiffusionTransformer.NUM_CONTROL_POINTS
    CONFIG_KEYS = frozenset({
        "n_layers", "n_heads", "d_model", "d_inner", "dropout",
        "coordinate_scale", "map_channels", "use_radial_output",
    })
    ARCHITECTURE_NAME = "compact"

    def __init__(
        self,
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_inner=768,
        dropout=0.1,
        coordinate_scale=MAP_HALF_EXTENT,
        map_channels=4,
        use_radial_output=False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if map_channels not in (3, 4):
            raise ValueError("map_channels must be 3 or 4")
        if use_radial_output:
            raise ValueError("compact does not support radial output")
        self.coordinate_scale = float(coordinate_scale)
        self.map_channels = int(map_channels)
        self.use_radial_output = False
        self.use_gradient_checkpoint = True

        def map_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(),
            )

        self.map_encoder = nn.Sequential(
            map_block(self.map_channels, d_model // 4),
            map_block(d_model // 4, d_model // 2),
            map_block(d_model // 2, d_model),
        )
        self.map_position_enc = MultiScalePositionalEncoding(d_model)
        self.map_norm = nn.LayerNorm(d_model)
        self.path_patchify = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.path_pos_embed = nn.Parameter(
            torch.randn(1, self.NUM_EDGES, d_model) * 0.02
        )
        self.map_cross_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.map_cross_attn = CompactAttention(d_model, n_heads, dropout)
        self.time_embedder = TimestepEmbedder(d_model)
        self.pose_embedder = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.path_blocks = nn.ModuleList([
            CompactPathBlock(d_model, n_heads, d_inner, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        self.num_edges = self.NUM_EDGES
        self.num_control_points = self.NUM_CONTROL_POINTS
        self.trajectory_representation = BoundaryConstrainedPathRepresentation()
        self.register_buffer("t_sample_mean", torch.tensor(-0.8))
        self.register_buffer("t_sample_std", torch.tensor(0.8))

    def validate_path_coordinates(self, state):
        if state.ndim != 3 or state.shape[1:] != (self.NUM_EDGES, 2):
            raise ValueError(
                f"Expected state (B,{self.NUM_EDGES},2), got {tuple(state.shape)}"
            )
        return state

    def project_zero_sum(self, state):
        # Compatibility name retained; current free coordinates are unconstrained.
        return self.validate_path_coordinates(state)

    @staticmethod
    def _condition_pose_to_xy_yaw(condition_pose):
        if condition_pose.ndim != 2 or condition_pose.shape[1] != 4:
            raise ValueError("condition_pose must have shape (B,4)")
        yaw = torch.atan2(condition_pose[:, 3], condition_pose[:, 2])
        return torch.cat([condition_pose[:, :2], yaw[:, None]], dim=1)

    def decode_path_coordinates(self, state, start_pose, goal_pose):
        return self.trajectory_representation.decode_control_points(
            self.validate_path_coordinates(state),
            self._condition_pose_to_xy_yaw(start_pose),
            self._condition_pose_to_xy_yaw(goal_pose),
        )

    def evaluate_trajectory_state(self, state, start_pose, goal_pose):
        geometry = self.trajectory_representation.evaluate(
            self.validate_path_coordinates(state),
            self._condition_pose_to_xy_yaw(start_pose),
            self._condition_pose_to_xy_yaw(goal_pose),
        )
        scale = float(self.coordinate_scale)
        for key in ("control_points", "position", "first_derivative", "second_derivative", "speed"):
            geometry[key] = geometry[key] * scale
        geometry["curvature"] = geometry["curvature"] / scale
        return geometry

    def audit_trajectory_state_curvature(self, state, start_pose, goal_pose):
        curvature = self.trajectory_representation.audit_curvature(
            self.validate_path_coordinates(state),
            self._condition_pose_to_xy_yaw(start_pose),
            self._condition_pose_to_xy_yaw(goal_pose),
        )
        return curvature / float(self.coordinate_scale)

    def sample_timesteps(self, n, device=None, generator=None):
        if device is None:
            device = self.t_sample_mean.device
        logits = (
            torch.randn(n, device=device, generator=generator)
            * self.t_sample_std
            + self.t_sample_mean
        )
        return torch.sigmoid(logits).clamp(1e-5, 1.0 - 1e-5)

    def forward(self, map_input, noisy_path, timestep, timestep_r, start_pose, goal_pose, return_features=False):
        if map_input.ndim != 4 or map_input.shape[1] != self.map_channels:
            raise ValueError(
                f"Expected map_input (B,{self.map_channels},H,W), got {tuple(map_input.shape)}"
            )
        state = self.validate_path_coordinates(noisy_path)
        map_features = self.map_encoder(map_input)
        map_tokens = rearrange(map_features, "b c h w -> b (h w) c")
        map_tokens = self.map_position_enc(map_tokens, map_features.shape[-2:])
        map_tokens = self.map_norm(map_tokens)

        path_tokens = self.path_patchify(state) + self.path_pos_embed
        path_queries = self.map_cross_norm(path_tokens)
        map_context = self.map_cross_attn(path_queries, map_tokens, map_tokens)
        x = path_tokens + map_context

        t_emb = self.time_embedder(timestep)
        h_emb = self.time_embedder(torch.clamp(timestep - timestep_r, 0.0, 1.0))
        pose_emb = self.pose_embedder(torch.cat([start_pose, goal_pose], dim=-1))
        condition = self.cond_mlp(torch.cat([t_emb, h_emb, pose_emb], dim=-1))
        for block in self.path_blocks:
            if self.training and self.use_gradient_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, condition, use_reentrant=False
                )
            else:
                x = block(x, condition)
        output = self.output_head(self.final_norm(x))
        if return_features:
            return output, {"path_tokens": x, "map_tokens": map_tokens}
        return output

    def q_sample(self, x_start, t, noise=None):
        x_start = self.validate_path_coordinates(x_start)
        noise = torch.randn_like(x_start) if noise is None else self.validate_path_coordinates(noise)
        t = t.view(-1, 1, 1)
        return (1.0 - t) * x_start + t * noise, noise

    @torch.no_grad()
    def sample_pmf_onestep(self, map_input, start_pose, goal_pose, source_noise=None, return_residual=False):
        if source_noise is None:
            source_noise = torch.randn(
                map_input.shape[0], self.NUM_EDGES, 2,
                device=map_input.device, dtype=map_input.dtype,
            )
        source_noise = self.validate_path_coordinates(source_noise)
        state = self(
            map_input, source_noise,
            torch.ones(map_input.shape[0], device=map_input.device),
            torch.zeros(map_input.shape[0], device=map_input.device),
            start_pose, goal_pose,
        )
        if return_residual:
            return state
        return self.decode_path_coordinates(state, start_pose, goal_pose)


class SpatialMapTrajBlock(nn.Module):
    """JVP-compatible SA/CA/FFN block for the spatial-map condition test."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = CompactAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.map_cross_attn = CompactAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, map_tokens, condition):
        (
            shift_sa, scale_sa, gate_sa,
            shift_ca, scale_ca, gate_ca,
            shift_ffn, scale_ffn, gate_ffn,
        ) = self.adaLN_modulation(condition).chunk(9, dim=-1)
        normalized = self.norm1(x)
        normalized = normalized * (1.0 + scale_sa[:, None]) + shift_sa[:, None]
        x = x + gate_sa[:, None] * self.self_attn(normalized, normalized, normalized)

        normalized = self.norm2(x)
        normalized = normalized * (1.0 + scale_ca[:, None]) + shift_ca[:, None]
        x = x + gate_ca[:, None] * self.map_cross_attn(
            normalized, map_tokens, map_tokens
        )

        normalized = self.norm3(x)
        normalized = normalized * (1.0 + scale_ffn[:, None]) + shift_ffn[:, None]
        x = x + gate_ffn[:, None] * self.ffn(normalized)
        return x


class SpatialMapPathMeanFlowTransformer(PathDiffusionTransformer):
    """Production-width Path MeanFlow with direct spatial map conditioning.

    The path tokens query the position-encoded CNN map tokens in every
    ``TrajDiTBlock``. Start/goal and time remain a single global AdaLN
    condition. This removes the progress-indexed guidance bottleneck while
    keeping the established path representation and deployment endpoint.

    ``map_layout`` changes only the static map memory supplied to each
    cross-attention layer. It does not change the path/query stream or the
    global condition stream:

    * ``single_12``: the established 12x12 map memory;
    * ``single_25``: the same CNN stack without the final 25->12 pooling;
    * ``multi``: concatenated 25x25 and 12x12 memories with 2-D and scale
      embeddings.
    """

    ARCHITECTURE_NAME = "spatial_map"
    MAP_LAYOUTS = ("single_12", "single_25", "multi")
    OUTPUT_HEAD_TYPE = "tokenwise_22x2"

    def __init__(self, *args, map_layout="single_12", **kwargs):
        if map_layout not in self.MAP_LAYOUTS:
            raise ValueError(
                f"Unknown map_layout {map_layout!r}; "
                f"choose from {self.MAP_LAYOUTS}"
            )
        super().__init__(*args, **kwargs)
        self.map_layout = map_layout
        # The experiment isolates the conditioning change: the production
        # backbone and map encoder remain unchanged, while the guidance query
        # encoder is removed from the trainable graph.
        del self.guidance_encoder
        self.dit_blocks = nn.ModuleList([
            SpatialMapTrajBlock(
                d_model=self.dit_blocks[0].norm1.normalized_shape[0],
                n_heads=self.dit_blocks[0].n_heads,
                d_ff=self.dit_blocks[0].ffn[0].out_features,
                dropout=self.dropout.p,
            )
            for _ in self.dit_blocks
        ])
        if self.map_layout == "multi":
            d_model = self.dit_blocks[0].norm1.normalized_shape[0]
            self.map_scale_embed = nn.Parameter(torch.zeros(2, d_model))
            nn.init.normal_(self.map_scale_embed, std=0.02)

    def _encode_map_levels(self, map_input):
        """Return the static CNN feature levels used by ``map_layout``.

        The first three operations of ``map_fe_block3`` are kept separate from
        its final pooling operation so the 25x25 feature can be reused without
        changing the convolutional channel schedule.
        """
        feat_50 = self.map_fe_block1(map_input)
        feat_25_pre = self.map_fe_block2(feat_50)
        feat_25_mid = self.map_fe_block3[0](feat_25_pre)
        feat_25_mid = self.map_fe_block3[1](feat_25_mid)
        feat_25_mid = self.map_fe_block3[2](feat_25_mid)

        if self.map_layout == "single_25":
            return {"25": self.map_fe_block4(feat_25_mid)}

        feat_12_mid = self.map_fe_block3[3](feat_25_mid)
        feat_12 = self.map_fe_block4(feat_12_mid)
        if self.map_layout == "single_12":
            return {"12": feat_12}

        # The two levels share the established block-4 projection. This adds
        # no new trainable map encoder and keeps the multi-scale variable in
        # the memory, rather than in the path transformer.
        feat_25 = self.map_fe_block4(feat_25_mid)
        return {"25": feat_25, "12": feat_12}

    def _build_map_memory(self, map_input):
        levels = self._encode_map_levels(map_input)
        memories = []
        for scale_name, features in levels.items():
            tokens = self.reorder_dims(features)
            tokens = self.map_position_enc(
                tokens, conv_shape=features.shape[-2:]
            )
            if self.map_layout == "multi":
                scale_index = 0 if scale_name == "25" else 1
                tokens = tokens + self.map_scale_embed[scale_index].view(
                    1, 1, -1
                )
            memories.append(tokens)
        return torch.cat(memories, dim=1)

    def forward(
        self,
        map_input,
        noisy_path,
        timestep,
        timestep_r,
        start_pose,
        goal_pose,
        return_features=False,
    ):
        if map_input.ndim != 4 or map_input.shape[1] != self.map_channels:
            raise ValueError(
                f"Expected map_input (B,{self.map_channels},H,W), got "
                f"{tuple(map_input.shape)}"
            )
        map_tokens = self._build_map_memory(map_input)

        if noisy_path.ndim != 3 or noisy_path.shape[1:] != (self.num_edges, 2):
            raise ValueError(
                f"Expected noisy_path (B,{self.num_edges},2), got "
                f"{tuple(noisy_path.shape)}"
            )
        path_tokens = self.path_patchify(noisy_path)
        x = path_tokens + self.path_pos_embed
        x = self.layer_norm(x)
        x = self.dropout(x)

        t_emb = self.time_embedder(timestep)
        h_emb = self.time_embedder(
            torch.clamp(timestep - timestep_r, min=0.0, max=1.0)
        )
        s_emb = self.pose_embedder(start_pose)
        g_emb = self.pose_embedder(goal_pose)
        condition = self.cond_mlp(
            torch.cat([t_emb, h_emb, s_emb, g_emb], dim=-1)
        )

        for block in self.dit_blocks:
            if self.training and self.use_gradient_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, map_tokens, condition, use_reentrant=False
                )
            else:
                x = block(x, map_tokens, condition)
        x = self.layer_norm(x)
        state = self.project_zero_sum(self.main_pred(x))
        if return_features:
            features = {
                "path_tokens": x,
                "map_tokens": map_tokens,
            }
            return state, features
        return state


class CoordinateSincosEncoding(nn.Module):
    """Learnable-free 2-D coordinate encoding for arbitrary task frames.

    Unlike ``MultiScalePositionalEncoding``, this class accepts continuous
    ``(x, y)`` coordinates instead of raster grid indices.  Frequencies are
    log-spaced up to ``max_freq`` and the output dimension is ``d_model``.
    """

    def __init__(self, d_model, max_freq=16.0):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = int(d_model)
        self.max_freq = float(max_freq)
        if self.max_freq <= 0.0:
            raise ValueError("max_freq must be positive")
        d_quarter = self.d_model // 4
        if d_quarter > 0:
            frequencies = torch.exp(
                torch.linspace(0.0, math.log(self.max_freq), d_quarter)
            )
        else:
            frequencies = torch.empty(0)
        self.register_buffer("frequencies", frequencies)

    def forward(self, coordinates):
        if coordinates.ndim < 2 or coordinates.shape[-1] != 2:
            raise ValueError(
                "coordinates must end in 2, got "
                f"{tuple(coordinates.shape)}"
            )
        x = coordinates[..., 0:1]
        y = coordinates[..., 1:2]
        frequencies = self.frequencies.to(
            device=coordinates.device, dtype=coordinates.dtype
        )
        phases_x = x * frequencies
        phases_y = y * frequencies
        pieces = [
            torch.sin(phases_x),
            torch.cos(phases_x),
            torch.sin(phases_y),
            torch.cos(phases_y),
        ]
        encoding = torch.cat(pieces, dim=-1)
        width = encoding.shape[-1]
        if width < self.d_model:
            padding = torch.zeros(
                *encoding.shape[:-1],
                self.d_model - width,
                device=coordinates.device,
                dtype=coordinates.dtype,
            )
            encoding = torch.cat([encoding, padding], dim=-1)
        return encoding


class DualConditionSpatialTrajBlock(nn.Module):
    """Spatial-map block with separated time and task AdaLN branches.

    ``condition_fusion`` selects how the two semantic conditions produce the
    per-sub-layer modulation vector:

    * ``additive``: ``mod_time(c_time) + mod_task(c_task)`` (default);
    * ``joint_mlp``: one MLP over ``[c_time, c_task]``;
    * ``additive_with_interaction``: additive base plus a zero-initialized
      joint interaction head.

    All modulation heads use AdaLN-Zero initialization.
    """

    FUSIONS = ("additive", "joint_mlp", "additive_with_interaction")

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout=0.1,
        condition_fusion="additive",
    ):
        super().__init__()
        if condition_fusion not in self.FUSIONS:
            raise ValueError(
                f"Unknown condition_fusion {condition_fusion!r}; "
                f"choose from {self.FUSIONS}"
            )
        self.condition_fusion = condition_fusion
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = CompactAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.map_cross_attn = CompactAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.adaLN_mod_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model),
        )
        self.adaLN_mod_task = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model),
        )
        if self.condition_fusion in (
            "joint_mlp",
            "additive_with_interaction",
        ):
            self.adaLN_mod_joint = nn.Sequential(
                nn.SiLU(),
                nn.Linear(2 * d_model, 9 * d_model),
            )
        for head in (self.adaLN_mod_time, self.adaLN_mod_task):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        if hasattr(self, "adaLN_mod_joint"):
            nn.init.zeros_(self.adaLN_mod_joint[-1].weight)
            nn.init.zeros_(self.adaLN_mod_joint[-1].bias)

    def forward(self, x, map_tokens, time_condition, task_condition=None):
        time_mod = self.adaLN_mod_time(time_condition)
        task_mod = time_mod.new_zeros(time_mod.shape)
        if task_condition is not None:
            task_mod = self.adaLN_mod_task(task_condition)
        if self.condition_fusion == "additive":
            modulation = time_mod + task_mod
        elif self.condition_fusion == "joint_mlp":
            if task_condition is None:
                raise ValueError(
                    "joint_mlp requires a task condition tensor"
                )
            modulation = self.adaLN_mod_joint(
                torch.cat([time_condition, task_condition], dim=-1)
            )
        elif self.condition_fusion == "additive_with_interaction":
            if task_condition is None:
                raise ValueError(
                    "additive_with_interaction requires a task condition tensor"
                )
            modulation = (
                time_mod
                + task_mod
                + self.adaLN_mod_joint(
                    torch.cat([time_condition, task_condition], dim=-1)
                )
            )
        else:  # pragma: no cover - guarded by __init__
            raise RuntimeError(self.condition_fusion)

        (
            shift_sa, scale_sa, gate_sa,
            shift_ca, scale_ca, gate_ca,
            shift_ffn, scale_ffn, gate_ffn,
        ) = modulation.chunk(9, dim=-1)
        normalized = self.norm1(x)
        normalized = normalized * (1.0 + scale_sa[:, None]) + shift_sa[:, None]
        x = x + gate_sa[:, None] * self.self_attn(
            normalized, normalized, normalized
        )

        normalized = self.norm2(x)
        normalized = normalized * (1.0 + scale_ca[:, None]) + shift_ca[:, None]
        x = x + gate_ca[:, None] * self.map_cross_attn(
            normalized, map_tokens, map_tokens
        )

        normalized = self.norm3(x)
        normalized = normalized * (1.0 + scale_ffn[:, None]) + shift_ffn[:, None]
        x = x + gate_ffn[:, None] * self.ffn(normalized)
        return x


class TokenAlignedSpatialPathMeanFlowTransformer(
    SpatialMapPathMeanFlowTransformer
):
    """Tier-A token-level alignment candidate.

    This is **not** full task-frame canonicalization: the map raster keeps its
    original orientation and metric receptive fields.  Only three geometric
    responsibilities are made explicit:

    1. horizontal normal channels are rotated into the canonical basis;
    2. map tokens receive a continuous coordinate encoding of their
       chord-normalized task-frame positions ``p_c = d^-1 R(-theta)(p - S)``;
    3. every path token keeps its index embedding and additionally receives
       the coordinate encoding of its current canonical free control point
       ``C_free(z_t; u_s, u_g)``.

    Raw start/goal tensors never enter a learnable module.  Time and task
    conditions use separate AdaLN branches.  The analytic B-spline decoder and
    the 44-D representation are unchanged.
    """

    ARCHITECTURE_NAME = "token_aligned"
    MAP_LAYOUTS = ("single_12",)
    OUTPUT_HEAD_TYPE = "tokenwise_22x2"
    TASK_COND_MODES = ("none", "scale", "direction", "direction_scale")
    CONDITION_FUSIONS = DualConditionSpatialTrajBlock.FUSIONS
    # Feature centers are computed on this normalized source raster.
    MAP_FEATURE_CENTER_BOUNDS_NORM = (-1.0, 0.98, -1.0, 0.98)
    CONFIG_KEYS = PathDiffusionTransformer.CONFIG_KEYS | frozenset({
        "coord_pe_max_freq",
        "task_cond_mode",
        "condition_fusion",
    })

    def __init__(
        self,
        *args,
        map_layout="single_12",
        coord_pe_max_freq=16.0,
        task_cond_mode="direction_scale",
        condition_fusion="additive",
        **kwargs,
    ):
        if map_layout != "single_12":
            raise ValueError(
                "TokenAlignedSpatialPathMeanFlowTransformer currently "
                "supports only map_layout='single_12'"
            )
        if task_cond_mode not in self.TASK_COND_MODES:
            raise ValueError(
                f"Unknown task_cond_mode {task_cond_mode!r}; "
                f"choose from {self.TASK_COND_MODES}"
            )
        if condition_fusion not in DualConditionSpatialTrajBlock.FUSIONS:
            raise ValueError(
                f"Unknown condition_fusion {condition_fusion!r}; "
                f"choose from {DualConditionSpatialTrajBlock.FUSIONS}"
            )
        if task_cond_mode == "none" and condition_fusion != "additive":
            raise ValueError(
                "task_cond_mode='none' is only meaningful with "
                "condition_fusion='additive'"
            )
        super().__init__(*args, map_layout=map_layout, **kwargs)
        self.coord_pe_max_freq = float(coord_pe_max_freq)
        self.task_cond_mode = task_cond_mode
        self.condition_fusion = condition_fusion

        d_model = self.dit_blocks[0].norm1.normalized_shape[0]
        n_heads = self.dit_blocks[0].map_cross_attn.n_heads
        d_ff = self.dit_blocks[0].ffn[0].out_features
        self.dit_blocks = nn.ModuleList([
            DualConditionSpatialTrajBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=self.dropout.p,
                condition_fusion=self.condition_fusion,
            )
            for _ in self.dit_blocks
        ])
        self.coord_pe = CoordinateSincosEncoding(
            d_model, max_freq=self.coord_pe_max_freq
        )
        self.time_condition_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        task_input_dim = {
            "none": 0,
            "scale": 1,
            "direction": 4,
            "direction_scale": 5,
        }[self.task_cond_mode]
        if task_input_dim > 0:
            self.task_condition_mlp = nn.Sequential(
                nn.Linear(task_input_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        # Unused inherited single-vector condition modules are removed so the
        # parameter budget and checkpoint state match this architecture's
        # actual information flow.
        del self.cond_mlp
        del self.pose_embedder
        if hasattr(self, "map_scale_embed"):
            del self.map_scale_embed
        self.register_buffer(
            "map_feature_centers_norm",
            self._build_map_feature_centers_norm(),
        )

    def _build_map_feature_centers_norm(self):
        xmin, xmax, ymin, ymax = self.MAP_FEATURE_CENTER_BOUNDS_NORM
        x = torch.linspace(xmin, xmax, 12)
        y = torch.linspace(ymin, ymax, 12)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(12 * 12, 2)

    @staticmethod
    def _wrap_angle(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @staticmethod
    def _rotate_horizontal_normals(map_input, theta):
        rotated = map_input.clone()
        cosine = torch.cos(theta).view(-1, 1, 1)
        sine = torch.sin(theta).view(-1, 1, 1)
        nx = map_input[:, 0]
        ny = map_input[:, 1]
        rotated[:, 0] = cosine * nx + sine * ny
        rotated[:, 1] = -sine * nx + cosine * ny
        return rotated

    def _task_frame(self, start_pose, goal_pose):
        """Return d_norm, theta, and canonical endpoint directions."""
        start_yaw = torch.atan2(start_pose[:, 3], start_pose[:, 2])
        goal_yaw = torch.atan2(goal_pose[:, 3], goal_pose[:, 2])
        delta = goal_pose[:, :2] - start_pose[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=1)
        if torch.any(distance <= 1e-8):
            raise ValueError("Start and goal positions must differ")
        theta = torch.atan2(delta[:, 1], delta[:, 0])
        start_direction = torch.stack(
            [
                torch.cos(self._wrap_angle(start_yaw - theta)),
                torch.sin(self._wrap_angle(start_yaw - theta)),
            ],
            dim=1,
        )
        goal_direction = torch.stack(
            [
                torch.cos(self._wrap_angle(goal_yaw - theta)),
                torch.sin(self._wrap_angle(goal_yaw - theta)),
            ],
            dim=1,
        )
        return distance, theta, start_direction, goal_direction

    def _canonical_xy(self, points, start_pose, distance, theta):
        relative = points - start_pose[:, :2].unsqueeze(1)
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        x = (cosine[:, None] * relative[..., 0]
             + sine[:, None] * relative[..., 1])
        y = (-sine[:, None] * relative[..., 0]
             + cosine[:, None] * relative[..., 1])
        scale = distance.clamp_min(1e-6).view(-1, 1)
        return torch.stack([x / scale, y / scale], dim=-1)

    def _task_condition_vector(
        self, distance, start_direction, goal_direction
    ):
        if self.task_cond_mode == "none":
            return None
        pieces = []
        if self.task_cond_mode in ("direction", "direction_scale"):
            pieces.extend([start_direction, goal_direction])
        if self.task_cond_mode in ("scale", "direction_scale"):
            pieces.append(torch.log(distance.clamp_min(1e-6)).view(-1, 1))
        return torch.cat(pieces, dim=1)

    def _build_map_memory(self, map_input, distance, theta, start_pose):
        levels = self._encode_map_levels(map_input)
        if set(levels) != {"12"}:
            raise ValueError(
                "TokenAligned map memory expects only the single_12 level"
            )
        features = levels["12"]
        tokens = self.reorder_dims(features)
        centers = self.map_feature_centers_norm.to(
            device=tokens.device, dtype=tokens.dtype
        )
        canonical_centers = self._canonical_xy(
            centers.unsqueeze(0).expand(tokens.shape[0], -1, -1),
            start_pose,
            distance,
            theta,
        )
        tokens = tokens + self.coord_pe(canonical_centers)
        return tokens, canonical_centers

    def forward(
        self,
        map_input,
        noisy_path,
        timestep,
        timestep_r,
        start_pose,
        goal_pose,
        return_features=False,
    ):
        if map_input.ndim != 4 or map_input.shape[1] != self.map_channels:
            raise ValueError(
                f"Expected map_input (B,{self.map_channels},H,W), got "
                f"{tuple(map_input.shape)}"
            )
        if noisy_path.ndim != 3 or noisy_path.shape[1:] != (self.num_edges, 2):
            raise ValueError(
                f"Expected noisy_path (B,{self.num_edges},2), got "
                f"{tuple(noisy_path.shape)}"
            )

        distance, theta, start_direction, goal_direction = self._task_frame(
            start_pose, goal_pose
        )
        rotated_map = self._rotate_horizontal_normals(map_input, theta)
        map_tokens, canonical_map_centers = self._build_map_memory(
            rotated_map, distance, theta, start_pose
        )

        free_controls = self.trajectory_representation.canonical_control_points(
            self.validate_path_coordinates(noisy_path),
            start_direction,
            goal_direction,
        )[:, 2:24]
        path_coord_pe = self.coord_pe(free_controls)
        path_tokens = (
            self.path_patchify(noisy_path)
            + self.path_pos_embed
            + path_coord_pe
        )
        x = self.layer_norm(path_tokens)
        x = self.dropout(x)

        t_emb = self.time_embedder(timestep)
        h_emb = self.time_embedder(
            torch.clamp(timestep - timestep_r, min=0.0, max=1.0)
        )
        time_condition = self.time_condition_mlp(
            torch.cat([t_emb, h_emb], dim=-1)
        )
        task_vector = self._task_condition_vector(
            distance, start_direction, goal_direction
        )
        task_condition = (
            None
            if task_vector is None
            else self.task_condition_mlp(task_vector)
        )

        for block in self.dit_blocks:
            if self.training and self.use_gradient_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    map_tokens,
                    time_condition,
                    task_condition,
                    use_reentrant=False,
                )
            else:
                x = block(x, map_tokens, time_condition, task_condition)

        x = self.layer_norm(x)
        state = self.project_zero_sum(self.main_pred(x))
        if return_features:
            features = {
                "path_tokens": x,
                "map_tokens": map_tokens,
                "canonical_map_centers": canonical_map_centers,
                "path_free_controls": free_controls,
                "time_condition": time_condition,
                "task_condition": task_condition,
                "task_frame": {
                    "distance": distance,
                    "theta": theta,
                    "start_direction": start_direction,
                    "goal_direction": goal_direction,
                },
            }
            return state, features
        return state


class AlignmentOnlySpatialPathMeanFlowTransformer(
    SpatialMapPathMeanFlowTransformer
):
    """Alignment-only Phase-2 arm.

    Keeps the ``single_12`` condition path exactly as-is (raw S/G + time in
    one ``cond_mlp``, one AdaLN head per sub-layer) and only adds:

    * horizontal normal-channel rotation into the canonical basis;
    * canonical coordinate PE for map tokens (in addition to the original
      grid-index PE);
    * canonical coordinate PE for path tokens (in addition to the original
      learned index PE).

    This isolates the path-map geometric address question from condition
    decoupling.  The class adds no learnable parameters beyond the baseline
    ``SpatialMapPathMeanFlowTransformer``.
    """

    ARCHITECTURE_NAME = "alignment_only"
    MAP_LAYOUTS = ("single_12",)
    OUTPUT_HEAD_TYPE = "tokenwise_22x2"
    MAP_FEATURE_CENTER_BOUNDS_NORM = (-1.0, 0.98, -1.0, 0.98)
    CONFIG_KEYS = PathDiffusionTransformer.CONFIG_KEYS | frozenset({
        "coord_pe_max_freq",
    })

    def __init__(
        self,
        *args,
        map_layout="single_12",
        coord_pe_max_freq=16.0,
        **kwargs,
    ):
        if map_layout != "single_12":
            raise ValueError(
                "AlignmentOnlySpatialPathMeanFlowTransformer currently "
                "supports only map_layout='single_12'"
            )
        super().__init__(*args, map_layout=map_layout, **kwargs)
        self.coord_pe_max_freq = float(coord_pe_max_freq)
        d_model = self.dit_blocks[0].norm1.normalized_shape[0]
        self.coord_pe = CoordinateSincosEncoding(
            d_model, max_freq=self.coord_pe_max_freq
        )
        self.register_buffer(
            "map_feature_centers_norm",
            self._build_map_feature_centers_norm(),
        )

    def _build_map_feature_centers_norm(self):
        xmin, xmax, ymin, ymax = self.MAP_FEATURE_CENTER_BOUNDS_NORM
        x = torch.linspace(xmin, xmax, 12)
        y = torch.linspace(ymin, ymax, 12)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(12 * 12, 2)

    @staticmethod
    def _wrap_angle(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @staticmethod
    def _rotate_horizontal_normals(map_input, theta):
        rotated = map_input.clone()
        cosine = torch.cos(theta).view(-1, 1, 1)
        sine = torch.sin(theta).view(-1, 1, 1)
        nx = map_input[:, 0]
        ny = map_input[:, 1]
        rotated[:, 0] = cosine * nx + sine * ny
        rotated[:, 1] = -sine * nx + cosine * ny
        return rotated

    def _task_frame(self, start_pose, goal_pose):
        start_yaw = torch.atan2(start_pose[:, 3], start_pose[:, 2])
        goal_yaw = torch.atan2(goal_pose[:, 3], goal_pose[:, 2])
        delta = goal_pose[:, :2] - start_pose[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=1)
        if torch.any(distance <= 1e-8):
            raise ValueError("Start and goal positions must differ")
        theta = torch.atan2(delta[:, 1], delta[:, 0])
        start_direction = torch.stack(
            [
                torch.cos(self._wrap_angle(start_yaw - theta)),
                torch.sin(self._wrap_angle(start_yaw - theta)),
            ],
            dim=1,
        )
        goal_direction = torch.stack(
            [
                torch.cos(self._wrap_angle(goal_yaw - theta)),
                torch.sin(self._wrap_angle(goal_yaw - theta)),
            ],
            dim=1,
        )
        return distance, theta, start_direction, goal_direction

    def _canonical_xy(self, points, start_pose, distance, theta):
        relative = points - start_pose[:, :2].unsqueeze(1)
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        x = (cosine[:, None] * relative[..., 0]
             + sine[:, None] * relative[..., 1])
        y = (-sine[:, None] * relative[..., 0]
             + cosine[:, None] * relative[..., 1])
        scale = distance.clamp_min(1e-6).view(-1, 1)
        return torch.stack([x / scale, y / scale], dim=-1)

    def forward(
        self,
        map_input,
        noisy_path,
        timestep,
        timestep_r,
        start_pose,
        goal_pose,
        return_features=False,
    ):
        if map_input.ndim != 4 or map_input.shape[1] != self.map_channels:
            raise ValueError(
                f"Expected map_input (B,{self.map_channels},H,W), got "
                f"{tuple(map_input.shape)}"
            )
        if noisy_path.ndim != 3 or noisy_path.shape[1:] != (self.num_edges, 2):
            raise ValueError(
                f"Expected noisy_path (B,{self.num_edges},2), got "
                f"{tuple(noisy_path.shape)}"
            )

        distance, theta, start_direction, goal_direction = self._task_frame(
            start_pose, goal_pose
        )
        rotated_map = self._rotate_horizontal_normals(map_input, theta)
        map_tokens = self._build_map_memory(rotated_map)
        centers = self.map_feature_centers_norm.to(
            device=map_tokens.device, dtype=map_tokens.dtype
        )
        canonical_map_centers = self._canonical_xy(
            centers.unsqueeze(0).expand(map_tokens.shape[0], -1, -1),
            start_pose,
            distance,
            theta,
        )
        map_tokens = map_tokens + self.coord_pe(canonical_map_centers)

        free_controls = self.trajectory_representation.canonical_control_points(
            self.validate_path_coordinates(noisy_path),
            start_direction,
            goal_direction,
        )[:, 2:24]
        path_coord_pe = self.coord_pe(free_controls)
        path_tokens = (
            self.path_patchify(noisy_path)
            + self.path_pos_embed
            + path_coord_pe
        )
        x = self.layer_norm(path_tokens)
        x = self.dropout(x)

        t_emb = self.time_embedder(timestep)
        h_emb = self.time_embedder(
            torch.clamp(timestep - timestep_r, min=0.0, max=1.0)
        )
        s_emb = self.pose_embedder(start_pose)
        g_emb = self.pose_embedder(goal_pose)
        condition = self.cond_mlp(
            torch.cat([t_emb, h_emb, s_emb, g_emb], dim=-1)
        )

        for block in self.dit_blocks:
            if self.training and self.use_gradient_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, map_tokens, condition, use_reentrant=False
                )
            else:
                x = block(x, map_tokens, condition)
        x = self.layer_norm(x)
        state = self.project_zero_sum(self.main_pred(x))
        if return_features:
            features = {
                "path_tokens": x,
                "map_tokens": map_tokens,
                "canonical_map_centers": canonical_map_centers,
                "path_free_controls": free_controls,
                "condition": condition,
            }
            return state, features
        return state


class AlignmentOnlyV2SpatialPathMeanFlowTransformer(
    AlignmentOnlySpatialPathMeanFlowTransformer
):
    """Single-variable bug-fix arm over ``alignment_only``.

    The only change is the assumed 12x12 CNN feature-center mapping:

        input pixel index = 8*j + 3.5,  j = 0..11
        normalized window x/y = -1 + 0.02 * (8*j + 3.5)

    instead of ``linspace(-1.0, 0.98, 12)``.  Everything else (path
    coordinate PE, max_freq=16, map grid PE + canonical coord PE, normal
    rotation, joint raw S/G + time conditioning, parameter count) is
    inherited unchanged from ``alignment_only``.
    """

    ARCHITECTURE_NAME = "alignment_only_v2"
    FEATURE_CENTER_SEMANTICS = "cnn_pool_receptive_centers_8j_plus_3p5_v1"

    def _build_map_feature_centers_norm(self):
        xmin, xmax, ymin, ymax = self.MAP_FEATURE_CENTER_BOUNDS_NORM
        # Pixel centers of the 100x100 normalized source raster are
        # -1 + 0.02*i.  The four conv(3,p1)+maxpool(2) stages map feature
        # index j to input index 8*j + 3.5 (pool floors 25 -> 12 once).
        x_index = 8.0 * torch.arange(12, dtype=torch.float32) + 3.5
        spacing = (xmax - xmin) / 99.0
        x = xmin + spacing * x_index
        y = ymin + spacing * x_index
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(12 * 12, 2)


class PathXYSpatialPathMeanFlowTransformer(
    SpatialMapPathMeanFlowTransformer
):
    """Weak path-token XY hint over the spatial_map / single_12 baseline.

    The only forward change is

        X_i = PathMLP(z_i) + E_index(i) + alpha * Linear(C_i)

    where ``C_i`` is the canonical XY of the i-th free B-spline control.
    Map memory, cross-attention, joint raw S/G + time conditioning, and the
    decoder stay exactly as in ``SpatialMapPathMeanFlowTransformer``.  This
    is not Fourier PE, map coordinate alignment, or condition decoupling.
    """

    ARCHITECTURE_NAME = "path_xy"
    MAP_LAYOUTS = ("single_12",)
    OUTPUT_HEAD_TYPE = "tokenwise_22x2"
    PATH_TOKEN_SEMANTICS = "path_xy_linear_hint_v1"
    DEFAULT_PATH_XY_ALPHA = 0.1
    CONFIG_KEYS = PathDiffusionTransformer.CONFIG_KEYS | frozenset({
        "path_xy_alpha",
    })

    def __init__(
        self,
        *args,
        map_layout="single_12",
        path_xy_alpha=DEFAULT_PATH_XY_ALPHA,
        **kwargs,
    ):
        if map_layout != "single_12":
            raise ValueError(
                "PathXYSpatialPathMeanFlowTransformer currently "
                "supports only map_layout='single_12'"
            )
        super().__init__(*args, map_layout=map_layout, **kwargs)
        alpha = float(path_xy_alpha)
        if alpha < 0.0:
            raise ValueError("path_xy_alpha must be non-negative")
        self.path_xy_alpha = alpha
        d_model = self.dit_blocks[0].norm1.normalized_shape[0]
        self.path_xy_proj = nn.Linear(2, d_model)

    def _canonical_free_xy(self, noisy_path, start_pose, goal_pose):
        start_xy_yaw = self._condition_pose_to_xy_yaw(start_pose)
        goal_xy_yaw = self._condition_pose_to_xy_yaw(goal_pose)
        _, _, _, _, start_direction, goal_direction = (
            self.trajectory_representation._condition(
                start_xy_yaw,
                goal_xy_yaw,
                dtype=noisy_path.dtype,
                device=noisy_path.device,
            )
        )
        return self.trajectory_representation.canonical_control_points(
            self.validate_path_coordinates(noisy_path),
            start_direction,
            goal_direction,
        )[:, 2:24]

    def path_token_components(self, noisy_path, start_pose, goal_pose):
        path_mlp = self.path_patchify(noisy_path)
        index_pe = self.path_pos_embed.expand_as(path_mlp)
        free_xy = self._canonical_free_xy(noisy_path, start_pose, goal_pose)
        xy_raw = self.path_xy_proj(free_xy)
        xy_hint = self.path_xy_alpha * xy_raw
        tokens = path_mlp + self.path_pos_embed + xy_hint
        return {
            "path_mlp": path_mlp,
            "path_index_pe": index_pe,
            "path_free_xy": free_xy,
            "path_xy_raw": xy_raw,
            "path_xy_hint": xy_hint,
            "path_tokens": tokens,
        }

    def forward(
        self,
        map_input,
        noisy_path,
        timestep,
        timestep_r,
        start_pose,
        goal_pose,
        return_features=False,
    ):
        if map_input.ndim != 4 or map_input.shape[1] != self.map_channels:
            raise ValueError(
                f"Expected map_input (B,{self.map_channels},H,W), got "
                f"{tuple(map_input.shape)}"
            )
        map_tokens = self._build_map_memory(map_input)

        if noisy_path.ndim != 3 or noisy_path.shape[1:] != (self.num_edges, 2):
            raise ValueError(
                f"Expected noisy_path (B,{self.num_edges},2), got "
                f"{tuple(noisy_path.shape)}"
            )
        components = self.path_token_components(
            noisy_path, start_pose, goal_pose
        )
        x = components["path_tokens"]
        x = self.layer_norm(x)
        x = self.dropout(x)

        t_emb = self.time_embedder(timestep)
        h_emb = self.time_embedder(
            torch.clamp(timestep - timestep_r, min=0.0, max=1.0)
        )
        s_emb = self.pose_embedder(start_pose)
        g_emb = self.pose_embedder(goal_pose)
        condition = self.cond_mlp(
            torch.cat([t_emb, h_emb, s_emb, g_emb], dim=-1)
        )

        for block in self.dit_blocks:
            if self.training and self.use_gradient_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, map_tokens, condition, use_reentrant=False
                )
            else:
                x = block(x, map_tokens, condition)
        x = self.layer_norm(x)
        state = self.project_zero_sum(self.main_pred(x))
        if return_features:
            features = {
                "path_tokens": x,
                "map_tokens": map_tokens,
                "condition": condition,
                **components,
            }
            return state, features
        return state


class SharedXYSpatialPathMeanFlowTransformer(
    PathXYSpatialPathMeanFlowTransformer
):
    """Shared weak XY language for path and map tokens.

    Path tokens stay exactly as in ``path_xy``.  Map tokens add the same
    ``alpha * Linear(2, d)`` projection of canonical feature-center XY.
    The Linear is shared, so this arm has the same parameter count as
    ``path_xy``.  Map CNN + grid PE, cross-attention, joint conditioning,
    and the decoder are unchanged.  Feature centers use the corrected
    ``8j+3.5`` receptive-field mapping.
    """

    ARCHITECTURE_NAME = "shared_xy"
    PATH_TOKEN_SEMANTICS = "shared_xy_linear_hint_v1"
    FEATURE_CENTER_SEMANTICS = "cnn_pool_receptive_centers_8j_plus_3p5_v1"
    MAP_FEATURE_CENTER_BOUNDS_NORM = (-1.0, 0.98, -1.0, 0.98)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "map_feature_centers_norm",
            self._build_map_feature_centers_norm(),
        )

    def _build_map_feature_centers_norm(self):
        xmin, xmax, ymin, ymax = self.MAP_FEATURE_CENTER_BOUNDS_NORM
        x_index = 8.0 * torch.arange(12, dtype=torch.float32) + 3.5
        spacing = (xmax - xmin) / 99.0
        x = xmin + spacing * x_index
        y = ymin + spacing * x_index
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(12 * 12, 2)

    def _task_geometry(self, start_pose, goal_pose, dtype, device):
        start_xy_yaw = self._condition_pose_to_xy_yaw(start_pose)
        goal_xy_yaw = self._condition_pose_to_xy_yaw(goal_pose)
        start, _, distance, theta, start_direction, goal_direction = (
            self.trajectory_representation._condition(
                start_xy_yaw,
                goal_xy_yaw,
                dtype=dtype,
                device=device,
            )
        )
        return start, distance, theta, start_direction, goal_direction

    @staticmethod
    def _canonical_window_xy(points, start_xy, distance, theta):
        relative = points - start_xy.unsqueeze(1)
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        x = (
            cosine[:, None] * relative[..., 0]
            + sine[:, None] * relative[..., 1]
        )
        y = (
            -sine[:, None] * relative[..., 0]
            + cosine[:, None] * relative[..., 1]
        )
        scale = distance.clamp_min(1e-6).view(-1, 1)
        return torch.stack([x / scale, y / scale], dim=-1)

    def _canonical_map_xy(self, start_pose, goal_pose, batch, dtype, device):
        start, distance, theta, _, _ = self._task_geometry(
            start_pose, goal_pose, dtype=dtype, device=device
        )
        centers = self.map_feature_centers_norm.to(device=device, dtype=dtype)
        return self._canonical_window_xy(
            centers.unsqueeze(0).expand(batch, -1, -1),
            start[:, :2],
            distance,
            theta,
        )

    def map_token_components(self, map_input, start_pose, goal_pose):
        base_map = self._build_map_memory(map_input)
        map_xy = self._canonical_map_xy(
            start_pose,
            goal_pose,
            batch=map_input.shape[0],
            dtype=map_input.dtype,
            device=map_input.device,
        )
        map_xy_raw = self.path_xy_proj(map_xy)
        map_xy_hint = self.path_xy_alpha * map_xy_raw
        return {
            "map_base": base_map,
            "map_free_xy": map_xy,
            "map_xy_raw": map_xy_raw,
            "map_xy_hint": map_xy_hint,
            "map_tokens": base_map + map_xy_hint,
        }

    def forward(
        self,
        map_input,
        noisy_path,
        timestep,
        timestep_r,
        start_pose,
        goal_pose,
        return_features=False,
    ):
        if map_input.ndim != 4 or map_input.shape[1] != self.map_channels:
            raise ValueError(
                f"Expected map_input (B,{self.map_channels},H,W), got "
                f"{tuple(map_input.shape)}"
            )
        if noisy_path.ndim != 3 or noisy_path.shape[1:] != (self.num_edges, 2):
            raise ValueError(
                f"Expected noisy_path (B,{self.num_edges},2), got "
                f"{tuple(noisy_path.shape)}"
            )
        map_components = self.map_token_components(
            map_input, start_pose, goal_pose
        )
        map_tokens = map_components["map_tokens"]
        components = self.path_token_components(
            noisy_path, start_pose, goal_pose
        )
        x = components["path_tokens"]
        x = self.layer_norm(x)
        x = self.dropout(x)

        t_emb = self.time_embedder(timestep)
        h_emb = self.time_embedder(
            torch.clamp(timestep - timestep_r, min=0.0, max=1.0)
        )
        s_emb = self.pose_embedder(start_pose)
        g_emb = self.pose_embedder(goal_pose)
        condition = self.cond_mlp(
            torch.cat([t_emb, h_emb, s_emb, g_emb], dim=-1)
        )

        for block in self.dit_blocks:
            if self.training and self.use_gradient_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, map_tokens, condition, use_reentrant=False
                )
            else:
                x = block(x, map_tokens, condition)
        x = self.layer_norm(x)
        state = self.project_zero_sum(self.main_pred(x))
        if return_features:
            features = {
                "path_tokens": x,
                "map_tokens": map_tokens,
                "condition": condition,
                **components,
                **map_components,
            }
            return state, features
        return state


class TaskEncoderSpatialPathMeanFlowTransformer(
    SpatialMapPathMeanFlowTransformer
):
    """Feature-level task encoder over the spatial_map baseline.

    ``F_M = CNN(M) + grid PE`` is unchanged from baseline. Start/goal are
    encoded as a single vector ``h_SG = MLP([S, G])`` and fused onto every
    map token with ``Linear([F_M,j ; h_SG])``. AdaLN sees only MeanFlow
    time ``(t, t-r)``. There is no S/G heatmap, raster, or canonical
    spatial alignment. S/G still drive the analytic B-spline decoder.
    """

    ARCHITECTURE_NAME = "task_encoder"
    MAP_LAYOUTS = ("single_12",)
    OUTPUT_HEAD_TYPE = "tokenwise_22x2"
    CONDITION_SEMANTICS = "task_encoder_feature_fuse_v2"

    def __init__(self, *args, map_layout="single_12", **kwargs):
        if map_layout != "single_12":
            raise ValueError(
                "TaskEncoderSpatialPathMeanFlowTransformer currently "
                "supports only map_layout='single_12'"
            )
        super().__init__(*args, map_layout=map_layout, **kwargs)
        d_model = self.dit_blocks[0].norm1.normalized_shape[0]
        del self.cond_mlp
        del self.pose_embedder
        self.sg_mlp = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.task_fuse = nn.Linear(2 * d_model, d_model)
        self.time_cond_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 5),
            nn.GELU(),
            nn.Linear(d_model * 5, d_model),
        )

    def encode_task_memory(self, map_input, start_pose, goal_pose):
        map_features = self._build_map_memory(map_input)
        start_goal = torch.cat([start_pose, goal_pose], dim=-1)
        task_vector = self.sg_mlp(start_goal)
        fused = self.task_fuse(
            torch.cat(
                [
                    map_features,
                    task_vector.unsqueeze(1).expand_as(map_features),
                ],
                dim=-1,
            )
        )
        return fused, map_features, task_vector

    def forward(
        self,
        map_input,
        noisy_path,
        timestep,
        timestep_r,
        start_pose,
        goal_pose,
        return_features=False,
    ):
        if map_input.ndim != 4 or map_input.shape[1] != self.map_channels:
            raise ValueError(
                f"Expected map_input (B,{self.map_channels},H,W), got "
                f"{tuple(map_input.shape)}"
            )
        if noisy_path.ndim != 3 or noisy_path.shape[1:] != (self.num_edges, 2):
            raise ValueError(
                f"Expected noisy_path (B,{self.num_edges},2), got "
                f"{tuple(noisy_path.shape)}"
            )
        map_tokens, map_features, task_vector = self.encode_task_memory(
            map_input, start_pose, goal_pose
        )
        path_tokens = self.path_patchify(noisy_path)
        x = path_tokens + self.path_pos_embed
        x = self.layer_norm(x)
        x = self.dropout(x)

        t_emb = self.time_embedder(timestep)
        h_emb = self.time_embedder(
            torch.clamp(timestep - timestep_r, min=0.0, max=1.0)
        )
        condition = self.time_cond_mlp(torch.cat([t_emb, h_emb], dim=-1))

        for block in self.dit_blocks:
            if self.training and self.use_gradient_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, map_tokens, condition, use_reentrant=False
                )
            else:
                x = block(x, map_tokens, condition)
        x = self.layer_norm(x)
        state = self.project_zero_sum(self.main_pred(x))
        if return_features:
            features = {
                "path_tokens": x,
                "map_tokens": map_tokens,
                "map_features": map_features,
                "task_vector": task_vector,
                "condition": condition,
            }
            return state, features
        return state


class ConditionSelfAttentionLayer(nn.Module):
    """One pre-norm SA+FFN layer for the condition-side task memory."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attn = CompactAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens):
        normalized = self.norm1(tokens)
        tokens = tokens + self.self_attn(normalized, normalized, normalized)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens


class TaskMemorySpatialPathMeanFlowTransformer(
    SpatialMapPathMeanFlowTransformer
):
    """Tokenized map + S/G task memory with shallow condition-side SA.

    The map is patch-embedded rather than CNN-pooled. Start and goal are
    two tokens. These tokens are concatenated and updated by a shallow
    Transformer so S/G-map relations form inside the task encoder. Path
    tokens stay ``PathMLP(z)+E_index`` and read the resulting memory by
    CA. AdaLN sees only ``(t, t-r)``.
    """

    ARCHITECTURE_NAME = "task_memory"
    MAP_LAYOUTS = ("single_12",)
    OUTPUT_HEAD_TYPE = "tokenwise_22x2"
    PATCH_SIZE = 10
    TASK_SA_LAYERS = 2
    CONDITION_SEMANTICS = "task_memory_patch_sg_sa_v1"

    def __init__(self, *args, map_layout="single_12", **kwargs):
        if map_layout != "single_12":
            raise ValueError(
                "TaskMemorySpatialPathMeanFlowTransformer currently "
                "supports only map_layout='single_12'"
            )
        super().__init__(*args, map_layout=map_layout, **kwargs)
        d_model = self.dit_blocks[0].norm1.normalized_shape[0]
        n_heads = self.dit_blocks[0].self_attn.n_heads
        d_ff = self.dit_blocks[0].ffn[0].out_features
        dropout = self.dropout.p
        del self.map_fe_block1
        del self.map_fe_block2
        del self.map_fe_block3
        del self.map_fe_block4
        del self.cond_mlp
        self.map_patch_embed = nn.Conv2d(
            self.map_channels,
            d_model,
            kernel_size=self.PATCH_SIZE,
            stride=self.PATCH_SIZE,
        )
        self.task_type_embed = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.task_type_embed, std=0.02)
        self.task_sa = nn.ModuleList(
            [
                ConditionSelfAttentionLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(self.TASK_SA_LAYERS)
            ]
        )
        self.time_cond_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 5),
            nn.GELU(),
            nn.Linear(d_model * 5, d_model),
        )

    def encode_task_memory(self, map_input, start_pose, goal_pose):
        if map_input.shape[-2] % self.PATCH_SIZE != 0:
            raise ValueError(
                f"Map height {map_input.shape[-2]} is not divisible by "
                f"patch size {self.PATCH_SIZE}"
            )
        if map_input.shape[-1] % self.PATCH_SIZE != 0:
            raise ValueError(
                f"Map width {map_input.shape[-1]} is not divisible by "
                f"patch size {self.PATCH_SIZE}"
            )
        patches = self.map_patch_embed(map_input)
        map_tokens = self.reorder_dims(patches)
        map_tokens = self.map_position_enc(
            map_tokens, conv_shape=patches.shape[-2:]
        )
        start_token = self.pose_embedder(start_pose).unsqueeze(1)
        goal_token = self.pose_embedder(goal_pose).unsqueeze(1)
        start_token = start_token + self.task_type_embed[0]
        goal_token = goal_token + self.task_type_embed[1]
        map_tokens = map_tokens + self.task_type_embed[2]
        tokens = torch.cat([start_token, goal_token, map_tokens], dim=1)
        for layer in self.task_sa:
            tokens = layer(tokens)
        return tokens

    def forward(
        self,
        map_input,
        noisy_path,
        timestep,
        timestep_r,
        start_pose,
        goal_pose,
        return_features=False,
    ):
        if map_input.ndim != 4 or map_input.shape[1] != self.map_channels:
            raise ValueError(
                f"Expected map_input (B,{self.map_channels},H,W), got "
                f"{tuple(map_input.shape)}"
            )
        if noisy_path.ndim != 3 or noisy_path.shape[1:] != (self.num_edges, 2):
            raise ValueError(
                f"Expected noisy_path (B,{self.num_edges},2), got "
                f"{tuple(noisy_path.shape)}"
            )
        task_memory = self.encode_task_memory(
            map_input, start_pose, goal_pose
        )
        path_tokens = self.path_patchify(noisy_path)
        x = path_tokens + self.path_pos_embed
        x = self.layer_norm(x)
        x = self.dropout(x)

        t_emb = self.time_embedder(timestep)
        h_emb = self.time_embedder(
            torch.clamp(timestep - timestep_r, min=0.0, max=1.0)
        )
        condition = self.time_cond_mlp(torch.cat([t_emb, h_emb], dim=-1))

        for block in self.dit_blocks:
            if self.training and self.use_gradient_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, task_memory, condition, use_reentrant=False
                )
            else:
                x = block(x, task_memory, condition)
        x = self.layer_norm(x)
        state = self.project_zero_sum(self.main_pred(x))
        if return_features:
            features = {
                "path_tokens": x,
                "map_tokens": task_memory,
                "task_memory": task_memory,
                "condition": condition,
            }
            return state, features
        return state


class SplitConditionSpatialTrajBlock(nn.Module):
    """Specialized-routing diagnostic block (secondary arm).

    Time condition modulates the two path-processing sub-layers (SA, FFN)
    with ``6*d`` parameters; task condition modulates the map-reading
    sub-layer (CA) with ``3*d`` parameters.  The two heads therefore have the
    same total modulation width ``9*d`` as the baseline single-condition
    ``SpatialMapTrajBlock``, but they hard-wire which condition may affect
    which sub-layer.  This arm tests that routing prior; it is not the main
    Q2 decoupling arm.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = CompactAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.map_cross_attn = CompactAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.adaLN_mod_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        self.adaLN_mod_task = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model),
        )
        for head in (self.adaLN_mod_time, self.adaLN_mod_task):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x, map_tokens, time_condition, task_condition=None):
        (
            shift_sa, scale_sa, gate_sa,
            shift_ffn, scale_ffn, gate_ffn,
        ) = self.adaLN_mod_time(time_condition).chunk(6, dim=-1)
        if task_condition is None:
            shift_ca = scale_ca = gate_ca = time_condition.new_zeros(
                time_condition.shape[0], time_condition.shape[1]
            )
        else:
            shift_ca, scale_ca, gate_ca = self.adaLN_mod_task(
                task_condition
            ).chunk(3, dim=-1)

        normalized = self.norm1(x)
        normalized = normalized * (1.0 + scale_sa[:, None]) + shift_sa[:, None]
        x = x + gate_sa[:, None] * self.self_attn(
            normalized, normalized, normalized
        )

        normalized = self.norm2(x)
        normalized = normalized * (1.0 + scale_ca[:, None]) + shift_ca[:, None]
        x = x + gate_ca[:, None] * self.map_cross_attn(
            normalized, map_tokens, map_tokens
        )

        normalized = self.norm3(x)
        normalized = normalized * (1.0 + scale_ffn[:, None]) + shift_ffn[:, None]
        x = x + gate_ffn[:, None] * self.ffn(normalized)
        return x


class HalfWidthAdditiveSpatialTrajBlock(nn.Module):
    """Main Q2 capacity-matched additive block.

    ``c_time`` and ``c_task`` are each encoded to width ``d_model // 2``.
    Both conditions have their own zero-initialized head from that half
    width to the full ``9*d`` modulation vector, and the two modulation
    vectors are added.  Consequently both time and task can affect SA, CA
    and FFN without a hard functional-routing prior, while the two heads
    together have essentially the same parameter count as the baseline
    single ``d_model -> 9*d_model`` head.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("HalfWidthAdditiveSpatialTrajBlock needs even d_model")
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = CompactAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.map_cross_attn = CompactAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        half = d_model // 2
        self.adaLN_mod_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(half, 9 * d_model),
        )
        self.adaLN_mod_task = nn.Sequential(
            nn.SiLU(),
            nn.Linear(half, 9 * d_model),
        )
        for head in (self.adaLN_mod_time, self.adaLN_mod_task):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x, map_tokens, time_condition, task_condition=None):
        time_mod = self.adaLN_mod_time(time_condition)
        task_mod = time_mod.new_zeros(time_mod.shape)
        if task_condition is not None:
            task_mod = self.adaLN_mod_task(task_condition)
        modulation = time_mod + task_mod
        (
            shift_sa, scale_sa, gate_sa,
            shift_ca, scale_ca, gate_ca,
            shift_ffn, scale_ffn, gate_ffn,
        ) = modulation.chunk(9, dim=-1)

        normalized = self.norm1(x)
        normalized = normalized * (1.0 + scale_sa[:, None]) + shift_sa[:, None]
        x = x + gate_sa[:, None] * self.self_attn(
            normalized, normalized, normalized
        )

        normalized = self.norm2(x)
        normalized = normalized * (1.0 + scale_ca[:, None]) + shift_ca[:, None]
        x = x + gate_ca[:, None] * self.map_cross_attn(
            normalized, map_tokens, map_tokens
        )

        normalized = self.norm3(x)
        normalized = normalized * (1.0 + scale_ffn[:, None]) + shift_ffn[:, None]
        x = x + gate_ffn[:, None] * self.ffn(normalized)
        return x


class SplitRoutingTokenAlignedSpatialPathMeanFlowTransformer(
    TokenAlignedSpatialPathMeanFlowTransformer
):
    """Secondary specialized-routing diagnostic arm.

    Inherits the full token-aligned geometry (rotated normals, canonical
    map/path coordinate PE, no raw S/G learnable path).  It routes time to
    SA+FFN and task to CA.  This arm tests whether that hard routing prior
    has additional value; it is not the main Q2 decoupling arm.
    """

    ARCHITECTURE_NAME = "token_aligned_split_routing"
    CONDITION_FUSIONS = ("split_sub_heads",)
    CONDITION_CAPACITY_HIDDEN_MULTIPLIER = 4

    def __init__(
        self,
        *args,
        map_layout="single_12",
        coord_pe_max_freq=16.0,
        task_cond_mode="direction_scale",
        **kwargs,
    ):
        super().__init__(
            *args,
            map_layout=map_layout,
            coord_pe_max_freq=coord_pe_max_freq,
            task_cond_mode=task_cond_mode,
            condition_fusion="additive",
            **kwargs,
        )
        d_model = self.dit_blocks[0].norm1.normalized_shape[0]
        n_heads = self.dit_blocks[0].map_cross_attn.n_heads
        d_ff = self.dit_blocks[0].ffn[0].out_features
        self.dit_blocks = nn.ModuleList([
            SplitConditionSpatialTrajBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=self.dropout.p,
            )
            for _ in self.dit_blocks
        ])
        hidden = self.CONDITION_CAPACITY_HIDDEN_MULTIPLIER * d_model
        self.time_condition_mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        task_input_dim = {
            "none": 0,
            "scale": 1,
            "direction": 4,
            "direction_scale": 5,
        }[self.task_cond_mode]
        if task_input_dim > 0:
            self.task_condition_mlp = nn.Sequential(
                nn.Linear(task_input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            )
        self.condition_fusion = "split_sub_heads"


class AdditiveCapacityMatchedTokenAlignedSpatialPathMeanFlowTransformer(
    TokenAlignedSpatialPathMeanFlowTransformer
):
    """Main Q2 capacity-matched decoupled arm.

    Both ``c_time`` and ``c_task`` are encoded to width ``d_model // 2`` and
    both additively modulate SA, CA and FFN.  The two half-width heads have
    essentially the same modulation parameter count as the baseline single
    head, while the widened encoders keep the whole generator close to the
    baseline parameter count.  No hard functional routing is assumed.
    """

    ARCHITECTURE_NAME = "token_aligned_capacity_matched"
    CONDITION_FUSIONS = ("additive_half_width",)
    CONDITION_CAPACITY_HIDDEN_MULTIPLIER = 8

    def __init__(
        self,
        *args,
        map_layout="single_12",
        coord_pe_max_freq=16.0,
        task_cond_mode="direction_scale",
        **kwargs,
    ):
        super().__init__(
            *args,
            map_layout=map_layout,
            coord_pe_max_freq=coord_pe_max_freq,
            task_cond_mode=task_cond_mode,
            condition_fusion="additive",
            **kwargs,
        )
        d_model = self.dit_blocks[0].norm1.normalized_shape[0]
        if d_model % 2 != 0:
            raise ValueError("additive capacity-matched arm needs even d_model")
        n_heads = self.dit_blocks[0].map_cross_attn.n_heads
        d_ff = self.dit_blocks[0].ffn[0].out_features
        self.dit_blocks = nn.ModuleList([
            HalfWidthAdditiveSpatialTrajBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=self.dropout.p,
            )
            for _ in self.dit_blocks
        ])
        hidden = self.CONDITION_CAPACITY_HIDDEN_MULTIPLIER * d_model
        half = d_model // 2
        self.time_condition_mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, half),
        )
        task_input_dim = {
            "none": 0,
            "scale": 1,
            "direction": 4,
            "direction_scale": 5,
        }[self.task_cond_mode]
        if task_input_dim > 0:
            self.task_condition_mlp = nn.Sequential(
                nn.Linear(task_input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, half),
            )
        self.condition_fusion = "additive_half_width"


# Canonical production model name. The subclass does not add parameters or
# change computation; it only separates the current Path MeanFlow method from
# unrelated diffusion-era baselines that still import the legacy class name.
class PathMeanFlowTransformer(PathDiffusionTransformer):
    pass


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
    
    def __init__(
        self,
        n_layers=6,
        n_heads=8,
        d_model=512,
        d_inner=1024,
        dropout=0.1,
        coordinate_scale=MAP_HALF_EXTENT,
        map_channels=3,
        use_radial_output=True,
    ):
        super().__init__(
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            coordinate_scale=coordinate_scale,
            map_channels=map_channels,
            use_radial_output=use_radial_output,
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
        
        # 计算速度场
        t_val = t_scalar if isinstance(t_scalar, (int, float)) else t_scalar.item()
        if t_val < 1e-5:
            return torch.zeros_like(z_in)

        return self.project_zero_sum((z_in - model_out) / t_val)
    
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

        # 计算速度场（可微分）
        eps = 1e-5
        return self.project_zero_sum((z_in - model_out) / (t_scalar + eps))

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
    def sample_euler(self, map_input, start_pose, goal_pose, cost_scalar=0.0, num_steps=50, w=1.0):
        """Euler ODE 采样，支持 cost 条件。"""
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
    def sample_heun(self, map_input, start_pose, goal_pose, cost_scalar=0.0, num_steps=50, w=1.0):
        """Heun ODE 采样，支持 cost 条件。"""
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

