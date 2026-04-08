"""
轨迹精细化模块 - 在轨迹点级别进行 Cross-Attention

这个模块解决的核心问题：
- 当前的 Cross-Attention 作用在 24 个控制点上
- 但小车的倾覆风险取决于 100 个实际轨迹点的朝向
- 这个模块在轨迹点级别进行精细化调整

关键特性：
1. 100 个轨迹点与地图特征的 Cross-Attention
2. 相邻轨迹点间的自注意力（平滑性约束）
3. 可微分的轨迹点偏移预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajFinetuningBlock(nn.Module):
    """
    单个轨迹精细化块
    
    架构：
    Input: traj_tokens (B, 100, D)
        |
        ├─ Self-Attention (轨迹点间) → 平滑性约束
        |
        ├─ Cross-Attention (轨迹点查询地图) → 避障
        |
        └─ FFN → 特征变换
        
    Output: refined_traj_tokens (B, 100, D)
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # 轨迹点位置编码（学习相邻点的序列关系）
        # 这让模型知道每个token在轨迹上的相对位置
        self.register_parameter(
            'traj_pos_embed',
            nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        )
        
        # 自注意力：轨迹点之间的关系
        # 目的：学习相邻点应该平滑过渡，避免尖锐转弯
        self.self_attn = nn.MultiheadAttention(
            d_model, 
            num_heads=n_heads, 
            dropout=dropout, 
            batch_first=True,
            bias=True
        )
        
        # Cross-Attention：轨迹点查询地图特征
        # 目的：每个轨迹点感知其局部地图环境，避开障碍或适应地形
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
            bias=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, traj_tokens: torch.Tensor, map_tokens: torch.Tensor) -> torch.Tensor:
        """
        轨迹精细化前向传播
        
        Args:
            traj_tokens: (B, 100, D) 轨迹点特征
                         - B: batch size
                         - 100: 重建后的轨迹点数量
                         - D: 特征维度
            
            map_tokens: (B, H*W, D) 地图特征（来自CNN）
                       - H*W: 下采样后的地图大小（e.g., 12*12=144）
                       - D: 特征维度（与 traj_tokens 相同）
        
        Returns:
            refined_traj_tokens: (B, 100, D) 精细化后的轨迹点特征
        
        设计细节：
        1. 位置编码使得模型知道每个点在轨迹上的位置
        2. 自注意力保证轨迹的连续性和平滑性
        3. Cross-Attention 让每个点感知地形，避开障碍或调整朝向
        """
        # ========== 1. 自注意力：轨迹点间的关系 ==========
        # 规范化输入
        traj_norm = self.norm1(traj_tokens)
        
        # 添加位置编码（让模型知道序列顺序）
        traj_norm = traj_norm + self.traj_pos_embed
        
        # 自注意力（同一来源的Q、K、V）
        attn_out, attn_weights = self.self_attn(traj_norm, traj_norm, traj_norm)
        attn_out = self.dropout(attn_out)
        
        # 残差连接
        traj_tokens = traj_tokens + attn_out
        
        # ========== 2. Cross-Attention：轨迹点查询地图 ==========
        # ⭐ 关键：这里轨迹点（Q）查询地图特征（K/V）
        # 地图中哪些区域与这个轨迹点相关，模型会自动学习
        
        # 规范化轨迹特征作为 Query
        traj_norm = self.norm2(traj_tokens)
        
        # Cross-Attention
        # Q: 轨迹点（我在哪里？）
        # K/V: 地图特征（周围有什么？）
        cross_out, cross_weights = self.cross_attn(
            query=traj_norm,           # (B, 100, D)
            key=map_tokens,            # (B, 144, D)
            value=map_tokens           # (B, 144, D)
        )
        cross_out = self.dropout(cross_out)
        
        # 残差连接
        traj_tokens = traj_tokens + cross_out
        
        # ========== 3. 前馈网络 ==========
        traj_norm = self.norm3(traj_tokens)
        ffn_out = self.ffn(traj_norm)
        ffn_out = self.dropout(ffn_out)
        
        # 残差连接
        traj_tokens = traj_tokens + ffn_out
        
        return traj_tokens


class TrajPointEmbedder(nn.Module):
    """
    将轨迹点坐标 (x, y) 转换为 d_model 维的特征向量
    
    这一步很关键：
    - 输入：轨迹点坐标 (B, 100, 2)
    - 输出：轨迹点特征 (B, 100, d_model)
    
    相当于把"物理坐标"转换为"语义特征"
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
    
    def forward(self, traj_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            traj_points: (B, 100, 2) 轨迹点坐标
        
        Returns:
            traj_features: (B, 100, d_model) 轨迹点特征
        """
        return self.net(traj_points)


class TrajectoryOffsetPredictor(nn.Module):
    """
    轨迹点微调预测头
    
    输入：精细化后的轨迹点特征 (B, 100, d_model)
    输出：坐标偏移 (B, 100, 2) 
    
    设计逻辑：
    - 模型预测每个轨迹点应该如何微调位置
    - dx, dy ∈ [-0.5, 0.5] 像素范围
    - Zero 初始化：训练初期不做任何调整
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # 输出 (dx, dy)
        )
        
        # Zero 初始化以保证训练稳定性
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, traj_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            traj_features: (B, 100, d_model)
        
        Returns:
            offset: (B, 100, 2) 范围 [-0.5, 0.5]
        """
        offset = self.net(traj_features)
        # 限制偏移范围
        offset = torch.tanh(offset) * 0.5  # 范围: [-0.5, 0.5]
        return offset


class TrajectoryCorrectionLoss(nn.Module):
    """
    轨迹级别的损失函数
    
    包含三部分：
    1. 坐标损失：轨迹点应该接近目标
    2. 朝向损失：相邻点的朝向角应该平滑
    3. 曲率损失：避免过度转弯
    """
    
    def __init__(self, weight_coord=1.0, weight_angle=0.5, weight_curvature=0.2):
        super().__init__()
        self.weight_coord = weight_coord
        self.weight_angle = weight_angle
        self.weight_curvature = weight_curvature
    
    def compute_angles(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        计算轨迹上每两个相邻点之间的朝向角
        
        Args:
            trajectory: (B, N, 2)
        
        Returns:
            angles: (B, N-1) 每个相邻点对的朝向角
        """
        # 计算方向向量
        diff = trajectory[:, 1:, :] - trajectory[:, :-1, :]  # (B, N-1, 2)
        
        # 计算朝向角 θ = atan2(dy, dx)
        angles = torch.atan2(diff[:, :, 1], diff[:, :, 0])  # (B, N-1)
        
        return angles
    
    def compute_curvature(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        计算轨迹曲率（相邻两段的朝向变化）
        
        Args:
            trajectory: (B, N, 2)
        
        Returns:
            curvature: (B, N-2) 曲率
        """
        angles = self.compute_angles(trajectory)  # (B, N-1)
        
        # 角度差异 = 曲率
        curvature = torch.abs(angles[:, 1:] - angles[:, :-1])  # (B, N-2)
        
        # 处理角度的周期性（180° = -180°）
        curvature = torch.min(curvature, 2 * 3.14159 - curvature)
        
        return curvature
    
    def forward(self, pred_trajectory: torch.Tensor, target_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_trajectory: (B, 100, 2) 预测的轨迹
            target_trajectory: (B, 100, 2) 目标轨迹
        
        Returns:
            loss: 标量损失
        """
        B = pred_trajectory.shape[0]
        
        # 1. 坐标损失
        loss_coord = F.smooth_l1_loss(pred_trajectory, target_trajectory)
        
        # 2. 朝向损失：相邻点的朝向应该保持一致
        pred_angles = self.compute_angles(pred_trajectory)  # (B, 99)
        target_angles = self.compute_angles(target_trajectory)  # (B, 99)
        loss_angle = F.smooth_l1_loss(pred_angles, target_angles)
        
        # 3. 曲率损失：避免过度转弯
        pred_curvature = self.compute_curvature(pred_trajectory)  # (B, 98)
        target_curvature = self.compute_curvature(target_trajectory)  # (B, 98)
        loss_curvature = F.smooth_l1_loss(pred_curvature, target_curvature)
        
        # 加权求和
        total_loss = (
            self.weight_coord * loss_coord +
            self.weight_angle * loss_angle +
            self.weight_curvature * loss_curvature
        )
        
        return total_loss


# ============ 使用示例 ============

if __name__ == "__main__":
    """
    演示如何使用轨迹精细化模块
    """
    
    B, D = 8, 256  # batch size, feature dimension
    
    # 创建模块
    refine_block = TrajFinetuningBlock(d_model=D, n_heads=8, d_ff=1024)
    point_embedder = TrajPointEmbedder(d_model=D)
    offset_predictor = TrajectoryOffsetPredictor(d_model=D)
    loss_fn = TrajectoryCorrectionLoss()
    
    # 模拟数据
    traj_points = torch.randn(B, 100, 2) * 0.5  # (B, 100, 2) 轨迹点坐标
    map_tokens = torch.randn(B, 144, D)  # (B, 144, D) 地图特征
    target_traj = torch.randn(B, 100, 2) * 0.5  # (B, 100, 2) 目标轨迹
    
    # 1. 嵌入轨迹点
    traj_features = point_embedder(traj_points)  # (B, 100, D)
    
    # 2. 精细化（多次）
    for _ in range(3):
        traj_features = refine_block(traj_features, map_tokens)
    
    # 3. 预测偏移
    offset = offset_predictor(traj_features)  # (B, 100, 2)
    
    # 4. 微调后的轨迹
    final_traj = traj_points + offset  # (B, 100, 2)
    
    # 5. 计算损失
    loss = loss_fn(final_traj, target_traj)
    
    print(f"Input trajectory shape: {traj_points.shape}")
    print(f"Trajectory features shape: {traj_features.shape}")
    print(f"Offset shape: {offset.shape}")
    print(f"Final trajectory shape: {final_traj.shape}")
    print(f"Loss: {loss.item():.4f}")
