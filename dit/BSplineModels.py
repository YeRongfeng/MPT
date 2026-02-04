"""
BSplineModels.py - 基于B样条参数化的扩散模型

【核心优势】
1. 局部控制：移动控制点只影响局部曲线，不像多项式全局影响
2. 灵活性高：可以处理急转弯、复杂形状
3. 完全可微：基函数计算可微，梯度正常传递
4. 数值稳定：避免高阶多项式的数值问题

【B样条基础】
- 使用n个控制点定义曲线
- 每个点只影响局部区域（由阶数k决定）
- 3次B样条（k=4）是最常用的，C2连续
- 曲线在控制点凸包内

【参数化方案】
网络预测控制点的角度：
    θ_control = [θ₀, θ₁, ..., θₙ]
    
B样条插值得到密集角度：
    θ(s) = Σ Bᵢ,ₖ(s) · θᵢ
    
运动学积分得到轨迹：
    x(s) = x₀ + ∫cos(θ(s))ds
    y(s) = y₀ + ∫sin(θ(s))ds

【优势对比多项式】
✓ 可以处理急转弯（局部控制）
✓ 数值稳定（无高阶项）
✓ 灵活性高（控制点数量可调）
✓ 梯度传递友好
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from dit.DiTLayers import DiTBlock, TimestepEmbedder


class BSplineInterpolator:
    """
    B样条插值器：从控制点生成平滑曲线
    
    完全可微，支持梯度传递
    """
    
    @staticmethod
    def uniform_knot_vector(n_control_points, degree):
        """
        生成均匀节点向量
        
        Args:
            n_control_points: 控制点数量
            degree: B样条阶数（3=三次样条）
            
        Returns:
            knots: 节点向量
        """
        # 均匀节点向量：[0, 0, 0, 0, 1/(n-k), 2/(n-k), ..., 1, 1, 1, 1]
        n = n_control_points
        k = degree + 1
        
        # 内部节点数量
        # Clamped B-spline: n_knots = n + degree + 1
        # n_knots = (degree+1) + n_internal + (degree+1)
        # => n_internal = n - degree - 1
        n_internal = n - degree - 1
        
        if n_internal > 0:
            internal_knots = torch.linspace(0, 1, n_internal + 2)[1:-1]
            knots = torch.cat([
                torch.zeros(k),
                internal_knots,
                torch.ones(k)
            ])
        else:
            # 如果控制点太少，使用开放均匀节点
            knots = torch.cat([
                torch.zeros(k),
                torch.ones(k)
            ])
        
        return knots
    
    @staticmethod
    def basis_function(i, k, t, knots):
        """
        递归计算B样条基函数 B_{i,k}(t)
        
        Cox-de Boor递推公式（完全可微）
        
        Args:
            i: 基函数索引
            k: 阶数（degree + 1）
            t: 参数值 (scalar or tensor)
            knots: 节点向量
            
        Returns:
            basis: 基函数值（与t形状相同）
        """
        # 边界检查：确保访问knots[i+k]时不越界
        if i < 0 or i + k > len(knots):
            return torch.zeros_like(t)
        
        if k == 1:
            # 0阶基函数（分段常数）
            # B_{i,1}(t) = 1 if knots[i] <= t < knots[i+1], else 0
            # 特殊处理：如果knots[i+1]是最后一个唯一值（后面都是重复的），则包含右端点
            result = ((t >= knots[i]) & (t < knots[i+1])).float()
            
            # 检查是否是通向最后唯一节点值的区间
            # 即：knots[i] < knots[i+1] == knots[-1]
            is_last_unique_interval = (knots[i] < knots[i+1]) and (knots[i+1] >= knots[-1] - 1e-10)
            if is_last_unique_interval:
                result = result + (t >= knots[i+1]).float()
            
            return result
        
        # 递归计算
        # B_{i,k}(t) = w1 * B_{i,k-1}(t) + w2 * B_{i+1,k-1}(t)
        
        # 第一项权重
        denom1 = knots[i+k-1] - knots[i]
        if denom1 > 1e-10:
            w1 = (t - knots[i]) / denom1
        else:
            w1 = torch.zeros_like(t)
        
        # 第二项权重
        denom2 = knots[i+k] - knots[i+1]
        if denom2 > 1e-10:
            w2 = (knots[i+k] - t) / denom2
        else:
            w2 = torch.zeros_like(t)
        
        # 递归
        basis1 = BSplineInterpolator.basis_function(i, k-1, t, knots)
        basis2 = BSplineInterpolator.basis_function(i+1, k-1, t, knots)
        
        return w1 * basis1 + w2 * basis2
    
    @staticmethod
    def evaluate_bspline(control_points, t_values, degree=3):
        """
        计算B样条曲线（完全可微，支持批处理）
        
        Args:
            control_points: (B, n_control, dim) 控制点
            t_values: (n_samples,) 或 (B, n_samples) 参数值，范围[0,1]
            degree: B样条阶数（3=三次样条）
            
        Returns:
            curve: (B, n_samples, dim) 插值曲线
        """
        B, n_control, dim = control_points.shape
        device = control_points.device
        
        # 生成节点向量
        knots = BSplineInterpolator.uniform_knot_vector(n_control, degree).to(device)
        k = degree + 1
        
        # 处理t_values形状
        if t_values.dim() == 1:
            t_values = t_values.unsqueeze(0).expand(B, -1)  # (B, n_samples)
        
        n_samples = t_values.shape[1]
        
        # 计算所有基函数值（可微分）
        # 为了效率，向量化计算
        basis_matrix = torch.zeros(B, n_samples, n_control, device=device)
        
        for i in range(n_control):
            for b in range(B):
                basis_matrix[b, :, i] = BSplineInterpolator.basis_function(
                    i, k, t_values[b], knots
                )
        
        # 矩阵乘法：(B, n_samples, n_control) @ (B, n_control, dim) -> (B, n_samples, dim)
        curve = torch.bmm(basis_matrix, control_points)
        
        return curve
    
    @staticmethod
    def integrate_kinematics_from_angle_control_points(angle_control_points, start_pose, 
                                                       total_length, n_path_points=21):
        """
        从角度控制点生成SE(2)轨迹（完全可微）
        
        Args:
            angle_control_points: (B, n_control) 角度控制点
            start_pose: (B, 3) 起点 [x, y, theta]
            total_length: float or (B,) 轨迹总长度
            n_path_points: 输出路径点数（包含起点）
            
        Returns:
            trajectory: (B, n_path_points-1, 3) 轨迹（去掉起点）
        """
        B = angle_control_points.shape[0]
        device = angle_control_points.device
        
        # B样条插值角度：从控制点得到密集角度曲线
        # 使用更多采样点确保积分精度（梯度仍然可以传递）
        n_integration_points = n_path_points * 5  # 密集采样
        t_values = torch.linspace(0, 1, n_integration_points, device=device)
        
        # (B, n_control) -> (B, n_control, 1) -> B样条 -> (B, n_integration_points)
        angle_curve = BSplineInterpolator.evaluate_bspline(
            angle_control_points.unsqueeze(-1),  # (B, n_control, 1)
            t_values,
            degree=3
        ).squeeze(-1)  # (B, n_integration_points)
        
        # 运动学积分（可微分）
        ds = total_length / (n_integration_points - 1)
        if isinstance(ds, torch.Tensor):
            ds = ds.view(B, 1)
        
        # 梯形法则积分
        cos_theta = torch.cos(angle_curve)
        sin_theta = torch.sin(angle_curve)
        
        # 第一步特殊处理：使用真实起点角度
        cos_theta[:, 0] = torch.cos(start_pose[:, 2])
        sin_theta[:, 0] = torch.sin(start_pose[:, 2])
        
        # 累积积分
        dx = ((cos_theta[:, :-1] + cos_theta[:, 1:]) / 2) * ds
        dy = ((sin_theta[:, :-1] + sin_theta[:, 1:]) / 2) * ds
        
        x = start_pose[:, 0:1] + torch.cumsum(dx, dim=1)  # (B, n_integration_points-1)
        y = start_pose[:, 1:2] + torch.cumsum(dy, dim=1)
        
        # 下采样到目标点数（去掉起点）
        indices = torch.linspace(0, n_integration_points-2, n_path_points-1, device=device).long()
        
        x_sampled = x[:, indices]  # (B, n_path_points-1)
        y_sampled = y[:, indices]
        theta_sampled = angle_curve[:, indices+1]  # +1是因为去掉了起点
        
        trajectory = torch.stack([x_sampled, y_sampled, theta_sampled], dim=-1)  # (B, n_path_points-1, 3)
        
        return trajectory


class BSplineDiffusionTransformer(nn.Module):
    """
    基于B样条参数化的扩散Transformer
    
    预测角度控制点，通过B样条生成轨迹
    """
    
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner,
                 dropout=0.1, n_angle_control_points=12, n_path_points=21,
                 diffusion_steps=50, prediction_type='x0',
                 n_position=225, train_shape=(15, 15)):
        """
        Args:
            n_angle_control_points: 角度控制点数量（建议10-15个）
            n_path_points: 输出路径点数（21 = 起点 + 20中间点）
            其他参数同DiT
        """
        super().__init__()
        
        self.n_angle_control_points = n_angle_control_points
        self.n_path_points = n_path_points
        self.diffusion_steps = diffusion_steps
        self.prediction_type = prediction_type
        self.d_model = d_model
        
        # 图像编码器（CNN特征提取）
        self.map_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.map_proj = nn.Linear(128 * 16, d_model)
        
        # 控制点嵌入（每个控制点是一个角度标量）
        self.control_point_embed = nn.Linear(1, d_model)
        
        # 起终点嵌入
        self.start_embed = nn.Linear(3, d_model)
        self.goal_embed = nn.Linear(3, d_model)
        
        # 时间步嵌入
        self.time_embed = TimestepEmbedder(d_model)
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_angle_control_points + 3, d_model) * 0.02
        )  # 控制点 + 地图 + 起点 + 终点
        
        # Transformer层
        mlp_ratio = d_inner / d_model  # 计算MLP比例
        self.layers = nn.ModuleList([
            DiTBlock(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # 预测头：输出角度控制点
        self.control_point_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)  # 每个token预测一个角度
        )
        
        # 扩散参数
        self.register_diffusion_parameters(diffusion_steps)
    
    def register_diffusion_parameters(self, steps):
        """注册扩散参数（与DDPM相同）"""
        betas = torch.linspace(1e-4, 0.02, steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
    
    def forward(self, map_input, noisy_control_points, t, start_pose, goal_pose):
        """
        前向传播
        
        Args:
            map_input: (B, 3, H, W) 地图
            noisy_control_points: (B, n_control) 加噪的控制点
            t: (B,) 时间步
            start_pose: (B, 3) 起点
            goal_pose: (B, 3) 终点
            
        Returns:
            pred_control_points: (B, n_control) 预测的控制点（或噪声，取决于prediction_type）
        """
        B = map_input.shape[0]
        device = map_input.device
        
        # 地图编码
        map_feat = self.map_encoder(map_input)  # (B, 128, 4, 4)
        map_feat = map_feat.view(B, -1)  # (B, 128*16)
        map_token = self.map_proj(map_feat).unsqueeze(1)  # (B, 1, d_model)
        
        # 控制点嵌入
        control_tokens = self.control_point_embed(
            noisy_control_points.unsqueeze(-1)
        )  # (B, n_control, d_model)
        
        # 起终点嵌入
        start_token = self.start_embed(start_pose).unsqueeze(1)  # (B, 1, d_model)
        goal_token = self.goal_embed(goal_pose).unsqueeze(1)
        
        # 拼接所有token
        tokens = torch.cat([
            map_token, start_token, goal_token, control_tokens
        ], dim=1)  # (B, n_control+3, d_model)
        
        # 位置编码
        tokens = tokens + self.pos_embed
        
        # 时间步调制
        t_embed = self.time_embed(t)  # (B, d_model)
        
        # Transformer处理
        for layer in self.layers:
            tokens = layer(tokens, t_embed)
        
        # 只取控制点token
        control_tokens = tokens[:, 3:]  # (B, n_control, d_model)
        
        # 预测控制点
        pred_control_points = self.control_point_predictor(control_tokens).squeeze(-1)  # (B, n_control)
        
        return pred_control_points
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散：给x_0加噪"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise
    
    def control_points_to_trajectory(self, control_points, start_pose, total_length):
        """
        从控制点生成轨迹（可微分）
        
        Args:
            control_points: (B, n_control) 角度控制点
            start_pose: (B, 3) 起点
            total_length: float or (B,) 弧长
            
        Returns:
            trajectory: (B, 20, 3) 轨迹（20个中间点）
        """
        return BSplineInterpolator.integrate_kinematics_from_angle_control_points(
            control_points, start_pose, total_length, self.n_path_points
        )
    
    @torch.no_grad()
    def ddim_sample(self, map_input, start_pose, goal_pose, ddim_steps=20):
        """
        DDIM采样（加速推理）
        
        Returns:
            trajectory: (B, 20, 3) 采样的轨迹
        """
        B = map_input.shape[0]
        device = map_input.device
        
        # 从噪声开始
        x = torch.randn(B, self.n_angle_control_points, device=device)
        
        # DDIM步长
        step_size = self.diffusion_steps // ddim_steps
        time_steps = list(range(0, self.diffusion_steps, step_size))[::-1]
        
        for i, t in enumerate(time_steps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # 预测
            pred_output = self(map_input, x, t_batch, start_pose, goal_pose)
            
            # 恢复x0
            if self.prediction_type == 'epsilon':
                sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
                sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
                pred_x0 = (x - sqrt_one_minus_alpha_t * pred_output) / sqrt_alpha_t
            else:
                pred_x0 = pred_output
            
            # DDIM更新
            if i < len(time_steps) - 1:
                t_next = time_steps[i + 1]
                alpha_t = self.alphas_cumprod[t]
                alpha_t_next = self.alphas_cumprod[t_next]
                
                x = torch.sqrt(alpha_t_next) * pred_x0 + \
                    torch.sqrt(1 - alpha_t_next) * pred_output
            else:
                x = pred_x0
        
        # 从控制点生成轨迹（需要弧长信息）
        # 这里使用平均弧长作为估计
        avg_length = 30.0  # 根据数据统计调整
        
        trajectory = self.control_points_to_trajectory(x, start_pose, avg_length)
        
        return trajectory
    
    def differentiable_sample(self, map_input, start_pose, goal_pose, ddim_steps=20):
        """
        可微分采样（用于第二阶段优化）
        
        与ddim_sample相同，但保留梯度
        """
        B = map_input.shape[0]
        device = map_input.device
        
        # 从噪声开始
        x = torch.randn(B, self.n_angle_control_points, device=device)
        
        # DDIM步长
        step_size = self.diffusion_steps // ddim_steps
        time_steps = list(range(0, self.diffusion_steps, step_size))[::-1]
        
        for i, t in enumerate(time_steps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            pred_output = self(map_input, x, t_batch, start_pose, goal_pose)
            
            if self.prediction_type == 'epsilon':
                sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
                sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
                pred_x0 = (x - sqrt_one_minus_alpha_t * pred_output) / sqrt_alpha_t
            else:
                pred_x0 = pred_output
            
            if i < len(time_steps) - 1:
                t_next = time_steps[i + 1]
                alpha_t_next = self.alphas_cumprod[t_next]
                x = torch.sqrt(alpha_t_next) * pred_x0 + \
                    torch.sqrt(1 - alpha_t_next) * pred_output
            else:
                x = pred_x0
        
        avg_length = 30.0
        trajectory = self.control_points_to_trajectory(x, start_pose, avg_length)
        
        return trajectory
    
    def unfreeze_for_stage1(self):
        """
        阶段1：解冻所有参数（轨迹重建训练）
        """
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_for_stage2(self):
        """
        阶段2：冻结编码器，只训练解码器（cost优化）
        """
        # 冻结图像编码器
        for param in self.map_encoder.parameters():
            param.requires_grad = False
        
        # 冻结嵌入层
        for param in self.map_proj.parameters():
            param.requires_grad = False
        for param in self.start_embed.parameters():
            param.requires_grad = False
        for param in self.goal_embed.parameters():
            param.requires_grad = False
        
        # 保持Transformer和输出层可训练
        for param in self.time_embed.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.output_projection.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """
        获取所有 requires_grad=True 的参数
        """
        return [p for p in self.parameters() if p.requires_grad]
