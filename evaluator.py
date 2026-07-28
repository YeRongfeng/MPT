import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from map_config import MAP_BOUNDS


class TrajectoryEvaluator:
    """
    轨迹评估器类，用于计算SE(2)空间中轨迹的各项指标
    
    支持的指标包括：
    - 碰撞风险（基于ESDF或占据地图）
    - 平滑度（基于二阶导数）
    - 几何曲率
    - 角速度约束
    - 速度分布
    - 轨迹长度
    - 估计时间
    - 控制点均匀性
    """
    
    def __init__(self, 
                 occupancy_map: Optional[torch.Tensor] = None,
                 yaw_stability_map: Optional[torch.Tensor] = None,
                 map_info: Optional[Dict] = None,
                 device: str = 'cuda'):
        """
        初始化评估器
        
        Args:
            occupancy_map: (D, H, W) 的占据地图或ESDF地图，D是yaw维度，H是y，W是x
            yaw_stability_map: (D, H, W) 的yaw稳定性地图，值表示该位置该yaw的稳定性
            map_info: 地图信息字典，包含 'resolution', 'origin', 'size'
            device: 计算设备
        """
        self.device = torch.device(device)
        self.map_info = map_info

        if occupancy_map is not None:
            occ = self._normalize_map_layout(occupancy_map)
            self.occupancy_map = occ.to(device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.has_map = True
        else:
            self.occupancy_map = None
            self.has_map = False
            
        if yaw_stability_map is not None:
            stab = self._normalize_map_layout(yaw_stability_map)
            self.yaw_stability_map = stab.to(device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.has_stability_map = True
        else:
            self.yaw_stability_map = None
            self.has_stability_map = False
            
        if map_info is not None:
            self.map_res = map_info['resolution']
            self.map_origin = map_info['origin']  # (x, y, yaw)
            # 优先使用 yaw_stability_map 的形状，如果没有则使用 occupancy_map
            if self.has_stability_map:
                # 内部统一为 (D, H, W)
                self.map_size_pixels = tuple(self.yaw_stability_map.shape[-3:])
            elif self.has_map:
                # 内部统一为 (D, H, W)
                self.map_size_pixels = tuple(self.occupancy_map.shape[-3:])
            else:
                self.map_size_pixels = None
        else:
            self.map_res = None
            self.map_origin = None
            self.map_size_pixels = None

    def _normalize_map_layout(self, map_tensor: torch.Tensor) -> torch.Tensor:
        """
        将输入地图统一为 (D, H, W)。
        支持输入为 (D,H,W) 或 (H,W,D)。
        """
        if not isinstance(map_tensor, torch.Tensor):
            map_tensor = torch.as_tensor(map_tensor)

        if map_tensor.ndim != 3:
            raise ValueError(f"Expected 3D map tensor, got shape={tuple(map_tensor.shape)}")

        if self.map_info is None:
            # 无法判断时维持原样（兼容旧行为）
            return map_tensor

        W, H, D = self.map_info['size']
        s0, s1, s2 = map_tensor.shape

        # 已是 (D,H,W)
        if (s0, s1, s2) == (D, H, W):
            return map_tensor

        # 是 (H,W,D)，转为 (D,H,W)
        if (s0, s1, s2) == (H, W, D):
            return map_tensor.permute(2, 0, 1).contiguous()

        # 兜底：如果最后一维看起来是 yaw 维，则按 (H,W,D) 处理
        if s2 == D:
            return map_tensor.permute(2, 0, 1).contiguous()

        return map_tensor
    
    def world_to_grid_normalized(self, poses_world: torch.Tensor) -> torch.Tensor:
        """
        将世界坐标转换为归一化的网格坐标（用于grid_sample）
        
        Args:
            poses_world: (K, 3) 形状的张量 (x, y, yaw)
            
        Returns:
            (1, K, 1, 1, 3) 形状的归一化坐标，顺序为 (x, y, z) 对应 (W, H, D)
        """
        if not self.has_map:
            raise ValueError("需要提供占据地图才能进行坐标转换")
        
        x_w, y_w, yaw_w = poses_world[:, 0], poses_world[:, 1], poses_world[:, 2]
        ox, oy, oyaw = self.map_origin
        D, H, W = self.map_size_pixels
        
        # 转换为像素坐标
        x_pix = (x_w - ox) / self.map_res
        y_pix = (y_w - oy) / self.map_res
        yaw_normalized = (yaw_w - oyaw) / (2.0 * np.pi)
        yaw_pix = yaw_normalized * D
        
        # 归一化到 [-1, 1]
        x_norm = (x_pix / (W - 1)) * 2.0 - 1.0
        y_norm = (y_pix / (H - 1)) * 2.0 - 1.0
        yaw_norm = (yaw_pix / (D - 1)) * 2.0 - 1.0
        
        # grid_sample 对于 5D 输入 (N, C, D, H, W)，grid 的最后一维是 (x, y, z)
        # 对应输入张量的 (W, H, D) 维度
        # 所以顺序应该是 (x, y, z) = (W, H, D) = (x_norm, y_norm, yaw_norm)
        coords = torch.stack([x_norm, y_norm, yaw_norm], dim=1)  # (K, 3) - 正确顺序：(x, y, z)
        coords = coords.view(1, -1, 1, 1, 3)  # (1, K, 1, 1, 3)
        
        return coords
    
    def evaluate_trajectory(self, 
                          trajectory: torch.Tensor,
                          velocities: Optional[torch.Tensor] = None,
                          accelerations: Optional[torch.Tensor] = None,
                          control_points: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        评估轨迹的所有指标
        
        Args:
            trajectory: (K, 3) 形状的轨迹点 (x, y, yaw)
            velocities: (K, 3) 形状的速度 (vx, vy, vyaw)，可选
            accelerations: (K, 3) 形状的加速度 (ax, ay, ayaw)，可选
            control_points: (N, 3) 形状的控制点，用于计算均匀性，可选
            
        Returns:
            包含各项指标的字典
        """
        metrics = {}
        
        # 确保输入是张量
        if not isinstance(trajectory, torch.Tensor):
            trajectory = torch.tensor(trajectory, dtype=torch.float32, device=self.device)
        else:
            trajectory = trajectory.to(device=self.device, dtype=torch.float32)
        
        K = trajectory.shape[0]
        
        # 1. 碰撞风险指标
        if self.has_map:
            metrics.update(self._evaluate_collision_risk(trajectory))
        
        # 2. Yaw稳定性指标
        if self.has_stability_map:
            metrics.update(self._evaluate_yaw_stability(trajectory))
        
        # 3. 轨迹长度
        metrics['path_length'] = self._compute_path_length(trajectory).item()
        
        # 4. 如果没有提供速度和加速度，则从轨迹数值微分计算
        if velocities is None or accelerations is None:
            velocities_computed, accelerations_computed = self._compute_derivatives(trajectory)
            if velocities is None:
                velocities = velocities_computed
            if accelerations is None:
                accelerations = accelerations_computed
        
        # 5. 平滑度指标
        metrics.update(self._evaluate_smoothness(accelerations))
        
        # 6. 几何曲率
        metrics.update(self._evaluate_curvature(velocities, accelerations))
        
        # 7. 角速度指标
        metrics.update(self._evaluate_angular_velocity(velocities[:, 2]))
        
        # 8. 速度相关指标
        metrics.update(self._evaluate_velocity_profile(velocities))
        
        # 9. 时间估计
        metrics['estimated_time'] = self._estimate_time(trajectory, velocities).item()
        
        # 10. 航向一致性
        metrics.update(self._evaluate_heading_consistency(trajectory, velocities))

        # 11. 几何诊断：周期 yaw、切向速度、曲率爆点等，用于定位 B-spline 曲线问题
        metrics.update(self._evaluate_geometry_diagnostics(trajectory, velocities, accelerations))
        
        # 12. 控制点均匀性（如果提供）
        if control_points is not None:
            metrics.update(self._evaluate_control_point_uniformity(control_points))
        
        # 13. 边界检查
        metrics.update(self._check_bounds(trajectory))
        
        return metrics
    
    def _evaluate_collision_risk(self, trajectory: torch.Tensor) -> Dict[str, float]:
        """评估碰撞风险"""
        metrics = {}
        
        grid_coords = self.world_to_grid_normalized(trajectory)
        
        # 从地图采样（假设是ESDF）
        esdf_sample = F.grid_sample(
            self.occupancy_map,
            grid_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        esdf_flat = esdf_sample.reshape(-1)
        
        # 方法1: 基于ESDF的碰撞风险（假设地图存储带符号距离）
        d_safe = 0.15  # 安全距离阈值（米）
        kalpa = 0.7  # 平滑参数（米）
        # z = (-(esdf_flat - d_safe) / (kalpa + 1e-12))
        # collision_risk = torch.sigmoid(torch.clamp(z, min=-50.0, max=50.0))
        alpha = 10.0 
        collision_risk = torch.nn.functional.softplus(-alpha * (esdf_flat - d_safe)) / alpha
        
        metrics['collision_risk_mean'] = collision_risk.mean().item()
        metrics['collision_risk_max'] = collision_risk.max().item()
        metrics['collision_risk_std'] = collision_risk.std().item()
        
        # 计算危险点的比例（基于ESDF插值结果，距离<=0视为碰撞/危险）
        dangerous_points = (esdf_flat < 0.0).sum().float()
        metrics['dangerous_point_ratio'] = (dangerous_points / len(esdf_flat)).item()
        
        # 最小距离
        metrics['min_obstacle_distance'] = esdf_flat.min().item()
        
        return metrics
    
    def _evaluate_yaw_stability(self, trajectory: torch.Tensor, stability_threshold: float = 0., debug: bool = False) -> Dict[str, float]:
        """
        评估轨迹点的yaw稳定性（基于二值化稳定性地图）
        
        Args:
            trajectory: (K, 3) 形状的轨迹点 (x, y, yaw)
            stability_threshold: 稳定性阈值，采样值低于此阈值视为不稳定
                                默认为0.5（因为地图是0/1二值，插值后可能在0-1之间）
            debug: 是否输出调试信息
            
        Returns:
            包含稳定性指标的字典
        
        注意：
            yaw_stability_map 是二值化地图：
            - 1.0 表示该位置该角度稳定（可达）
            - 0.0 表示该位置该角度不稳定（不可达/会倾覆）
        """
        metrics = {}
        
        if debug:
            print("\n[DEBUG] _evaluate_yaw_stability:")
            print(f"  Trajectory shape: {trajectory.shape}")
            print(f"  Map shape (D, H, W): {self.map_size_pixels}")
            print(f"  Map resolution: {self.map_res}")
            print(f"  Map origin: {self.map_origin}")
            print(f"  Sample trajectory points:")
            for i in [0, len(trajectory)//2, -1]:
                x, y, yaw = trajectory[i].cpu().numpy()
                print(f"    Point {i}: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        
        grid_coords = self.world_to_grid_normalized(trajectory)
        
        if debug:
            print(f"  Grid coords shape: {grid_coords.shape}")
            print(f"  Sample grid coords (normalized to [-1, 1]):")
            coords_np = grid_coords.squeeze().cpu().numpy()
            for i in [0, len(coords_np)//2, -1]:
                yaw_n, y_n, x_n = coords_np[i, 0, 0]
                print(f"    Point {i}: x_norm={x_n:.4f}, y_norm={y_n:.4f}, yaw_norm={yaw_n:.4f}")
        
        # 从稳定性地图采样（使用最近邻插值避免插值导致的模糊）
        # 注意：使用 'nearest' 模式来保持二值特性
        stability_sample = F.grid_sample(
            self.yaw_stability_map,
            grid_coords,
            mode='bilinear',  # 改用最近邻插值，保持0/1的二值特性
            padding_mode='border',
            align_corners=True
        )
        
        stability_flat = stability_sample.reshape(-1)
        
        if debug:
            print(f"  Stability sample shape: {stability_sample.shape}")
            print(f"  Stability values (first 10): {stability_flat[:10].cpu().numpy()}")
            print(f"  Stability values (last 10): {stability_flat[-10:].cpu().numpy()}")
            print(f"  Unique stability values: {torch.unique(stability_flat).cpu().numpy()}")
        
        # 计算不稳定点比例：stability < threshold 表示不稳定/不可达
        # 对于二值地图，stability=0 表示不可达，stability=1 表示可达
        unstable_points = (stability_flat < stability_threshold).sum().float()
        metrics['unstable_point_ratio'] = (unstable_points / len(stability_flat)).item()
        
        if debug:
            print(f"  Unstable points: {unstable_points.item()} / {len(stability_flat)}")
            print(f"  Unstable ratio: {metrics['unstable_point_ratio']:.4f}")
        
        # 稳定性统计（用于诊断）
        metrics['stability_mean'] = stability_flat.mean().item()
        metrics['stability_min'] = stability_flat.min().item()
        metrics['stability_max'] = stability_flat.max().item()
        metrics['stability_std'] = stability_flat.std().item()
        
        # 稳定点比例（stability >= threshold）
        stable_points = (stability_flat >= stability_threshold).sum().float()
        metrics['stable_point_ratio'] = (stable_points / len(stability_flat)).item()
        
        # 完全不可达点比例（stability == 0，精确匹配）
        unreachable_points = (stability_flat < 0.1).sum().float()  # 使用0.1作为容差
        metrics['unreachable_point_ratio'] = (unreachable_points / len(stability_flat)).item()
        
        return metrics
    
    def _compute_path_length(self, trajectory: torch.Tensor) -> torch.Tensor:
        """计算轨迹总长度"""
        dx = trajectory[1:, 0] - trajectory[:-1, 0]
        dy = trajectory[1:, 1] - trajectory[:-1, 1]
        segment_lengths = torch.sqrt(dx**2 + dy**2)
        return segment_lengths.sum()
    
    def _compute_derivatives(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从轨迹数值计算速度和加速度"""
        K = trajectory.shape[0]
        
        # 一阶导数（速度）- 使用中心差分
        velocities = torch.zeros_like(trajectory)
        velocities[1:-1] = (trajectory[2:] - trajectory[:-2]) / 2.0
        velocities[0] = trajectory[1] - trajectory[0]
        velocities[-1] = trajectory[-1] - trajectory[-2]
        
        # 二阶导数（加速度）
        accelerations = torch.zeros_like(trajectory)
        accelerations[1:-1] = trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]
        accelerations[0] = trajectory[0] - 2 * trajectory[1] + trajectory[2] if K > 2 else torch.zeros(3, device=self.device)
        accelerations[-1] = trajectory[-3] - 2 * trajectory[-2] + trajectory[-1] if K > 2 else torch.zeros(3, device=self.device)
        
        return velocities, accelerations
    
    def _evaluate_smoothness(self, accelerations: torch.Tensor) -> Dict[str, float]:
        """评估轨迹平滑度"""
        metrics = {}
        
        # 基于加速度的平滑度
        ax, ay, ayaw = accelerations[:, 0], accelerations[:, 1], accelerations[:, 2]
        
        metrics['smoothness_x'] = (ax**2).mean().item()
        metrics['smoothness_y'] = (ay**2).mean().item()
        metrics['smoothness_yaw'] = (ayaw**2).mean().item()
        metrics['smoothness_total'] = ((ax**2 + ay**2 + ayaw**2).mean()).item()

        # jerk 应为加速度的一阶差分（离散时间下 j[k] = a[k+1]-a[k]）
        # 这里没有显式 dt，按每采样步长为 1 计算离散 jerk 指标。
        if accelerations.shape[0] >= 2:
            jerk = accelerations[1:] - accelerations[:-1]  # (K-1, 3)
            jx, jy, jyaw = jerk[:, 0], jerk[:, 1], jerk[:, 2]
            # 使用 RMS 作为 jerk 强度指标
            metrics['jerk_x'] = torch.sqrt((jx**2).mean() + 1e-12).item()
            metrics['jerk_y'] = torch.sqrt((jy**2).mean() + 1e-12).item()
            metrics['jerk_yaw'] = torch.sqrt((jyaw**2).mean() + 1e-12).item()
        else:
            metrics['jerk_x'] = 0.0
            metrics['jerk_y'] = 0.0
            metrics['jerk_yaw'] = 0.0

        # 保留原有“加速度抖动”统计，避免语义混淆
        metrics['acc_std_x'] = ax.std().item()
        metrics['acc_std_y'] = ay.std().item()
        metrics['acc_std_yaw'] = ayaw.std().item()
        
        return metrics
    
    def _evaluate_curvature(self, velocities: torch.Tensor, accelerations: torch.Tensor) -> Dict[str, float]:
        """评估几何曲率"""
        metrics = {}
        
        vx, vy = velocities[:, 0], velocities[:, 1]
        ax, ay = accelerations[:, 0], accelerations[:, 1]
        
        eps = 1e-6
        speed = torch.sqrt(vx**2 + vy**2 + eps)
        
        # 几何曲率 κ = |v × a| / |v|³
        cross_product = torch.abs(vx * ay - vy * ax)
        curvature = cross_product / (speed**3 + 1e-6)
        
        metrics['curvature_mean'] = curvature.mean().item()
        metrics['curvature_max'] = curvature.max().item()
        metrics['curvature_std'] = curvature.std().item()
        
        # 曲率超限比例（假设限制为1.4 rad/m）
        curvature_limit = 1.4
        curvature_violations = (curvature > curvature_limit).sum().float()
        metrics['curvature_violation_ratio'] = (curvature_violations / len(curvature)).item()
        
        return metrics
    
    def _evaluate_angular_velocity(self, yaw_dot: torch.Tensor) -> Dict[str, float]:
        """评估角速度"""
        metrics = {}
        
        metrics['angular_velocity_mean'] = yaw_dot.abs().mean().item()
        metrics['angular_velocity_max'] = yaw_dot.abs().max().item()
        metrics['angular_velocity_std'] = yaw_dot.std().item()
        
        return metrics
    
    def _evaluate_velocity_profile(self, velocities: torch.Tensor) -> Dict[str, float]:
        """评估速度分布"""
        metrics = {}
        
        vx, vy = velocities[:, 0], velocities[:, 1]
        speed = torch.sqrt(vx**2 + vy**2)
        
        metrics['speed_mean'] = speed.mean().item()
        metrics['speed_max'] = speed.max().item()
        metrics['speed_min'] = speed.min().item()
        metrics['speed_std'] = speed.std().item()
        
        # 低速点比例（速度 < 0.1 m/s）
        slow_points = (speed < 0.1).sum().float()
        metrics['slow_point_ratio'] = (slow_points / len(speed)).item()
        
        # 速度变化率
        speed_changes = torch.abs(speed[1:] - speed[:-1])
        metrics['speed_change_mean'] = speed_changes.mean().item()
        metrics['speed_change_max'] = speed_changes.max().item()
        
        return metrics
    
    def _estimate_time(self, trajectory: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        """估计轨迹执行时间"""
        eps = 1e-6
        
        # 计算轨迹段长度
        dx = trajectory[1:, 0] - trajectory[:-1, 0]
        dy = trajectory[1:, 1] - trajectory[:-1, 1]
        segment_distances = torch.sqrt(dx**2 + dy**2 + eps)
        
        # 计算速度
        vx, vy = velocities[:, 0], velocities[:, 1]
        speed = torch.sqrt(vx**2 + vy**2 + eps)
        
        # 限制最小速度以避免除零
        min_speed = 0.1
        speed_safe = torch.clamp(speed, min=min_speed)
        
        # 梯形积分估计时间
        inv_speed_start = 1.0 / speed_safe[:-1]
        inv_speed_end = 1.0 / speed_safe[1:]
        inv_speed_avg = (inv_speed_start + inv_speed_end) / 2
        
        total_time = torch.sum(inv_speed_avg * segment_distances)
        
        return total_time
    
    def _evaluate_heading_consistency(self, trajectory: torch.Tensor, velocities: torch.Tensor) -> Dict[str, float]:
        """评估航向与运动方向的一致性"""
        metrics = {}
        
        vx, vy = velocities[:, 0], velocities[:, 1]
        yaw = trajectory[:, 2]
        
        # 计算速度方向
        velocity_angle = torch.atan2(vy, vx)
        
        # 计算航向与速度方向的夹角
        angle_diff = velocity_angle - yaw
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        
        metrics['heading_error_mean'] = angle_diff.abs().mean().item()
        metrics['heading_error_max'] = angle_diff.abs().max().item()
        metrics['heading_error_std'] = angle_diff.std().item()
        
        # 倒车检测（航向与速度方向相反，角度差接近π）
        backward_threshold = np.pi / 2  # 90度
        backward_motion = (angle_diff.abs() > backward_threshold).sum().float()
        metrics['backward_ratio'] = (backward_motion / len(angle_diff)).item()
        
        # 计算朝向的前进方向投影
        forward_x = torch.cos(yaw)
        forward_y = torch.sin(yaw)
        speed_projection = vx * forward_x + vy * forward_y
        
        # 实际倒车的比例（投影为负）
        actual_backward = (speed_projection < 0).sum().float()
        metrics['actual_backward_ratio'] = (actual_backward / len(speed_projection)).item()
        
        return metrics

    def _angle_diff(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """周期角差 a-b，返回 [-pi, pi]。"""
        diff = a - b
        return torch.atan2(torch.sin(diff), torch.cos(diff))

    def _evaluate_geometry_diagnostics(
        self,
        trajectory: torch.Tensor,
        velocities: torch.Tensor,
        accelerations: torch.Tensor
    ) -> Dict[str, float]:
        """额外几何诊断：不替代原指标，只帮助判断 yaw wrap、低切向速度和曲率尖峰。"""
        metrics = {}
        K = trajectory.shape[0]
        if K < 2:
            return {
                'smoothness_xy': 0.0,
                'periodic_smoothness_yaw': 0.0,
                'periodic_jerk_yaw': 0.0,
                'raw_yaw_step_max': 0.0,
                'periodic_yaw_step_max': 0.0,
                'yaw_wrap_jump_count': 0.0,
                'yaw_wrap_jump_ratio': 0.0,
                'tangent_norm_min': 0.0,
                'tangent_norm_mean': 0.0,
                'tangent_near_zero_ratio_1e-3': 0.0,
                'tangent_near_zero_ratio_1e-2': 0.0,
                'tangent_near_zero_ratio_5e-2': 0.0,
            }

        yaw = trajectory[:, 2]
        raw_yaw_step = yaw[1:] - yaw[:-1]
        periodic_yaw_step = self._angle_diff(yaw[1:], yaw[:-1])

        metrics['raw_yaw_step_max'] = raw_yaw_step.abs().max().item()
        metrics['raw_yaw_step_mean'] = raw_yaw_step.abs().mean().item()
        metrics['periodic_yaw_step_max'] = periodic_yaw_step.abs().max().item()
        metrics['periodic_yaw_step_mean'] = periodic_yaw_step.abs().mean().item()
        metrics['yaw_wrap_jump_count'] = (raw_yaw_step.abs() > np.pi).sum().float().item()
        metrics['yaw_wrap_jump_ratio'] = (raw_yaw_step.abs() > np.pi).float().mean().item()
        metrics['periodic_large_turn_ratio_pi_2'] = (periodic_yaw_step.abs() > (np.pi / 2)).float().mean().item()
        metrics['periodic_large_turn_ratio_pi_4'] = (periodic_yaw_step.abs() > (np.pi / 4)).float().mean().item()

        ax, ay = accelerations[:, 0], accelerations[:, 1]
        metrics['smoothness_xy'] = (ax**2 + ay**2).mean().item()
        metrics['smoothness_yaw_fraction'] = (
            metrics.get('smoothness_yaw', (accelerations[:, 2] ** 2).mean().item()) /
            (metrics['smoothness_xy'] + metrics.get('smoothness_yaw', (accelerations[:, 2] ** 2).mean().item()) + 1e-12)
        )

        if periodic_yaw_step.numel() >= 2:
            periodic_yaw_acc = self._angle_diff(periodic_yaw_step[1:], periodic_yaw_step[:-1])
            metrics['periodic_smoothness_yaw'] = (periodic_yaw_acc**2).mean().item()
            metrics['periodic_yaw_acc_max'] = periodic_yaw_acc.abs().max().item()
            if periodic_yaw_acc.numel() >= 2:
                periodic_yaw_jerk = self._angle_diff(periodic_yaw_acc[1:], periodic_yaw_acc[:-1])
                metrics['periodic_jerk_yaw'] = torch.sqrt((periodic_yaw_jerk**2).mean() + 1e-12).item()
                metrics['periodic_yaw_jerk_max'] = periodic_yaw_jerk.abs().max().item()
            else:
                metrics['periodic_jerk_yaw'] = 0.0
                metrics['periodic_yaw_jerk_max'] = 0.0
        else:
            metrics['periodic_smoothness_yaw'] = 0.0
            metrics['periodic_yaw_acc_max'] = 0.0
            metrics['periodic_jerk_yaw'] = 0.0
            metrics['periodic_yaw_jerk_max'] = 0.0

        vx, vy = velocities[:, 0], velocities[:, 1]
        tangent_norm = torch.sqrt(vx**2 + vy**2)
        metrics['tangent_norm_min'] = tangent_norm.min().item()
        metrics['tangent_norm_mean'] = tangent_norm.mean().item()
        metrics['tangent_norm_std'] = tangent_norm.std().item()
        metrics['tangent_near_zero_ratio_1e-3'] = (tangent_norm < 1e-3).float().mean().item()
        metrics['tangent_near_zero_ratio_1e-2'] = (tangent_norm < 1e-2).float().mean().item()
        metrics['tangent_near_zero_ratio_5e-2'] = (tangent_norm < 5e-2).float().mean().item()
        metrics['tangent_near_zero_ratio_1e-1'] = (tangent_norm < 1e-1).float().mean().item()

        curvature_eps = 1e-6
        cross_product = vx * ay - vy * ax
        curvature_abs = torch.abs(cross_product) / (tangent_norm.clamp_min(curvature_eps) ** 3 + curvature_eps)
        metrics['curvature_abs_mean'] = curvature_abs.mean().item()
        metrics['curvature_abs_max'] = curvature_abs.max().item()
        metrics['curvature_abs_std'] = curvature_abs.std().item()
        metrics['curvature_abs_q90'] = torch.quantile(curvature_abs, 0.9).item()
        metrics['curvature_abs_q99'] = torch.quantile(curvature_abs, 0.99).item()
        metrics['curvature_abs_violation_ratio_1p4'] = (curvature_abs > 1.4).float().mean().item()
        metrics['curvature_abs_violation_ratio_5'] = (curvature_abs > 5.0).float().mean().item()

        return metrics
    
    def _evaluate_control_point_uniformity(self, control_points: torch.Tensor) -> Dict[str, float]:
        """评估控制点的均匀性"""
        metrics = {}
        
        if not isinstance(control_points, torch.Tensor):
            control_points = torch.tensor(control_points, dtype=torch.float32, device=self.device)
        else:
            control_points = control_points.to(device=self.device, dtype=torch.float32)
        
        N = control_points.shape[0]
        
        if N < 2:
            return metrics
        
        # 计算相邻控制点之间的距离
        dx = control_points[1:, 0] - control_points[:-1, 0]
        dy = control_points[1:, 1] - control_points[:-1, 1]
        segment_distances = torch.sqrt(dx**2 + dy**2 + 1e-6)
        
        # 计算理想的均匀段长度
        total_length = segment_distances.sum()
        
        if total_length > 1e-8:
            ideal_segment_length = total_length / (N - 1)
            
            # 计算各种均匀性指标
            length_deviations = segment_distances - ideal_segment_length
            
            # 归一化均方根偏差
            metrics['uniformity_rmse'] = torch.sqrt((length_deviations**2).mean()).item() / (ideal_segment_length.item() + 1e-12)
            
            # 归一化标准差
            metrics['uniformity_std'] = segment_distances.std().item() / (ideal_segment_length.item() + 1e-12)
            
            # 最大偏差比
            metrics['uniformity_max_deviation'] = (length_deviations.abs().max() / (ideal_segment_length + 1e-12)).item()
            
            # 变异系数（CV）
            metrics['uniformity_cv'] = (segment_distances.std() / (segment_distances.mean() + 1e-12)).item()
            
            # 段长度范围
            metrics['segment_length_min'] = segment_distances.min().item()
            metrics['segment_length_max'] = segment_distances.max().item()
            metrics['segment_length_ratio'] = (segment_distances.max() / (segment_distances.min() + 1e-12)).item()
        
        return metrics
    
    def _check_bounds(
        self,
        trajectory: torch.Tensor,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, float]:
        """检查轨迹是否超出边界"""
        metrics = {}

        if bounds is None:
            if self.map_info is not None and 'bounds' in self.map_info:
                bounds = tuple(self.map_info['bounds'])
            elif self.map_info is not None:
                resolution = float(self.map_info['resolution'])
                origin = self.map_info['origin']
                W, H, _ = self.map_info['size']
                bounds = (
                    float(origin[0]),
                    float(origin[0]) + W * resolution,
                    float(origin[1]),
                    float(origin[1]) + H * resolution,
                )
            else:
                bounds = MAP_BOUNDS
        if len(bounds) != 4:
            raise ValueError(f"bounds must be (xmin,xmax,ymin,ymax), got {bounds}")
        x_min, x_max, y_min, y_max = map(float, bounds)
        
        x, y = trajectory[:, 0], trajectory[:, 1]
        
        # 计算超出距离
        exceed_x = torch.maximum(
            torch.tensor(x_min, device=x.device, dtype=x.dtype) - x,
            x - torch.tensor(x_max, device=x.device, dtype=x.dtype),
        )
        exceed_y = torch.maximum(
            torch.tensor(y_min, device=y.device, dtype=y.dtype) - y,
            y - torch.tensor(y_max, device=y.device, dtype=y.dtype),
        )
        
        # 超出点的数量和比例
        out_of_bounds_x = (exceed_x > 0).sum().float()
        out_of_bounds_y = (exceed_y > 0).sum().float()
        out_of_bounds_total = ((exceed_x > 0) | (exceed_y > 0)).sum().float()
        
        metrics['out_of_bounds_ratio'] = (out_of_bounds_total / len(trajectory)).item()
        metrics['out_of_bounds_x_ratio'] = (out_of_bounds_x / len(trajectory)).item()
        metrics['out_of_bounds_y_ratio'] = (out_of_bounds_y / len(trajectory)).item()
        
        # 最大超出距离
        metrics['max_exceed_x'] = torch.relu(exceed_x).max().item()
        metrics['max_exceed_y'] = torch.relu(exceed_y).max().item()
        
        # 边界距离统计
        distance_to_bound_x = torch.minimum(x - x_min, x_max - x)
        distance_to_bound_y = torch.minimum(y - y_min, y_max - y)
        min_distance_to_bound = torch.minimum(distance_to_bound_x, distance_to_bound_y)
        
        metrics['min_distance_to_boundary'] = min_distance_to_bound.min().item()
        metrics['mean_distance_to_boundary'] = min_distance_to_bound.mean().item()
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], title: str = "轨迹评估指标"):
        """格式化打印评估指标"""
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}\n")
        
        # 分类打印
        categories = {
            "碰撞风险": ['collision_risk_mean', 'collision_risk_max', 'collision_risk_std', 
                       'dangerous_point_ratio', 'min_obstacle_distance'],
            "Yaw稳定性": ['unstable_point_ratio', 'unreachable_point_ratio', 'stability_mean', 
                        'stability_min', 'stability_max', 'stability_std', 'stable_point_ratio'],
            "轨迹长度": ['path_length', 'estimated_time'],
            "平滑度": ['smoothness_total', 'smoothness_x', 'smoothness_y', 'smoothness_yaw',
                     'smoothness_xy', 'periodic_smoothness_yaw',
                     'jerk_x', 'jerk_y', 'jerk_yaw', 'periodic_jerk_yaw'],
            "曲率": ['curvature_mean', 'curvature_max', 'curvature_std', 'curvature_violation_ratio',
                   'curvature_abs_max', 'curvature_abs_q99', 'curvature_abs_violation_ratio_1p4'],
            "几何诊断": ['raw_yaw_step_max', 'periodic_yaw_step_max', 'yaw_wrap_jump_ratio',
                     'periodic_large_turn_ratio_pi_2', 'tangent_norm_min', 'tangent_norm_mean',
                     'tangent_near_zero_ratio_1e-2', 'tangent_near_zero_ratio_5e-2'],
            "角速度": ['angular_velocity_mean', 'angular_velocity_max', 'angular_velocity_std'],
            "速度": ['speed_mean', 'speed_max', 'speed_min', 'speed_std', 
                    'slow_point_ratio', 'speed_change_mean', 'speed_change_max'],
            "航向一致性": ['heading_error_mean', 'heading_error_max', 'heading_error_std',
                        'backward_ratio', 'actual_backward_ratio'],
            "控制点均匀性": ['uniformity_rmse', 'uniformity_std', 'uniformity_max_deviation',
                         'uniformity_cv', 'segment_length_min', 'segment_length_max', 
                         'segment_length_ratio'],
            "边界检查": ['out_of_bounds_ratio', 'out_of_bounds_x_ratio', 'out_of_bounds_y_ratio',
                       'max_exceed_x', 'max_exceed_y', 'min_distance_to_boundary', 
                       'mean_distance_to_boundary']
        }
        
        for category, keys in categories.items():
            category_metrics = {k: v for k, v in metrics.items() if k in keys}
            if category_metrics:
                print(f"\n{category}:")
                print("-" * 60)
                for key, value in category_metrics.items():
                    if isinstance(value, float):
                        print(f"  {key:40s}: {value:12.6f}")
                    else:
                        print(f"  {key:40s}: {value}")
        
        print(f"\n{'='*60}\n")
    
    def compare_trajectories(self, 
                           trajectories: Dict[str, torch.Tensor],
                           velocities: Optional[Dict[str, torch.Tensor]] = None,
                           accelerations: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Dict[str, float]]:
        """
        比较多条轨迹的指标
        
        Args:
            trajectories: 轨迹字典，键为名称，值为轨迹张量
            velocities: 速度字典（可选）
            accelerations: 加速度字典（可选）
            
        Returns:
            每条轨迹的评估指标字典
        """
        results = {}
        
        for name, traj in trajectories.items():
            vel = velocities.get(name) if velocities else None
            acc = accelerations.get(name) if accelerations else None
            
            metrics = self.evaluate_trajectory(traj, vel, acc)
            results[name] = metrics
        
        return results
    
    def get_summary_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        提取关键汇总指标
        
        Args:
            metrics: 完整指标字典
            
        Returns:
            关键指标的子集
        """
        summary_keys = [
            'collision_risk_mean',
            'unstable_point_ratio',
            'path_length',
            'estimated_time',
            'smoothness_total',
            'curvature_mean',
            'speed_mean',
            'out_of_bounds_ratio'
        ]
        
        return {k: metrics[k] for k in summary_keys if k in metrics}


def main():
    """使用示例"""
    # 创建示例数据
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("轨迹评估器功能演示".center(80))
    print("="*80)
    
    # =================== 示例1: 基础评估（无地图、无控制点）===================
    print("\n示例1: 基础轨迹评估（无碰撞检测、无控制点）")
    print("-"*80)
    
    # 圆形轨迹（100个点）
    t = torch.linspace(0, 2*np.pi, 100, device=device)
    trajectory = torch.stack([
        5 * torch.cos(t),  # x
        5 * torch.sin(t),  # y
        t + np.pi/2  # yaw - 切线方向（速度方向）
    ], dim=1)
    
    # 创建评估器（不使用地图）
    evaluator_basic = TrajectoryEvaluator(device=device)
    
    # 评估轨迹
    metrics_basic = evaluator_basic.evaluate_trajectory(trajectory)
    
    # 打印结果
    evaluator_basic.print_metrics(metrics_basic, "圆形轨迹评估（基础版）")
    
    # =================== 示例2: 带控制点的评估 ===================
    print("\n" + "="*80)
    print("示例2: 带控制点的轨迹评估")
    print("-"*80)
    
    # 创建稀疏控制点（10个点）
    t_control = torch.linspace(0, 2*np.pi, 10, device=device)
    control_points = torch.stack([
        5 * torch.cos(t_control),
        5 * torch.sin(t_control),
        t_control + np.pi/2
    ], dim=1)
    
    # 评估带控制点的轨迹
    metrics_with_control = evaluator_basic.evaluate_trajectory(
        trajectory, 
        control_points=control_points
    )
    
    # 打印结果
    evaluator_basic.print_metrics(metrics_with_control, "圆形轨迹评估（含控制点）")
    
    # =================== 示例3: 带虚拟地图的评估 ===================
    print("\n" + "="*80)
    print("示例3: 带碰撞检测的轨迹评估（虚拟地图）")
    print("-"*80)
    
    # 创建一个简单的虚拟ESDF地图（中心是障碍物，外围安全）
    map_size = (36, 200, 200)  # (yaw_bins, height, width)
    occupancy_map = torch.ones(map_size, device=device) * 5.0  # 初始化为5米距离（安全）
    
    # 在中心添加一个障碍物（距离为负）
    center_y, center_x = 100, 100
    radius = 30
    y_grid, x_grid = torch.meshgrid(
        torch.arange(map_size[1], device=device),
        torch.arange(map_size[2], device=device),
        indexing='ij'
    )
    distance_to_center = torch.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    obstacle_mask = distance_to_center < radius
    
    # 为所有yaw层设置障碍物
    for yaw_idx in range(map_size[0]):
        occupancy_map[yaw_idx][obstacle_mask] = -(radius - distance_to_center[obstacle_mask]) * 0.1
    
    # 创建地图信息
    map_info = {
        'resolution': 0.2,  # 0.2米/像素
        'origin': (-20.0, -20.0, -np.pi),  # 地图原点
        'size': map_size
    }
    
    # 创建带地图的评估器
    evaluator_with_map = TrajectoryEvaluator(
        occupancy_map=occupancy_map,
        map_info=map_info,
        device=device
    )
    
    # 评估同一轨迹（圆形轨迹不会碰到中心障碍物）
    metrics_with_map = evaluator_with_map.evaluate_trajectory(
        trajectory,
        control_points=control_points
    )
    
    # 打印完整结果（包含碰撞风险和控制点均匀性）
    evaluator_with_map.print_metrics(metrics_with_map, "圆形轨迹评估（完整版）")
    
    # =================== 示例4: 对比不同轨迹 ===================
    print("\n" + "="*80)
    print("示例4: 多轨迹对比")
    print("-"*80)
    
    # 创建几种不同的轨迹
    trajectories = {
        '圆形轨迹': trajectory,
        '直线轨迹': torch.stack([
            torch.linspace(-5, 5, 100, device=device),
            torch.zeros(100, device=device),
            torch.zeros(100, device=device)
        ], dim=1),
        '螺旋轨迹': torch.stack([
            t * torch.cos(t * 3) * 0.5,
            t * torch.sin(t * 3) * 0.5,
            t * 3
        ], dim=1)
    }
    
    # 批量评估
    comparison_results = evaluator_basic.compare_trajectories(trajectories)
    
    # 打印对比摘要
    print("\n轨迹对比摘要:")
    print("-"*80)
    print(f"{'指标':<30} {'圆形轨迹':>12} {'直线轨迹':>12} {'螺旋轨迹':>12}")
    print("-"*80)
    
    key_metrics = ['path_length', 'smoothness_total', 'curvature_mean', 'speed_mean']
    for metric in key_metrics:
        values = [comparison_results[name].get(metric, 0) for name in trajectories.keys()]
        print(f"{metric:<30} {values[0]:>12.6f} {values[1]:>12.6f} {values[2]:>12.6f}")
    
    print("\n" + "="*80)
    print("演示完成！".center(80))
    print("="*80)


if __name__ == '__main__':
    main()
