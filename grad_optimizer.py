import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import tqdm

# 假设旧的辅助函数和数据加载器依然存在
from dataLoader_uneven import UnevenPathDataLoader

class TrajectoryOptimizerSE2:
    def __init__(self, points, occupancy_map, map_info, device='cuda'):
        """
        在 SE(2) 空间中优化轨迹（但将 (x,y,yaw) 视为普通的 3D 坐标，不处理角度周期性或 unwrap）。
        
        Args:
            points (np.ndarray or torch.Tensor): (N, 3) 的控制点 (x, y, yaw).
            occupancy_map (torch.Tensor): (D, H, W) 的 C 空间占据地图，
                                           其中 D 是 yaw 的离散维度, H 是 y, W 是 x.
                                           地图中的值越大，代表成本越高（越可能碰撞）.
            map_info (dict): 包含地图元信息的字典，例如:
                             {'resolution': 0.1, 'origin': (-5.0, -5.0, -np.pi), 
                              'size': (100, 100, 36)}.
                              origin 的顺序是 (x, y, yaw).
            device (str): 'cuda' 或 'cpu'.
        """
        self.device = torch.device(device)

        # --- 1. 处理控制点 ---
        if torch.is_tensor(points):
            pts_t = points.to(device=self.device, dtype=torch.float32)
        else:
            pts_t = torch.tensor(np.array(points), device=self.device, dtype=torch.float32)

        self.initial_poses = pts_t.detach().clone()
        self.N = pts_t.shape[0]

        # 固定起点和终点
        self.fixed_indices = [0, self.N - 1]
        self.variable_indices = list(range(1, self.N - 1))
        
        self.start_pose = self.initial_poses[0].clone() # (x, y, yaw)
        self.end_pose = self.initial_poses[-1].clone()

        # 将中间点作为可优化的参数 (x, y, yaw)
        if self.variable_indices:
            var_poses = self.initial_poses[self.variable_indices].clone()
            self.variable_poses = torch.nn.Parameter(var_poses) # Shape: (N-2, 3)
        else:
            self.variable_poses = None

        # --- 2. 处理占据地图 ---
        # occupancy_map assumed shape (D, H, W)
        self.occupancy_map = occupancy_map.to(device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Shape: [1, 1, D, H, W] for grid_sample
        self.map_res = map_info['resolution']
        self.map_origin = map_info['origin'] # (x, y, yaw)
        self.map_size_pixels = occupancy_map.shape # (D, H, W)

        # --- 3. 样条参数化 ---
        # # 把 (x,y,yaw) 当作 3D 空间坐标来参数化 t（不做角度 unwrap）
        # # 注意：这里我们保留 rot_scale 作为可选项，但默认将其设为 0.0，让 t 只由平移 (x,y) 决定，避免 yaw 导致参数化不一致而使样条看起来不平滑。
        # rot_scale = 0.0  # 将 yaw 差值映射为“等效距离”的缩放；设为 0 则 yaw 不影响 t
        # xy = self.initial_poses[:, :2].to(self.device)
        # yaw = self.initial_poses[:, 2].to(self.device)
        # diffs_xy = xy[1:] - xy[:-1]
        # trans_dists = torch.norm(diffs_xy, dim=1)  # (N-1,)

        # # **注意**：此处如需把 yaw 也参与参数化，可将 rot_scale 设置为非零；
        # # 但为了避免 yaw 的跳变（或 wrap）引起 t 不连续，默认不参与。
        # # yaw_diffs = (yaw[1:] - yaw[:-1]).abs()  # 直接绝对差，保留但默认不影响（rot_scale=0）
        # yaw_diffs_raw = yaw[1:] - yaw[:-1]
        # yaw_diffs = torch.abs(torch.atan2(torch.sin(yaw_diffs_raw), torch.cos(yaw_diffs_raw)))

        # total_dists = torch.sqrt(trans_dists**2 + (rot_scale * yaw_diffs)**2) + 1e-8
        # t = torch.cat([torch.tensor([0.0], device=self.device), torch.cumsum(total_dists, dim=0)])
        # t = (t / (t[-1] if t[-1] > 0 else 1.0)).cpu().numpy()
        # self.t_points = torch.tensor(t, device=self.device, dtype=torch.float32).detach()

        # 关键修改：统一使用固定参数化策略
        self.parameterization_mode = 'uniform'  # 'uniform' 或 'geometric'
        
        if self.parameterization_mode == 'uniform':
            # 使用均匀参数化
            self.t_points = torch.linspace(0, 1, self.N, device=self.device, dtype=torch.float32)
        else:
            # 使用几何参数化（原有逻辑）
            rot_scale = 0.0
            xy = pts_t[:, :2]
            diffs_xy = xy[1:] - xy[:-1]
            trans_dists = torch.norm(diffs_xy, dim=1)
            yaw_diffs_raw = pts_t[1:, 2] - pts_t[:-1, 2]
            yaw_diffs = torch.abs(self._normalize_angle_diff(yaw_diffs_raw))
            total_dists = torch.sqrt(trans_dists**2 + (rot_scale * yaw_diffs)**2) + 1e-8
            t = torch.cat([torch.tensor([0.0], device=self.device), torch.cumsum(total_dists, dim=0)])
            t = (t / (t[-1] if t[-1] > 0 else 1.0))
            self.t_points = t.to(device=self.device, dtype=torch.float32)

        # --- 4. 预计算参考轨迹 (用于 follow_cost) ---
        self.K_cost = 400
        self.t_dense = torch.linspace(0, 1, self.K_cost, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            self.Sx_ref, self.Sy_ref, self.Syaw_ref, *_ = self._evaluate_spline_se2(self.initial_poses, self.t_dense)


    def _assemble_control_poses(self):
        """将固定的端点和可变的中间点组合成完整的控制点序列"""
        if self.variable_poses is None:
            return torch.stack([self.start_pose, self.end_pose])
        return torch.cat([self.start_pose.unsqueeze(0), self.variable_poses, self.end_pose.unsqueeze(0)], dim=0)


    def _solve_natural_cubic_M(self, y):
        """为自然三次样条求解二次导数 M (与原版相同)"""
        N = self.N
        t = self.t_points
        h = torch.clamp(t[1:] - t[:-1], min=1e-6)
        
        A = torch.zeros((N, N), device=self.device, dtype=torch.float32)
        rhs = torch.zeros(N, device=self.device, dtype=torch.float32)

        A[0, 0] = 1.0
        A[-1, -1] = 1.0

        if N > 2:
            idx = torch.arange(1, N - 1, device=self.device)
            A[idx, idx - 1] = h[idx - 1]
            A[idx, idx] = 2.0 * (h[idx - 1] + h[idx])
            A[idx, idx + 1] = h[idx]
            rhs[idx] = 6.0 * ((y[idx + 1] - y[idx]) / h[idx] - (y[idx] - y[idx - 1]) / h[idx - 1])
        
        # 增加抖动以保证数值稳定性
        jitter = 1e-6 * torch.eye(N, device=self.device)
        M = torch.linalg.solve(A + jitter, rhs.unsqueeze(1)).squeeze(1)
        return M
    
    def _normalize_angle_diff(self, angle_diff):
        """将角度差值规范化到 [-π, π] 范围，考虑周期性"""
        return torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

    def _evaluate_spline_se2(self, control_poses, t_eval):
        """完全修复的版本：确保参数化依赖于控制点坐标"""
        device = control_poses.device
        N_current = control_poses.shape[0]
        
        # 使用固定的均匀参数化，不依赖于控制点位置
        t_ctrl = torch.linspace(0, 1, N_current, device=device, dtype=torch.float32)
        
        # 执行插值（确保使用计算出的 t_ctrl）
        x_ctrl, y_ctrl, yaw_ctrl = control_poses.T
        Sx, Sx_dot, Sx_ddot = self._evaluate_scalar_spline(x_ctrl, t_eval, t_ctrl=t_ctrl)
        Sy, Sy_dot, Sy_ddot = self._evaluate_scalar_spline(y_ctrl, t_eval, t_ctrl=t_ctrl)
        Syaw, Syaw_dot, Syaw_ddot = self._evaluate_yaw_spline(yaw_ctrl, t_eval, t_ctrl=t_ctrl)
        
        return Sx, Sy, Syaw, Sx_dot, Sy_dot, Syaw_dot, Sx_ddot, Sy_ddot, Syaw_ddot

        # # 根据当前 control_poses 重新计算 t_ctrl，正确处理 yaw 的周期性
        # rot_scale = 0.0
        # xy = control_poses[:, :2]
        # diffs_xy = xy[1:] - xy[:-1]
        # trans_dists = torch.norm(diffs_xy, dim=1)
        
        # # 修复：正确计算 yaw 的最短角度差
        # yaw_diffs_raw = yaw_ctrl[1:] - yaw_ctrl[:-1]
        # yaw_diffs = torch.abs(self._normalize_angle_diff(yaw_diffs_raw))
        
        # total_dists = torch.sqrt(trans_dists**2 + (rot_scale * yaw_diffs)**2) + 1e-8
        # t_local = torch.cat([torch.tensor([0.0], device=self.device), torch.cumsum(total_dists, dim=0)])
        # t_local = (t_local / (t_local[-1] if t_local[-1] > 0 else 1.0)).to(device=self.device, dtype=torch.float32)

        # # 对 x, y 直接插值
        # Sx, Sx_dot, Sx_ddot = self._evaluate_scalar_spline(x_ctrl, t_eval, t_ctrl=t_local)
        # Sy, Sy_dot, Sy_ddot = self._evaluate_scalar_spline(y_ctrl, t_eval, t_ctrl=t_local)

        # # 修复：对 yaw 进行周期性感知的插值
        # Syaw, Syaw_dot, Syaw_ddot = self._evaluate_yaw_spline(yaw_ctrl, t_eval, t_ctrl=t_local)

        # return Sx, Sy, Syaw, Sx_dot, Sy_dot, Syaw_dot, Sx_ddot, Sy_ddot, Syaw_ddot

    def _evaluate_yaw_spline(self, yaw_ctrl, t_eval, t_ctrl=None):
        """对 yaw 角度进行周期性感知的样条插值"""
        if t_ctrl is None:
            t_ctrl = self.t_points
        else:
            t_ctrl = t_ctrl.to(device=self.device, dtype=torch.float32)

        N_local = t_ctrl.shape[0]
        
        # 将 yaw 序列展开（unwrap），消除跳跃
        yaw_unwrapped = self._unwrap_angles(yaw_ctrl)
        
        # 对展开后的角度进行标量插值
        M = self._solve_natural_cubic_M(yaw_unwrapped)
        
        h = torch.clamp(t_ctrl[1:] - t_ctrl[:-1], min=1e-6)
        
        idx = (torch.searchsorted(t_ctrl, t_eval, right=True) - 1).clamp(0, N_local - 2)
        
        t_k = t_ctrl[idx]
        t_k1 = t_ctrl[idx + 1]
        h_k = t_k1 - t_k
        dt = t_eval - t_k

        y_k = yaw_unwrapped[idx]
        y_k1 = yaw_unwrapped[idx + 1]
        M_k = M[idx]
        M_k1 = M[idx + 1]
        
        # 插值公式
        term1 = M_k * (t_k1 - t_eval)**3 / (6 * h_k)
        term2 = M_k1 * dt**3 / (6 * h_k)
        term3 = (y_k - M_k * h_k**2 / 6) * (t_k1 - t_eval) / h_k
        term4 = (y_k1 - M_k1 * h_k**2 / 6) * dt / h_k
        S_unwrapped = term1 + term2 + term3 + term4
        
        # 一阶导
        S_dot = -M_k * (t_k1 - t_eval)**2 / (2 * h_k) + M_k1 * dt**2 / (2 * h_k) \
                - (y_k - M_k * h_k**2 / 6) / h_k + (y_k1 - M_k1 * h_k**2 / 6) / h_k

        # 二阶导
        S_ddot = (M_k * (t_k1 - t_eval) + M_k1 * dt) / h_k
        
        # 将插值结果重新规范化到 [-π, π]
        S = torch.atan2(torch.sin(S_unwrapped), torch.cos(S_unwrapped))
        
        return S, S_dot, S_ddot

    def _unwrap_angles(self, angles):
        """展开角度序列，消除周期性跳跃"""
        if len(angles) <= 1:
            return angles
        
        unwrapped = torch.zeros_like(angles)
        unwrapped[0] = angles[0]
        
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i-1]
            diff = self._normalize_angle_diff(diff)
            unwrapped[i] = unwrapped[i-1] + diff
        
        return unwrapped

    def _evaluate_scalar_spline(self, y_ctrl, t_eval, t_ctrl=None):
        """对一维标量序列进行样条插值，返回值及其一阶/二阶导数。
        支持传入自定义的 t_ctrl（控制点参数化），否则使用 self.t_points。
        已修正 searchsorted 的用法以保证区间索引正确，避免边界歧义导致的不连续。"""
        # 允许传入替代的 t_ctrl（例如基于当前 control_poses 重新计算的参数化）
        if t_ctrl is None:
            t_ctrl = self.t_points
        else:
            t_ctrl = t_ctrl.to(device=self.device, dtype=torch.float32)

        N_local = t_ctrl.shape[0]
        M = self._solve_natural_cubic_M(y_ctrl)
        
        h = torch.clamp(t_ctrl[1:] - t_ctrl[:-1], min=1e-6)
        
        # 使用 right=True 后减一，确保 idx 指向区间的左端点 t_k（t_k <= t_eval < t_k1）
        idx = (torch.searchsorted(t_ctrl, t_eval, right=True) - 1).clamp(0, N_local - 2)
        
        t_k = t_ctrl[idx]
        t_k1 = t_ctrl[idx + 1]
        h_k = t_k1 - t_k
        dt = t_eval - t_k

        # 使用 gather 来按位置索引，避免广播歧义
        y_k = y_ctrl[idx]
        y_k1 = y_ctrl[idx + 1]
        M_k = M[idx]
        M_k1 = M[idx + 1]
        
        # 插值公式（与之前一致）
        term1 = M_k * (t_k1 - t_eval)**3 / (6 * h_k)
        term2 = M_k1 * dt**3 / (6 * h_k)
        term3 = (y_k - M_k * h_k**2 / 6) * (t_k1 - t_eval) / h_k
        term4 = (y_k1 - M_k1 * h_k**2 / 6) * dt / h_k
        S = term1 + term2 + term3 + term4
        
        # 一阶导 (速度)
        S_dot = -M_k * (t_k1 - t_eval)**2 / (2 * h_k) + M_k1 * dt**2 / (2 * h_k) \
                - (y_k - M_k * h_k**2 / 6) / h_k + (y_k1 - M_k1 * h_k**2 / 6) / h_k

        # 二阶导 (加速度)
        S_ddot = (M_k * (t_k1 - t_eval) + M_k1 * dt) / h_k
        
        return S, S_dot, S_ddot

    # def _evaluate_spline_se2(self, control_poses, t_eval):
    #     """对 SE(2) 控制点 (x, y, yaw) 进行样条插值，**把 yaw 视为普通标量**（不做周期/unwrap 处理）。
    #     关键修正：在插值前根据当前 control_poses 重新计算 t_ctrl，使参数化与控制点一致，避免因 t 固定而在控制点改变时造成不平滑。"""
    #     # control_poses: (N, 3)
    #     x_ctrl, y_ctrl, yaw_ctrl = control_poses.T

    #     # 根据当前 control_poses 重新计算 t_ctrl（默认仍使用 rot_scale=0，若需将 yaw 计入可手动调整）
    #     rot_scale = 0.0
    #     xy = control_poses[:, :2]
    #     diffs_xy = xy[1:] - xy[:-1]
    #     trans_dists = torch.norm(diffs_xy, dim=1)
    #     yaw_diffs = (yaw_ctrl[1:] - yaw_ctrl[:-1]).abs()
    #     total_dists = torch.sqrt(trans_dists**2 + (rot_scale * yaw_diffs)**2) + 1e-8
    #     t_local = torch.cat([torch.tensor([0.0], device=self.device), torch.cumsum(total_dists, dim=0)])
    #     t_local = (t_local / (t_local[-1] if t_local[-1] > 0 else 1.0)).to(device=self.device, dtype=torch.float32)

    #     # 对 x, y 直接插值（传入局部 t_ctrl）
    #     Sx, Sx_dot, Sx_ddot = self._evaluate_scalar_spline(x_ctrl, t_eval, t_ctrl=t_local)
    #     Sy, Sy_dot, Sy_ddot = self._evaluate_scalar_spline(y_ctrl, t_eval, t_ctrl=t_local)

    #     # 对 yaw：直接作为普通标量插值（无 unwrap），也使用局部 t_ctrl
    #     Syaw, Syaw_dot, Syaw_ddot = self._evaluate_scalar_spline(yaw_ctrl, t_eval, t_ctrl=t_local)

    #     return Sx, Sy, Syaw, Sx_dot, Sy_dot, Syaw_dot, Sx_ddot, Sy_ddot, Syaw_ddot
    
    # def _evaluate_spline_se2(self, control_poses, t_eval):
    #     """分段SE(2)插值，严格保证控制点处的位置和朝向一致"""
    #     N = len(control_poses)
    #     device = control_poses.device
        
    #     # 重新参数化
    #     xy = control_poses[:, :2]
    #     diffs_xy = xy[1:] - xy[:-1]
    #     trans_dists = torch.norm(diffs_xy, dim=1)
        
    #     yaw_diffs = torch.abs(control_poses[1:, 2] - control_poses[:-1, 2])
    #     yaw_diffs = torch.minimum(yaw_diffs, 2*np.pi - yaw_diffs)
        
    #     rot_scale = 0.1
    #     total_dists = torch.sqrt(trans_dists**2 + (rot_scale * yaw_diffs)**2) + 1e-8
        
    #     t_local = torch.cat([torch.tensor([0.0], device=device), torch.cumsum(total_dists, dim=0)])
    #     t_local = t_local / (t_local[-1] if t_local[-1] > 0 else 1.0)
        
    #     # 使用分段SE(2)插值，严格保证控制点约束
    #     poses, velocities, accelerations = self._piecewise_se2_interpolation(
    #         control_poses, t_local, t_eval
    #     )
        
    #     return poses[:, 0], poses[:, 1], poses[:, 2], \
    #         velocities[:, 0], velocities[:, 1], velocities[:, 2], \
    #         accelerations[:, 0], accelerations[:, 1], accelerations[:, 2]
            
    def _piecewise_se2_interpolation(self, control_poses, t_ctrl, t_eval):
        """基于约束求解的分段SE(2)插值"""
        N = len(control_poses)
        K = len(t_eval)
        device = control_poses.device
        
        # 找到每个评估点对应的区间索引
        indices = torch.searchsorted(t_ctrl, t_eval, right=False)
        indices = torch.clamp(indices - 1, 0, N - 2)
        
        # 边界处理
        mask_before = t_eval <= t_ctrl[0]
        mask_after = t_eval >= t_ctrl[-1]
        
        # 检查是否正好在控制点上
        mask_at_control = torch.zeros(K, dtype=torch.bool, device=device)
        for i in range(N):
            mask_at_control |= torch.abs(t_eval - t_ctrl[i]) < 1e-6
        
        mask_interior = ~(mask_before | mask_after | mask_at_control)
        
        # 初始化结果
        result_poses = torch.zeros((K, 3), device=device)
        result_velocities = torch.zeros((K, 3), device=device)
        result_accelerations = torch.zeros((K, 3), device=device)
        
        # 处理边界点
        if mask_before.any():
            result_poses[mask_before] = control_poses[0]
        if mask_after.any():
            result_poses[mask_after] = control_poses[-1]
        
        # 处理正好在控制点上的点
        if mask_at_control.any():
            for i in range(N):
                mask_i = torch.abs(t_eval - t_ctrl[i]) < 1e-6
                if mask_i.any():
                    result_poses[mask_i] = control_poses[i]
        
        # 处理内部点 - 基于约束求解的分段插值
        if mask_interior.any():
            t_eval_interior = t_eval[mask_interior]
            indices_interior = indices[mask_interior]
            
            # 提取段的端点
            t0 = t_ctrl[indices_interior]      # (K_interior,)
            t1 = t_ctrl[indices_interior + 1]  # (K_interior,)
            p0 = control_poses[indices_interior]      # (K_interior, 3)
            p1 = control_poses[indices_interior + 1]  # (K_interior, 3)
            
            # 归一化参数 s ∈ [0, 1]
            h = t1 - t0  # (K_interior,)
            s = (t_eval_interior - t0) / h  # (K_interior,)
            
            # 核心：基于约束求解每段的插值
            poses_interior, velocities_interior, accelerations_interior = self._solve_constrained_segment_interpolation(
                p0, p1, s, h
            )
            
            # 填入结果
            result_poses[mask_interior] = poses_interior
            result_velocities[mask_interior] = velocities_interior
            result_accelerations[mask_interior] = accelerations_interior
        
        return result_poses, result_velocities, result_accelerations

    def _solve_constrained_segment_interpolation(self, p0, p1, s, h):
        """
        改进的基于约束求解的段插值：
        强制前进方向，避免倒车现象
        """
        device = p0.device
        K = p0.shape[0]
        
        # 1. 提取端点信息
        x0, y0, yaw0 = p0[:, 0], p0[:, 1], p0[:, 2]  # (K,)
        x1, y1, yaw1 = p1[:, 0], p1[:, 1], p1[:, 2]  # (K,)
        
        # 2. 计算位移向量和方向
        dx = x1 - x0  # (K,)
        dy = y1 - y0  # (K,)
        disp_mag = torch.sqrt(dx*dx + dy*dy + 1e-8)
        displacement_angle = torch.atan2(dy, dx)  # 位移方向
        
        # 3. 计算朝向单位向量
        cos_yaw0 = torch.cos(yaw0)
        sin_yaw0 = torch.sin(yaw0)
        cos_yaw1 = torch.cos(yaw1)
        sin_yaw1 = torch.sin(yaw1)
        
        # 4. 强制前进策略 - 不再进行智能选择，直接使用前进方向
        # 所有段都强制使用前进方向
        forward0 = torch.ones_like(yaw0, dtype=torch.bool)  # 强制所有起点前进
        forward1 = torch.ones_like(yaw1, dtype=torch.bool)  # 强制所有终点前进
        
        # 5. 计算修正后的端点切线向量
        # 起点切线方向（强制前进）
        v0x = cos_yaw0  # 始终使用前进方向
        v0y = sin_yaw0  # 始终使用前进方向
        
        # 终点切线方向（强制前进）
        v1x = cos_yaw1  # 始终使用前进方向
        v1y = sin_yaw1  # 始终使用前进方向
        
        # 6. 计算朝向与位移方向的一致性，用于调整速度
        disp_dot_heading0 = dx * cos_yaw0 + dy * sin_yaw0  # 起点朝向与位移的点积
        disp_dot_heading1 = dx * cos_yaw1 + dy * sin_yaw1  # 终点朝向与位移的点积
        
        # 一致性度量：值越大表示朝向与位移方向越一致
        consistency0 = disp_dot_heading0 / (disp_mag + 1e-8)  # 归一化的一致性 [-1, 1]
        consistency1 = disp_dot_heading1 / (disp_mag + 1e-8)
        
        # 7. 智能速度缩放 - 根据一致性调整速度
        base_speed_scale = 1.5
        
        # 当一致性较低时（朝向与位移方向冲突），减小速度以减少插值扭曲
        # 但确保速度始终为正（前进）
        scale_factor0 = torch.clamp(0.3 + 0.7 * torch.relu(consistency0), min=0.3, max=1.5)
        scale_factor1 = torch.clamp(0.3 + 0.7 * torch.relu(consistency1), min=0.3, max=1.5)
        
        # 8. 特殊处理：避免极端的小半径转弯
        min_displacement = 1.0  # 最小位移阈值
        max_angle_change = np.pi / 12  # 最大允许角度变化（30度）
        
        small_displacement = disp_mag < min_displacement
        angle_change = torch.abs(torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0)))
        large_angle_change = angle_change > max_angle_change
        
        problematic_segments = small_displacement & large_angle_change
        
        # 对于问题段，进一步降低速度以减少插值扭曲
        conservative_factor = torch.where(problematic_segments, 0.2, 1.0)
        scale_factor0 *= conservative_factor
        scale_factor1 *= conservative_factor
        
        speed0 = base_speed_scale * scale_factor0 * disp_mag / h
        speed1 = base_speed_scale * scale_factor1 * disp_mag / h
        
        # 9. 构造端点切线向量
        v0 = torch.stack([speed0 * v0x, speed0 * v0y, torch.zeros_like(speed0)], dim=1)
        v1 = torch.stack([speed1 * v1x, speed1 * v1y, torch.zeros_like(speed1)], dim=1)
        
        # 10. 处理yaw插值
        dyaw = yaw1 - yaw0
        dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))  # 处理周期性
        
        # 添加yaw平滑度约束
        yaw_curvature_penalty = torch.abs(dyaw) / (h + 1e-8)
        # 当角度变化过大时，减小angular velocity
        angular_scale = torch.clamp(1.0 - 0.5 * yaw_curvature_penalty, min=0.1, max=1.0)
        
        vyaw0 = angular_scale * dyaw / h
        vyaw1 = angular_scale * dyaw / h
        
        # vyaw0 = dyaw / h
        # vyaw1 = dyaw / h
        
        v0[:, 2] = vyaw0
        v1[:, 2] = vyaw1
        
        # 11. 诊断信息（可选）
        if hasattr(self, '_debug_interpolation') and self._debug_interpolation:
            n_problematic = torch.sum(problematic_segments).item()
            avg_consistency = (torch.mean(consistency0) + torch.mean(consistency1)) / 2
            print(f"Forward-only interpolation: {n_problematic} problematic segments, avg consistency: {avg_consistency.item():.3f}")
        
        # 12. 使用Hermite插值
        poses, velocities, accelerations = self._hermite_interpolate_constrained(
            p0, p1, v0, v1, s, h
        )
        
        return poses, velocities, accelerations

    # def _solve_constrained_segment_interpolation(self, p0, p1, s, h):
    #     """
    #     基于约束求解的段插值：
    #     对每段求解满足端点位置、朝向约束的三次插值曲线
    #     保留一致性检查，但强制前进方向
    #     """
    #     # p0, p1: (K, 3) - 段的起点和终点SE(2)姿态
    #     # s: (K,) - 归一化参数 [0,1]
    #     # h: (K,) - 时间间隔
        
    #     device = p0.device
    #     K = p0.shape[0]
        
    #     # 1. 提取端点信息
    #     x0, y0, yaw0 = p0[:, 0], p0[:, 1], p0[:, 2]  # (K,)
    #     x1, y1, yaw1 = p1[:, 0], p1[:, 1], p1[:, 2]  # (K,)
        
    #     # 2. 计算端点的朝向单位向量
    #     cos_yaw0 = torch.cos(yaw0)  # (K,)
    #     sin_yaw0 = torch.sin(yaw0)  # (K,)
    #     cos_yaw1 = torch.cos(yaw1)  # (K,)
    #     sin_yaw1 = torch.sin(yaw1)  # (K,)
        
    #     # 3. 计算位移向量
    #     dx = x1 - x0  # (K,)
    #     dy = y1 - y0  # (K,)
        
    #     # 4. 初步判断每个端点的方向偏好
    #     disp_dot_heading0 = dx * cos_yaw0 + dy * sin_yaw0  # (K,)
    #     disp_dot_heading1 = dx * cos_yaw1 + dy * sin_yaw1  # (K,)
        
    #     # 5. 关键：一致性检查和修正（保留原有逻辑）
    #     # 检查是否出现"两端都指向内部"或"两端都指向外部"的情况
        
    #     # 初始方向判断
    #     prefer_forward0 = disp_dot_heading0 >= 0
    #     prefer_forward1 = disp_dot_heading1 >= 0
        
    #     # 检测异常情况
    #     both_inward = (~prefer_forward0) & prefer_forward1   # 起点后退，终点前进 → 内凹
    #     both_outward = prefer_forward0 & (~prefer_forward1)  # 起点前进，终点后退 → 外凸
        
    #     # 对于异常情况，使用一致性修正策略
    #     forward0 = prefer_forward0.clone()
    #     forward1 = prefer_forward1.clone()
        
    #     # 策略1：对于内凹情况，优先保持起点方向，调整终点
    #     if both_inward.any():
    #         # 内凹：起点想后退，终点想前进
    #         # 选择总体位移大的方向作为主导方向
    #         abs_proj0 = torch.abs(disp_dot_heading0)
    #         abs_proj1 = torch.abs(disp_dot_heading1)
            
    #         # 修改：对于内凹情况，强制选择前进方向
    #         # 如果起点的投影更大，本来应该统一为后退，但现在强制为前进
    #         dominant_backward = (abs_proj0 >= abs_proj1) & both_inward
    #         forward0[dominant_backward] = True  # 强制前进
    #         forward1[dominant_backward] = True  # 强制前进
            
    #         # 如果终点的投影更大，则统一为前进（保持不变）
    #         dominant_forward = (abs_proj0 < abs_proj1) & both_inward
    #         forward0[dominant_forward] = True
    #         forward1[dominant_forward] = True
        
    #     # 策略2：对于外凸情况，同样强制前进
    #     if both_outward.any():
    #         # 外凸：起点想前进，终点想后退
    #         abs_proj0 = torch.abs(disp_dot_heading0)
    #         abs_proj1 = torch.abs(disp_dot_heading1)
            
    #         # 修改：对于外凸情况，强制选择前进方向
    #         # 如果起点的投影更大，则统一为前进（保持不变）
    #         dominant_forward = (abs_proj0 >= abs_proj1) & both_outward
    #         forward0[dominant_forward] = True
    #         forward1[dominant_forward] = True
            
    #         # 如果终点的投影更大，本来应该统一为后退，但现在强制为前进
    #         dominant_backward = (abs_proj0 < abs_proj1) & both_outward
    #         forward0[dominant_backward] = True  # 强制前进
    #         forward1[dominant_backward] = True  # 强制前进
        
    #     # 6. 附加检查：避免极端的反向情况，但强制前进
    #     # 如果某个端点的朝向与位移方向夹角大于135度，强制调整为前进
    #     cos_angle0 = disp_dot_heading0 / (torch.sqrt(dx*dx + dy*dy + 1e-8))
    #     cos_angle1 = disp_dot_heading1 / (torch.sqrt(dx*dx + dy*dy + 1e-8))
        
    #     # cos(135°) ≈ -0.707
    #     extreme_angle_threshold = -0.707
        
    #     # 修改：如果角度过于极端，强制设为前进而不是保持一致
    #     extreme0 = cos_angle0 < extreme_angle_threshold
    #     if extreme0.any():
    #         forward0[extreme0] = True  # 强制前进
        
    #     extreme1 = cos_angle1 < extreme_angle_threshold
    #     if extreme1.any():
    #         forward1[extreme1] = True  # 强制前进
        
    #     # 7. 最终强制：确保所有段都是前进方向
    #     # 这是额外的安全检查，确保没有遗漏的后退情况
    #     forward0 = torch.ones_like(forward0, dtype=torch.bool)  # 强制所有起点前进
    #     forward1 = torch.ones_like(forward1, dtype=torch.bool)  # 强制所有终点前进
        
    #     # 8. 计算修正后的端点切线向量
    #     # 起点切线方向（强制前进）
    #     v0x = cos_yaw0  # 强制前进方向
    #     v0y = sin_yaw0  # 强制前进方向
        
    #     # 终点切线方向（强制前进）
    #     v1x = cos_yaw1  # 强制前进方向
    #     v1y = sin_yaw1  # 强制前进方向
        
    #     # 9. 计算切线速度的大小，考虑一致性问题
    #     disp_mag = torch.sqrt(dx*dx + dy*dy + 1e-8)  # (K,)
        
    #     # 检查位移方向与端点朝向的一致性
    #     consistency0 = disp_dot_heading0 / (disp_mag + 1e-8)  # 归一化的一致性 [-1, 1]
    #     consistency1 = disp_dot_heading1 / (disp_mag + 1e-8)
        
    #     # 基于一致性调整速度缩放
    #     base_speed_scale = 1.5
        
    #     # 当一致性较低时（朝向与位移方向冲突），减小速度以减少插值扭曲
    #     scale_factor0 = torch.clamp(consistency0 * 0.3 + 0.7, min=0.2, max=1.0)  # [0.2, 1.0]
    #     scale_factor1 = torch.clamp(consistency1 * 0.3 + 0.7, min=0.2, max=1.0)  # [0.2, 1.0]
        
    #     speed0 = base_speed_scale * scale_factor0 * disp_mag / h  # (K,)
    #     speed1 = base_speed_scale * scale_factor1 * disp_mag / h  # (K,)
        
    #     # 10. 构造端点切线向量（带大小）
    #     v0 = torch.stack([speed0 * v0x, speed0 * v0y, torch.zeros_like(speed0)], dim=1)  # (K, 3)
    #     v1 = torch.stack([speed1 * v1x, speed1 * v1y, torch.zeros_like(speed1)], dim=1)  # (K, 3)
        
    #     # 11. 对yaw进行单独处理
    #     dyaw = yaw1 - yaw0  # (K,)
    #     dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))  # 处理周期性
        
    #     vyaw0 = dyaw / h  # (K,)
    #     vyaw1 = dyaw / h  # (K,)
        
    #     v0[:, 2] = vyaw0
    #     v1[:, 2] = vyaw1
        
    #     # # 12. 诊断信息（可选）
    #     # if torch.any(both_inward) or torch.any(both_outward):
    #     #     n_inward = torch.sum(both_inward).item()
    #     #     n_outward = torch.sum(both_outward).item()
    #     #     print(f"Trajectory consistency correction: {n_inward} inward, {n_outward} outward segments - all forced to forward")
        
    #     # 13. 使用Hermite插值
    #     poses, velocities, accelerations = self._hermite_interpolate_constrained(
    #         p0, p1, v0, v1, s, h
    #     )
        
    #     return poses, velocities, accelerations

    # def _solve_constrained_segment_interpolation(self, p0, p1, s, h):
    #     """
    #     基于约束求解的段插值：
    #     对每段求解满足端点位置、朝向约束的三次插值曲线
    #     增加朝向连续性检查
    #     """
    #     # p0, p1: (K, 3) - 段的起点和终点SE(2)姿态
    #     # s: (K,) - 归一化参数 [0,1]
    #     # h: (K,) - 时间间隔
        
    #     device = p0.device
    #     K = p0.shape[0]
        
    #     # 1. 提取端点信息
    #     x0, y0, yaw0 = p0[:, 0], p0[:, 1], p0[:, 2]  # (K,)
    #     x1, y1, yaw1 = p1[:, 0], p1[:, 1], p1[:, 2]  # (K,)
        
    #     # 2. 计算端点的朝向单位向量
    #     cos_yaw0 = torch.cos(yaw0)  # (K,)
    #     sin_yaw0 = torch.sin(yaw0)  # (K,)
    #     cos_yaw1 = torch.cos(yaw1)  # (K,)
    #     sin_yaw1 = torch.sin(yaw1)  # (K,)
        
    #     # 3. 计算位移向量
    #     dx = x1 - x0  # (K,)
    #     dy = y1 - y0  # (K,)
        
    #     # 4. 初步判断每个端点的方向偏好
    #     disp_dot_heading0 = dx * cos_yaw0 + dy * sin_yaw0  # (K,)
    #     disp_dot_heading1 = dx * cos_yaw1 + dy * sin_yaw1  # (K,)
        
    #     # 5. 关键：一致性检查和修正
    #     # 检查是否出现"两端都指向内部"或"两端都指向外部"的情况
        
    #     # 初始方向判断
    #     prefer_forward0 = disp_dot_heading0 >= 0
    #     prefer_forward1 = disp_dot_heading1 >= 0
        
    #     # 检测异常情况
    #     both_inward = (~prefer_forward0) & prefer_forward1   # 起点后退，终点前进 → 内凹
    #     both_outward = prefer_forward0 & (~prefer_forward1)  # 起点前进，终点后退 → 外凸
        
    #     # 对于异常情况，使用一致性修正策略
    #     forward0 = prefer_forward0.clone()
    #     forward1 = prefer_forward1.clone()
        
    #     # 策略1：对于内凹情况，优先保持起点方向，调整终点
    #     if both_inward.any():
    #         # 内凹：起点想后退，终点想前进
    #         # 选择总体位移大的方向作为主导方向
    #         abs_proj0 = torch.abs(disp_dot_heading0)
    #         abs_proj1 = torch.abs(disp_dot_heading1)
            
    #         # 如果起点的投影更大，则统一为后退
    #         dominant_backward = (abs_proj0 >= abs_proj1) & both_inward
    #         forward0[dominant_backward] = False
    #         forward1[dominant_backward] = False
            
    #         # 如果终点的投影更大，则统一为前进
    #         dominant_forward = (abs_proj0 < abs_proj1) & both_inward
    #         forward0[dominant_forward] = True
    #         forward1[dominant_forward] = True
        
    #     # 策略2：对于外凸情况，同样基于投影大小统一方向
    #     if both_outward.any():
    #         # 外凸：起点想前进，终点想后退
    #         abs_proj0 = torch.abs(disp_dot_heading0)
    #         abs_proj1 = torch.abs(disp_dot_heading1)
            
    #         # 如果起点的投影更大，则统一为前进
    #         dominant_forward = (abs_proj0 >= abs_proj1) & both_outward
    #         forward0[dominant_forward] = True
    #         forward1[dominant_forward] = True
            
    #         # 如果终点的投影更大，则统一为后退
    #         dominant_backward = (abs_proj0 < abs_proj1) & both_outward
    #         forward0[dominant_backward] = False
    #         forward1[dominant_backward] = False
        
    #     # 6. 附加检查：避免极端的反向情况
    #     # 如果某个端点的朝向与位移方向夹角大于135度，强制调整
    #     cos_angle0 = disp_dot_heading0 / (torch.sqrt(dx*dx + dy*dy + 1e-8))
    #     cos_angle1 = disp_dot_heading1 / (torch.sqrt(dx*dx + dy*dy + 1e-8))
        
    #     # cos(135°) ≈ -0.707
    #     extreme_angle_threshold = -0.707
        
    #     # 如果起点角度过于极端，强制与终点保持一致
    #     extreme0 = cos_angle0 < extreme_angle_threshold
    #     if extreme0.any():
    #         forward0[extreme0] = forward1[extreme0]
        
    #     # 如果终点角度过于极端，强制与起点保持一致
    #     extreme1 = cos_angle1 < extreme_angle_threshold
    #     if extreme1.any():
    #         forward1[extreme1] = forward0[extreme1]
        
    #     # 7. 计算修正后的端点切线向量
    #     # 起点切线方向
    #     v0x = torch.where(forward0, cos_yaw0, -cos_yaw0)  # (K,)
    #     v0y = torch.where(forward0, sin_yaw0, -sin_yaw0)  # (K,)
        
    #     # 终点切线方向
    #     v1x = torch.where(forward1, cos_yaw1, -cos_yaw1)  # (K,)
    #     v1y = torch.where(forward1, sin_yaw1, -sin_yaw1)  # (K,)
        
    #     # 8. 计算切线速度的大小（保持原有逻辑）
    #     disp_mag = torch.sqrt(dx*dx + dy*dy + 1e-8)  # (K,)
    #     speed_scale = 1.5  # 可调参数
        
    #     speed0 = speed_scale * disp_mag / h  # (K,)
    #     speed1 = speed_scale * disp_mag / h  # (K,)
        
    #     # 9. 构造端点切线向量（带大小）
    #     v0 = torch.stack([speed0 * v0x, speed0 * v0y, torch.zeros_like(speed0)], dim=1)  # (K, 3)
    #     v1 = torch.stack([speed1 * v1x, speed1 * v1y, torch.zeros_like(speed1)], dim=1)  # (K, 3)
        
    #     # 10. 对yaw进行单独处理（保持原有逻辑）
    #     dyaw = yaw1 - yaw0  # (K,)
    #     dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))  # 处理周期性
        
    #     vyaw0 = dyaw / h  # (K,)
    #     vyaw1 = dyaw / h  # (K,)
        
    #     v0[:, 2] = vyaw0
    #     v1[:, 2] = vyaw1
        
    #     # # 11. 诊断信息（可选）
    #     # if torch.any(both_inward) or torch.any(both_outward):
    #     #     n_inward = torch.sum(both_inward).item()
    #     #     n_outward = torch.sum(both_outward).item()
    #     #     print(f"Trajectory consistency correction: {n_inward} inward, {n_outward} outward segments corrected")
        
    #     # 12. 使用Hermite插值
    #     poses, velocities, accelerations = self._hermite_interpolate_constrained(
    #         p0, p1, v0, v1, s, h
    #     )
        
    #     return poses, velocities, accelerations

    def _hermite_interpolate_constrained(self, p0, p1, v0, v1, s, h):
        """
        约束版本的Hermite插值
        这里的v0, v1已经是根据约束求解得到的端点切线向量
        """
        # 标准Hermite基函数
        s2 = s * s
        s3 = s2 * s
        
        h00 = 2*s3 - 3*s2 + 1        # H₀(s)
        h10 = s3 - 2*s2 + s          # H₁(s)
        h01 = -2*s3 + 3*s2           # H₂(s)
        h11 = s3 - s2                # H₃(s)
        
        # 一阶导数
        h00_dot = 6*s2 - 6*s         # H₀'(s)
        h10_dot = 3*s2 - 4*s + 1     # H₁'(s)
        h01_dot = -6*s2 + 6*s        # H₂'(s)
        h11_dot = 3*s2 - 2*s         # H₃'(s)
        
        # 二阶导数
        h00_ddot = 12*s - 6          # H₀''(s)
        h10_ddot = 6*s - 4           # H₁''(s)
        h01_ddot = -12*s + 6         # H₂''(s)
        h11_ddot = 6*s - 2           # H₃''(s)
        
        # 扩展维度
        h00 = h00.unsqueeze(1)       # (K, 1)
        h10 = h10.unsqueeze(1)
        h01 = h01.unsqueeze(1)
        h11 = h11.unsqueeze(1)
        h_exp = h.unsqueeze(1)       # (K, 1)
        
        h00_dot = h00_dot.unsqueeze(1)
        h10_dot = h10_dot.unsqueeze(1)
        h01_dot = h01_dot.unsqueeze(1)
        h11_dot = h11_dot.unsqueeze(1)
        
        h00_ddot = h00_ddot.unsqueeze(1)
        h10_ddot = h10_ddot.unsqueeze(1)
        h01_ddot = h01_ddot.unsqueeze(1)
        h11_ddot = h11_ddot.unsqueeze(1)
        
        # Hermite插值公式
        poses = (h00 * p0 + h10 * h_exp * v0 + 
                h01 * p1 + h11 * h_exp * v1)
        
        # 处理yaw的周期性
        yaw_interp = poses[:, 2]
        yaw_normalized = torch.atan2(torch.sin(yaw_interp), torch.cos(yaw_interp))
        poses = torch.cat([
            poses[:, :2],  # x, y保持不变
            yaw_normalized.unsqueeze(1)  # 替换规范化后的yaw
        ], dim=1)
        
        # 速度：dp/dt = (dp/ds) * (ds/dt) = (dp/ds) / h
        dp_ds = (h00_dot * p0 + h10_dot * h_exp * v0 + 
                h01_dot * p1 + h11_dot * h_exp * v1)
        velocities = dp_ds / h_exp
        
        # 加速度：d²p/dt² = (d²p/ds²) / h²
        d2p_ds2 = (h00_ddot * p0 + h10_ddot * h_exp * v0 + 
                h01_ddot * p1 + h11_ddot * h_exp * v1)
        accelerations = d2p_ds2 / (h_exp * h_exp)
        
        return poses, velocities, accelerations

    def _compute_optimal_tangent_magnitudes(self, p0, p1, v0_dir, v1_dir, h):
        """
        给定端点位置和切线方向，求解最优的切线速度大小
        使插值曲线最优地连接两点
        """
        # p0, p1: (K, 3) - 端点位置
        # v0_dir, v1_dir: (K, 2) - 端点切线方向（单位向量）
        # h: (K,) - 时间间隔
        
        K = p0.shape[0]
        device = p0.device
        
        # 位移向量
        dx = p1[:, 0] - p0[:, 0]  # (K,)
        dy = p1[:, 1] - p0[:, 1]  # (K,)
        
        # 设置线性方程组求解切线速度大小
        # Hermite插值的约束：p(1) = p0 + v0*h/3 + p1 - v1*h/3
        # 重新整理：v0*h/3 - v1*h/3 = p1 - p0
        # 即：v0_mag * v0_dir * h/3 - v1_mag * v1_dir * h/3 = [dx, dy]
        
        # 构造线性方程组 A * [v0_mag, v1_mag]^T = b
        # 对于x分量：v0_mag * v0_dir[0] * h/3 - v1_mag * v1_dir[0] * h/3 = dx
        # 对于y分量：v0_mag * v0_dir[1] * h/3 - v1_mag * v1_dir[1] * h/3 = dy
        
        # 简化系数
        scale = h / 3.0  # (K,)
        
        # 构造系数矩阵 A: (K, 2, 2)
        A = torch.zeros((K, 2, 2), device=device)
        A[:, 0, 0] = v0_dir[:, 0] * scale  # v0_mag的x分量系数
        A[:, 0, 1] = -v1_dir[:, 0] * scale  # v1_mag的x分量系数
        A[:, 1, 0] = v0_dir[:, 1] * scale  # v0_mag的y分量系数
        A[:, 1, 1] = -v1_dir[:, 1] * scale  # v1_mag的y分量系数
        
        # 构造右端项 b: (K, 2)
        b = torch.stack([dx, dy], dim=1)  # (K, 2)
        
        # 求解线性方程组
        try:
            # 使用torch.linalg.solve求解
            speeds = torch.linalg.solve(A, b)  # (K, 2)
            v0_mag = speeds[:, 0]  # (K,)
            v1_mag = speeds[:, 1]  # (K,)
        except:
            # 如果求解失败，使用启发式方法
            disp_mag = torch.sqrt(dx*dx + dy*dy + 1e-8)
            v0_mag = disp_mag / h
            v1_mag = disp_mag / h
        
        return v0_mag, v1_mag
            
    # def _piecewise_se2_interpolation(self, control_poses, t_ctrl, t_eval):
    #     """完全张量化的分段SE(2)插值"""
    #     N = len(control_poses)
    #     K = len(t_eval)
    #     device = control_poses.device
        
    #     # 找到每个评估点对应的区间索引
    #     indices = torch.searchsorted(t_ctrl, t_eval, right=False)
    #     indices = torch.clamp(indices - 1, 0, N - 2)
        
    #     # 边界处理
    #     mask_before = t_eval <= t_ctrl[0]
    #     mask_after = t_eval >= t_ctrl[-1]
        
    #     # 检查是否正好在控制点上
    #     mask_at_control = torch.zeros(K, dtype=torch.bool, device=device)
    #     for i in range(N):
    #         mask_at_control |= torch.abs(t_eval - t_ctrl[i]) < 1e-6
        
    #     mask_interior = ~(mask_before | mask_after | mask_at_control)
        
    #     # 初始化结果
    #     result_poses = torch.zeros((K, 3), device=device)
    #     result_velocities = torch.zeros((K, 3), device=device)
    #     result_accelerations = torch.zeros((K, 3), device=device)
        
    #     # 处理边界点
    #     if mask_before.any():
    #         result_poses[mask_before] = control_poses[0]
    #     if mask_after.any():
    #         result_poses[mask_after] = control_poses[-1]
        
    #     # 处理正好在控制点上的点
    #     if mask_at_control.any():
    #         for i in range(N):
    #             mask_i = torch.abs(t_eval - t_ctrl[i]) < 1e-6
    #             if mask_i.any():
    #                 result_poses[mask_i] = control_poses[i]
        
    #     # 处理内部点 - 张量化版本
    #     if mask_interior.any():
    #         t_eval_interior = t_eval[mask_interior]
    #         indices_interior = indices[mask_interior]
            
    #         # 提取段的端点
    #         t0 = t_ctrl[indices_interior]      # (K_interior,)
    #         t1 = t_ctrl[indices_interior + 1]  # (K_interior,)
    #         p0 = control_poses[indices_interior]      # (K_interior, 3)
    #         p1 = control_poses[indices_interior + 1]  # (K_interior, 3)
            
    #         # 计算插值参数
    #         s = (t_eval_interior - t0) / (t1 - t0)  # (K_interior,)
            
    #         # 位置线性插值
    #         x_interp = (1 - s) * p0[:, 0] + s * p1[:, 0]  # (K_interior,)
    #         y_interp = (1 - s) * p0[:, 1] + s * p1[:, 1]  # (K_interior,)
            
    #         # 角度插值（考虑周期性）
    #         yaw0 = p0[:, 2]  # (K_interior,)
    #         yaw1 = p1[:, 2]  # (K_interior,)
    #         dyaw = yaw1 - yaw0
    #         dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))  # 最短角度差
    #         yaw_interp = yaw0 + s * dyaw
    #         yaw_interp = torch.atan2(torch.sin(yaw_interp), torch.cos(yaw_interp))
            
    #         # 速度计算
    #         dt = t1 - t0  # (K_interior,)
    #         vx = (p1[:, 0] - p0[:, 0]) / dt
    #         vy = (p1[:, 1] - p0[:, 1]) / dt  
    #         vyaw = dyaw / dt
            
    #         # 填入结果
    #         poses_interior = torch.stack([x_interp, y_interp, yaw_interp], dim=1)
    #         velocities_interior = torch.stack([vx, vy, vyaw], dim=1)
            
    #         result_poses[mask_interior] = poses_interior
    #         result_velocities[mask_interior] = velocities_interior
    #         result_accelerations[mask_interior] = 0.0
        
    #     return result_poses, result_velocities, result_accelerations

    def world_to_grid_normalized(self, points_world):
        """
        将世界坐标 (x, y, yaw) 转换为 grid_sample 所需的归一化坐标 [-1, 1].
        说明：**不对 yaw 做周期化**，直接按线性比例映射到 D 方向像素索引上。
        points_world: (K, 3) tensor
        """
        # Map shape: (D, H, W) -> yaw, y, x
        map_size_w = self.map_size_pixels[2]
        map_size_h = self.map_size_pixels[1]
        map_size_d = self.map_size_pixels[0]
        
        origin_x, origin_y, origin_yaw = self.map_origin
        
        # 转换到像素坐标 (x,y)
        px = (points_world[:, 0] - origin_x) / self.map_res
        py = (points_world[:, 1] - origin_y) / self.map_res
        
        # **直接线性映射 yaw -> 像素索引**（不做 modulo）
        yaw_range = 2 * np.pi
        pyaw = (points_world[:, 2] - origin_yaw) / (yaw_range / map_size_d)
        
        # 归一化到 [-1, 1] （grid_sample 坐标系：x 对应 W, y 对应 H, z 对应 D）
        norm_x = 2 * (px / (map_size_w - 1)) - 1
        norm_y = 2 * (py / (map_size_h - 1)) - 1
        norm_z = 2 * (pyaw / (map_size_d - 1)) - 1
        
        # grid_sample 需要的 shape 是 (N, D_out, H_out, W_out, 3)
        return torch.stack([norm_x, norm_y, norm_z], dim=-1).unsqueeze(0).unsqueeze(2).unsqueeze(3)


    def debug_cost_gradients(self, ctrl_poses=None, verbose=True):
        """
        诊断各损失项数值与梯度对可变控制点的影响。
        如果 ctrl_poses=None，则用 self._assemble_control_poses()。
        返回 dict，包括每项的标量值、梯度范数、以及对“倒车方向”的投影（正值表示该项在抑制倒车）。
        """
        device = self.device
        # 1. 准备控制点
        if ctrl_poses is None:
            ctrl_poses = self._assemble_control_poses()
        # 只计算相对于 variable_poses 的梯度：构造一个临时参数克隆用于追踪
        var_idx = self.variable_indices
        N = ctrl_poses.shape[0]

        # 2. 我们需要 variable 参数用于反向传播；创建一个临时 tensor that requires_grad
        if len(var_idx) > 0:
            tmp_vars = ctrl_poses[var_idx].detach().clone().to(device).requires_grad_(True)
            # 重建一个 full control poses tensor that uses tmp_vars
            cp = torch.cat([ctrl_poses[0:1].to(device), tmp_vars, ctrl_poses[-1:].to(device)], dim=0)
        else:
            tmp_vars = None
            cp = ctrl_poses.to(device)

        # 3. 计算样条并各项（复用你已有计算）
        K = self.K_cost if hasattr(self, 'K_cost') else 200
        t_dense = self.t_dense if hasattr(self, 't_dense') else torch.linspace(0,1,K,device=device)
        Sx, Sy, Syaw, Sx_dot, Sy_dot, Syaw_dot, Sx_ddot, Sy_ddot, Syaw_ddot = self._evaluate_spline_se2(cp, t_dense)

        # Sampled occupancy
        dense_traj_world = torch.stack([Sx, Sy, Syaw], dim=1)
        grid_coords = self.world_to_grid_normalized(dense_traj_world)
        occ_sample = F.grid_sample(self.occupancy_map, grid_coords, mode='bilinear', padding_mode='border', align_corners=True)
        occ = occ_sample.reshape(-1)

        # Compute terms (same math as in your cost_function)
        mean_occ = torch.mean(occ)
        # curvature
        eps = 1e-9
        speed = torch.sqrt(Sx_dot**2 + Sy_dot**2 + eps)
        geom_curv = torch.abs(Sx_dot * Sy_ddot - Sy_dot * Sx_ddot) / (speed**3 + 1e-9)
        v_thresh, gate_k = 0.08, 80.0
        gate = torch.sigmoid((speed - v_thresh) * gate_k)
        curvature_limit = 2.1
        curvature_violation = torch.relu(geom_curv - curvature_limit)
        curvature_cost = torch.mean(gate * curvature_violation)

        # yaw_per_meter
        v_eps = 1e-3
        yaw_per_meter = torch.abs(Syaw_dot) / (speed + v_eps)
        yaw_per_meter_limit = 2.0
        yaw_per_meter_violation = torch.relu(yaw_per_meter - yaw_per_meter_limit)
        yaw_per_meter_cost = torch.mean(yaw_per_meter_violation**2)

        # angle_diff
        tangent_angle = torch.atan2(Sy_dot, Sx_dot)
        angle_diff = torch.atan2(torch.sin(tangent_angle - Syaw), torch.cos(tangent_angle - Syaw))
        angle_diff_cost = torch.mean(torch.sin(angle_diff)**2)

        # control cost
        w_angular_vel = 0.1
        control_cost = torch.mean(Sx_dot**2 + Sy_dot**2) + w_angular_vel * torch.mean(Syaw_dot**2)

        # obstacle components diagnostic (mean + peak approx + slope)
        alpha = 25.0
        max_approx = (1.0/alpha) * torch.logsumexp(alpha * occ, dim=0)
        tau, beta = 0.45, 0.05
        soft_barrier = torch.mean(F.softplus((occ - tau) / (beta + 1e-12)))
        occ_diff = occ[1:] - occ[:-1] if occ.numel() >= 2 else torch.tensor([0.0], device=device)
        slope_penalty = torch.mean(torch.relu(occ_diff)**2)

        # Now form a dict of scalar terms
        terms = {
            'mean_occ': mean_occ,
            'max_approx': max_approx,
            'soft_barrier': soft_barrier,
            'slope_penalty': slope_penalty,
            'curvature_cost': curvature_cost,
            'yaw_per_meter_cost': yaw_per_meter_cost,
            'angle_diff_cost': angle_diff_cost,
            'control_cost': control_cost
        }

        # We will compute gradients of each term separately w.r.t tmp_vars (if exist), and compute:
        # - grad_norm: L2 norm of gradient
        # - dir_proj: projection of gradient onto "backward_direction" vector (positive => pushes against backward)
        results = {}
        if tmp_vars is None:
            # no variables -> just return term values
            for k,v in terms.items():
                results[k] = {'value': float(v.detach().cpu().item()), 'grad_norm': 0.0, 'back_proj': 0.0}
            return results

        # Precompute a "backward direction" basis per variable control point.
        # For each variable control point i, we compute its current yaw (ctrl_poses[var_idx[i],2]) and
        # define a small displacement vector that moves the point slightly *backwards* along its yaw:
        with torch.no_grad():
            var_cp = cp[var_idx]  # (M,3)
            yaws = var_cp[:,2]
            back_dirs = torch.zeros_like(var_cp)  # shape (M,3)
            back_dirs[:,0] = -0.01 * torch.cos(yaws)   # small step backward in x
            back_dirs[:,1] = -0.01 * torch.sin(yaws)   # small step backward in y
            back_dirs[:,2] = 0.0                       # keep yaw perturbation separate

        # For numeric stability, accumulate grads per variable as flattened vector
        for name, term in terms.items():
            # zero grads
            if tmp_vars.grad is not None:
                tmp_vars.grad.zero_()
            # backward
            term.backward(retain_graph=True)
            g = tmp_vars.grad.detach().clone()  # (M,3)
            grad_norm = float(g.norm().cpu().item())
            # compute projection: sum_i ( g_i dot back_dirs_i ) / (||back_dirs|| + eps)
            proj_num = torch.sum(g * back_dirs.to(device))
            proj_den = torch.sum(back_dirs.to(device)**2) + 1e-12
            back_proj = float((proj_num / proj_den).cpu().item())  # positive => gradient aligns with back_dirs (i.e., moves backward increases loss), so positive means term resists moving backward
            results[name] = {'value': float(term.detach().cpu().item()), 'grad_norm': grad_norm, 'back_proj': back_proj}

            # clear gradient for next term
            tmp_vars.grad.zero_()

        if verbose:
            print("\\n=== Cost Term Diagnostics ===")
            for k,v in results.items():
                print(f"{k:20s}: value={v['value']:.6e}, grad_norm={v['grad_norm']:.6e}, back_proj={v['back_proj']:.6e}")
            print("(back_proj > 0 => 该项在总体上抵抗向后(倒车)的位移；数值越大代表阻力越强)\n")

        return results


    def cost_function(self):
        """计算总成本"""
        ctrl_poses = self._assemble_control_poses()
        
        # 1. 插值得到密集轨迹
        Sx, Sy, Syaw, Sx_dot, Sy_dot, Syaw_dot, Sx_ddot, Sy_ddot, Syaw_ddot = self._evaluate_spline_se2(ctrl_poses, self.t_dense)
        
        # --- 成本项 ---
        
        # a) 占据/碰撞成本（occupancy_map 现在保存的是带符号的 ESDF，单位：米）
        dense_traj_world = torch.stack([Sx, Sy, Syaw], dim=1)
        grid_coords = self.world_to_grid_normalized(dense_traj_world)  # [1, K, 1, 1, 3]

        # 先在连续坐标上对 ESDF 做插值，再把插值后的距离映射为 cost（sigmoid）
        esdf_sample = F.grid_sample(
            self.occupancy_map,    # shape [1,1,D,H,W], 内容为 signed ESDF (m)
            grid_coords,           # [1, K, 1, 1, 3]
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # 输出形状类似 [1,1,K,1,1]

        esdf_flat = esdf_sample.reshape(-1)   # (K,) —— 插值后的 ESDF 值（米）

        d_safe = 0.15  # 安全距离阈值（米）
        kalpa = 0.08   # 平滑参数（米）
        z = (-(esdf_flat - d_safe) / (kalpa + 1e-12))
        occupancy_values = torch.sigmoid(torch.clamp(z, min=-50.0, max=50.0))

        obstacle_cost = torch.mean(occupancy_values)
        
        # # --- a) 占据/碰撞成本（替换开始） ---
        # # dense_traj_world: (K,3)
        # dense_traj_world = torch.stack([Sx, Sy, Syaw], dim=1)
        # grid_coords = self.world_to_grid_normalized(dense_traj_world)  # shape -> [1, K, 1, 1, 3]

        # # 从 cost_map（已经是 sigmoid(esdf->cost)）中采样
        # occ_sample = F.grid_sample(
        #     self.occupancy_map,    # shape [1,1,D,H,W]
        #     grid_coords,           # [1, K, 1, 1, 3]
        #     mode='bilinear',
        #     padding_mode='border',
        #     align_corners=True
        # )  # output shape [1,1,K,1,1] (可能有不同维度排列)，下面 flatten 成 1D

        # occ = occ_sample.reshape(-1)   # (K,) —— 每个采样点的 cost in [0,1]

        # # ---- 1) peak-focused term: log-sum-exp approximating max(risk) ----
        # # alpha 越大越接近 max；25 是经验值，可调 15~40
        # alpha = 25.0
        # max_approx = (1.0 / alpha) * torch.logsumexp(alpha * occ, dim=0)  # 标量，关注峰值

        # # ---- 2) soft barrier around threshold tau （保持超过阈值的平滑惩罚） ----
        # # tau 应根据你的 cost_map 值域选取(0~1)，0.4~0.6 为常用区间
        # tau = 0.45
        # beta = 0.05   # 温度，越小越陡峭但仍可微
        # soft_barrier = torch.mean(F.softplus((occ - tau) / (beta + 1e-12)))

        # # ---- 3) penalize steep uphill in risk along trajectory ----
        # # 避免“穿越陡峭上坡”进入窄高风险区（对 risk 的上升段惩罚更重）
        # # if occ.numel() >= 2:
        # #     occ_diff = occ[1:] - occ[:-1]         # (K-1,)
        # #     # pos_increase = torch.relu(occ_diff)   # 仅惩罚上升部分
        # #     # slope_penalty = torch.mean(pos_increase**2)
        # #     slope_penalty = torch.mean(occ_diff**2)
        # # else:
        # #     slope_penalty = torch.tensor(0.0, device=occ.device)
        
        # # 增加一个强惩罚（hard_penalty）：对超出较高阈值的点施加强约束，防止穿越明显不安全区
        # tau_hard = 0.55
        # hard_penalty_weight = 400.0
        # hard_violation = torch.relu(occ - tau_hard)
        # hard_penalty = hard_penalty_weight * torch.mean(hard_violation**2)

        # # ---- 4) 保留均值作为温和约束（防止极端只关注单点） ----
        # mean_occ = torch.mean(occ)

        # # ---- 组合成最终 obstacle_cost（可调权重） ----
        # w_max = 8.0
        # w_barrier = 6.0
        # w_slope = 12.0
        # w_penalty = 10  # hard_penalty 的权重
        # w_mean = 0.5

        # # obstacle_cost = w_max * max_approx + w_barrier * soft_barrier + w_slope * slope_penalty + w_mean * mean_occ
        # obstacle_cost = w_max * max_approx + w_barrier * soft_barrier + w_penalty * hard_penalty + w_mean * mean_occ
        # # --- a) 占据/碰撞成本（替换结束） ---

        # b) 平滑度成本 (对整个线进行评价，避免中间出现频繁抖动的曲线）
        # 使用二阶导数的平方和作为平滑度成本
        smoothness_cost = torch.mean(1e-0*Sx_ddot**2 + 1e-0*Sy_ddot**2 + 1e-0*Syaw_ddot**2)

        # c) 几何曲率
        eps = 1e-6
        speed = torch.sqrt(Sx_dot**2 + Sy_dot**2 + eps)
        geom_curvature = torch.abs(Sx_dot * Sy_ddot - Sy_dot * Sx_ddot) / (speed**3 + 1e-6)
        v_thresh = 0.08
        gate_k = 80.0
        gate = torch.sigmoid((speed - v_thresh) * gate_k)
        curvature_limit = 1.4
        curvature_violation = torch.relu(geom_curvature - curvature_limit)
        curvature_cost = torch.mean(gate * curvature_violation**1)*1e3
        # curvature_cost = torch.mean(curvature_violation**1)*1e3

        # d) 禁止原地大角度转动（yaw per meter）
        # 衡量每米的朝向变化量：yaw_per_meter = |yaw_rate| / (speed + v_eps)
        v_eps = 1e-3
        yaw_per_meter = torch.abs(Syaw_dot) / (speed + v_eps)
        # yaw_per_meter = torch.abs(Syaw_dot)
        # 允许一定的角度/米（例如 1 rad/m），超过则惩罚；使用平滑平方惩罚
        yaw_per_meter_limit = 1.4
        yaw_per_meter_violation = torch.relu(yaw_per_meter - yaw_per_meter_limit)
        yaw_per_meter_cost = torch.mean(yaw_per_meter_violation**2)*1e1
        
        # e) 控制量最小化成本 (惩罚过大的速度)
        w_angular_vel = 1e-1
        control_cost = 0*torch.mean(Sx_dot**2 + Sy_dot**2) + w_angular_vel * torch.mean(Syaw_dot**2)
        
        # # ---------- 替换 control_cost：惩罚横向加速度，弱惩切向加速度 ----------
        # # 目的：允许短时沿向(切向)加速（比如倒车起步/制动）而惩罚横向剧烈加速度（不稳定）
        # eps = 1e-6

        # # 速度向量与加速度向量
        # vx = Sx_dot       # (K,)
        # vy = Sy_dot
        # ax = Sx_ddot
        # ay = Sy_ddot

        # # speed magnitude (用于归一化)
        # speed = torch.sqrt(vx * vx + vy * vy + eps)  # (K,)

        # # 切向加速度（tangential a_t = (v·a) / |v|）
        # a_t = (vx * ax + vy * ay) / (speed + 1e-9)   # (K,)

        # # 横向加速度（lateral a_n = (v x a) / |v| = (vx*ay - vy*ax)/|v|）
        # a_n = (vx * ay - vy * ax) / (speed + 1e-9)   # (K,)

        # # 若你更愿意用朝向 yaw 作为基准（而不是速度方向），也可：
        # # heading_x = torch.cos(Syaw); heading_y = torch.sin(Syaw)
        # # a_t = (ax * heading_x + ay * heading_y)
        # # a_n = (ax * (-heading_y) + ay * heading_x)

        # # 权重：重点惩罚横向，加弱切向；角速度仍弱惩
        # w_lat = 1.0          # 横向加速度权重（主要项）
        # w_tan = 0.0          # 切向加速度权重（很小，允许短时沿向加速）
        # w_ang_vel = 0.05     # 角速度弱惩

        # lateral_cost = torch.mean(a_n**2)
        # tangential_cost = torch.mean(a_t**2)
        # angvel_cost = torch.mean(Syaw_dot**2)

        # control_cost = w_lat * lateral_cost + w_tan * tangential_cost + w_ang_vel * angvel_cost
        
        # f) 角度与切线方向的一致性约束（放宽以允许倒车）
        # 重要：为了允许倒车（heading 与速度方向相反，即 angle_diff ~ pi），
        # 我们使用 sin^2(angle_diff) 作为代价，它在 angle_diff=0 或 pi 时都为 0，
        # 因此既允许前向也允许倒车行驶。
        x_dot = Sx_dot
        y_dot = Sy_dot
        
        # 计算朝向的前进方向向量
        forward_x = torch.cos(Syaw)  # (K,)
        forward_y = torch.sin(Syaw)  # (K,)
        
        # 计算速度在朝向方向的投影（前进分量）
        speed_projection = x_dot * forward_x + y_dot * forward_y  # (K,)
        
        # 方法1：直接惩罚负投影（倒车）
        backward_penalty = torch.relu(-speed_projection)  # 只惩罚负值（倒车）
        backward_cost = torch.mean(backward_penalty**2)
        
        tangent_angle = torch.atan2(y_dot, x_dot)
        angle_diff = tangent_angle - Syaw
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))  # 规范到 [-pi, pi]
        # angle_diff_cost = torch.mean(torch.sin(angle_diff)**2)
        # angle_diff_cost = torch.mean(angle_diff**2)*0 + backward_cost  # 结合倒车惩罚
        angle_diff_cost = torch.mean(angle_diff**2)
        
        # g) 端点 yaw 对齐成本
        start_yaw_opt = ctrl_poses[0, 2]
        end_yaw_opt = ctrl_poses[-1, 2]
        start_yaw_interp = Syaw[0]   # 插值轨迹的起点 yaw
        end_yaw_interp = Syaw[-1]    # 插值轨迹的终点 yaw
        yaw0 = self.start_pose[2]
        yawn = self.end_pose[2]
        # 确保插值轨迹的端点 yaw 与固定端点 yaw 一致
        yaw_diff_start = torch.atan2(torch.sin(start_yaw_interp - yaw0), torch.cos(start_yaw_interp - yaw0))
        yaw_diff_end = torch.atan2(torch.sin(end_yaw_interp - yawn), torch.cos(end_yaw_interp - yawn))
        yaw_endpoint_cost = torch.mean(yaw_diff_start**2) + torch.mean(yaw_diff_end**2)
        # yaw_diff_start = torch.atan2(torch.sin(start_yaw_opt - yaw0), torch.cos(start_yaw_opt - yaw0))
        # yaw_diff_end = torch.atan2(torch.sin(end_yaw_opt - yawn), torch.cos(end_yaw_opt - yawn))
        # yaw_endpoint_cost = torch.mean(torch.abs(yaw_diff_start)) + torch.mean(torch.abs(yaw_diff_end))
        
        # h) 控制点连线轨迹光滑性损失
        # 计算控制点的xy连线的二阶导数
        if self.variable_poses is not None:
            ctrl_x = ctrl_poses[:, 0]
            ctrl_y = ctrl_poses[:, 1]
            ctrl_yaw = ctrl_poses[:, 2]
            # 计算控制点的二阶导数，用控制点间的差分计算
            ctrl_x_dot = ctrl_x[1:] - ctrl_x[:-1]
            ctrl_y_dot = ctrl_y[1:] - ctrl_y[:-1]
            ctrl_yaw_dot = ctrl_yaw[1:] - ctrl_yaw[:-1]
            ctrl_x_ddot = ctrl_x_dot[1:] - ctrl_x_dot[:-1]
            ctrl_y_ddot = ctrl_y_dot[1:] - ctrl_y_dot[:-1]
            ctrl_yaw_ddot = ctrl_yaw_dot[1:] - ctrl_yaw_dot[:-1]
            
            # 计算控制点连线的二阶导数平方和作为光滑性损失
            ctrl_smoothness_cost = torch.mean(1*ctrl_x_ddot**2 + 1*ctrl_y_ddot**2
                                            + 1e2 * ctrl_yaw_ddot**2)
            
        # i) 重新设计：基于累积弧长的均匀性损失
        # N = ctrl_poses.shape[0]  # 控制点数量
        # K = len(self.t_dense)   # 密集采样点数量
        
        # if N >= 2 and K >= 3:
        #     # 1. 计算密集轨迹的累积弧长
        #     dx_dense = Sx[1:] - Sx[:-1]  # (K-1,)
        #     dy_dense = Sy[1:] - Sy[:-1]  # (K-1,)
        #     segment_distances = torch.sqrt(dx_dense**2 + dy_dense**2 + eps)  # (K-1,)
            
        #     # 累积弧长：s[0]=0, s[i] = sum(segment_distances[0:i])
        #     cumulative_arclength = torch.cat([
        #         torch.zeros(1, device=self.device),
        #         torch.cumsum(segment_distances, dim=0)
        #     ])  # (K,)
            
        #     total_length = cumulative_arclength[-1]
            
        #     if total_length > 1e-8:
        #         # 2. 计算理想的均匀分布弧长位置
        #         ideal_positions = torch.linspace(0, total_length.cpu().item(), N, device=self.device)  # (N,)
                
        #         # 3. 找到每个控制点对应的实际弧长位置
        #         # 控制点在参数空间的位置
        #         t_ctrl = torch.linspace(0, 1, N, device=self.device)
                
        #         # 通过插值找到控制点对应的累积弧长
        #         actual_positions = torch.zeros(N, device=self.device)
        #         for i in range(N):
        #             t_val = t_ctrl[i]
        #             # 在 t_dense 中找到对应的索引（线性插值）
        #             if t_val <= 0:
        #                 actual_positions[i] = 0
        #             elif t_val >= 1:
        #                 actual_positions[i] = total_length
        #             else:
        #                 # 线性插值
        #                 idx_float = t_val * (K - 1)
        #                 idx_low = int(torch.floor(idx_float))
        #                 idx_high = min(idx_low + 1, K - 1)
        #                 alpha = idx_float - idx_low
                        
        #                 if idx_low == idx_high:
        #                     actual_positions[i] = cumulative_arclength[idx_low]
        #                 else:
        #                     actual_positions[i] = ((1 - alpha) * cumulative_arclength[idx_low] + 
        #                                         alpha * cumulative_arclength[idx_high])
                
        #         # 4. 计算均匀性损失：实际位置与理想均匀位置的差异
        #         position_errors = actual_positions - ideal_positions  # (N,)
                
        #         # 方法1：均方误差
        #         uniformity_cost_mse = torch.mean(position_errors**2) / (total_length**2 + 1e-8)
                
        #         # 方法2：相对误差的平方和
        #         relative_errors = position_errors / (total_length / (N - 1) + 1e-8)
        #         uniformity_cost_rel = torch.mean(relative_errors**2)
                
        #         # 方法3：最大绝对误差的平方（关注最坏情况）
        #         max_abs_error = torch.max(torch.abs(position_errors))
        #         uniformity_cost_max = (max_abs_error / (total_length / (N - 1) + 1e-8))**2
                
        #         # 方法4：相邻段长度差异的平方和
        #         segment_lengths = actual_positions[1:] - actual_positions[:-1]  # (N-1,)
        #         ideal_segment_length = total_length / (N - 1)
        #         length_deviations = segment_lengths - ideal_segment_length
        #         uniformity_cost_dev = torch.mean(length_deviations**2) / (ideal_segment_length**2 + 1e-8)
                
        #         # 组合多种损失，提供更稳定的梯度
        #         uniformity_cost = (0.1 * uniformity_cost_mse + 
        #                         0. * uniformity_cost_rel + 
        #                         0. * uniformity_cost_max + 
        #                         0. * uniformity_cost_dev)
                
        #         # 调试信息
        #         # print(f"Total length: {total_length.item():.4f}")
        #         # print(f"Ideal positions: {ideal_positions.detach().cpu().numpy()}")
        #         # print(f"Actual positions: {actual_positions.detach().cpu().numpy()}")
        #         # print(f"Position errors: {position_errors.detach().cpu().numpy()}")
        #         # print(f"Uniformity costs: MSE={uniformity_cost_mse.item():.6f}, "
        #         #     f"REL={uniformity_cost_rel.item():.6f}, "
        #         #     f"MAX={uniformity_cost_max.item():.6f}, "
        #         #     f"DEV={uniformity_cost_dev.item():.6f}")
        #     else:
        #         uniformity_cost = torch.tensor(0.0, device=self.device)
        # else:
        #     uniformity_cost = torch.tensor(0.0, device=self.device)
        
        # i) 重新设计：基于控制点间直线距离的均匀性损失
        N = ctrl_poses.shape[0]  # 控制点数量
                
        if N >= 2:
            # 1. 计算控制点之间的直线距离
            dx = ctrl_poses[1:, 0] - ctrl_poses[:-1, 0]  # (N-1,)
            dy = ctrl_poses[1:, 1] - ctrl_poses[:-1, 1]  # (N-1,)
            segment_distances = torch.sqrt(dx**2 + dy**2 + eps)  # (N-1,) 相邻控制点间的直线距离
            
            # 2. 计算总路径长度（所有线段距离之和）
            total_length = torch.sum(segment_distances)
            
            # 修复：降低阈值判断，添加调试信息
            if total_length > 1e-10:  # 降低阈值从 1e-8 到 1e-10
                # 3. 计算理想的均匀段长度
                ideal_segment_length = total_length / (N - 1)  # 理想情况下每段的长度
                
                # 4. 计算各种均匀性损失
                
                # 方法1：段长度方差（标准化）
                length_deviations = segment_distances - ideal_segment_length  # (N-1,)
                uniformity_cost_var = torch.var(segment_distances) / (ideal_segment_length**2 + 1e-12)
                
                # 方法2：段长度的均方偏差
                uniformity_cost_mse = torch.mean(length_deviations**2) / (ideal_segment_length**2 + 1e-12)
                
                # 方法3：相对偏差的平方和
                relative_deviations = length_deviations / (ideal_segment_length + 1e-12)
                uniformity_cost_rel = torch.mean(relative_deviations**2)
                
                # 方法4：最大偏差的平方（关注最不均匀的段）
                max_deviation = torch.max(torch.abs(length_deviations))
                uniformity_cost_max = (max_deviation / (ideal_segment_length + 1e-12))**2
                
                # 方法5：基于比值的损失（避免某段过长或过短）
                # 计算每段与理想长度的比值，理想情况下所有比值都应该接近1
                length_ratios = segment_distances / (ideal_segment_length + 1e-12)  # (N-1,)
                # 使用对数来惩罚比值偏离1的情况（对称地惩罚过长和过短）
                log_ratios = torch.log(torch.clamp(length_ratios, min=1e-12))  # 添加clamp防止log(0)
                uniformity_cost_ratio = torch.mean(log_ratios**2)
                
                # 修复：调整权重，确保至少一个方法有非零权重
                uniformity_cost = (1.0 * uniformity_cost_mse + 
                                0.0 * uniformity_cost_var + 
                                0.0 * uniformity_cost_rel + 
                                0.0 * uniformity_cost_max + 
                                0.0 * uniformity_cost_ratio)
                
                # 添加调试信息 - 临时开启
                # print(f"Debug uniformity cost:")
                # print(f"  N={N}, total_length={total_length.item():.6f}")
                # print(f"  ideal_segment_length={ideal_segment_length.item():.6f}")
                # print(f"  segment_distances={segment_distances.detach().cpu().numpy()}")
                # print(f"  length_deviations={length_deviations.detach().cpu().numpy()}")
                # print(f"  uniformity_cost_mse={uniformity_cost_mse.item():.6f}")
                # print(f"  uniformity_cost_var={uniformity_cost_var.item():.6f}")
                # print(f"  uniformity_cost_rel={uniformity_cost_rel.item():.6f}")
                # print(f"  uniformity_cost_max={uniformity_cost_max.item():.6f}")
                # print(f"  uniformity_cost_ratio={uniformity_cost_ratio.item():.6f}")
                # print(f"  final uniformity_cost={uniformity_cost.item():.6f}")
                
            else:
                # print(f"Debug: total_length too small: {total_length.item():.10f}")
                uniformity_cost = torch.tensor(0.0, device=self.device)
        else:
            # print(f"Debug: N={N} < 2, skipping uniformity cost")
            uniformity_cost = torch.tensor(0.0, device=self.device)
            
        # j) 超出地图范围的惩罚（修正：判断是否在 [-map_limit, map_limit] 外）
        map_limit = 20.0
        overflow_x = torch.relu(torch.abs(Sx) - map_limit)
        overflow_y = torch.relu(torch.abs(Sy) - map_limit)
        out_of_bounds_cost = torch.mean(overflow_x + overflow_y)
        
        # # j) 超出地图范围的惩罚（最终推荐版本）
        # """改进的边界惩罚函数 - 更激进的惩罚"""
        # map_limit = 20.0
        
        # # 计算超出距离
        # exceed_x = torch.abs(Sx) - map_limit
        # exceed_y = torch.abs(Sy) - map_limit
        
        # # 只对超出边界的点计算惩罚
        # violation_x = torch.relu(exceed_x)
        # violation_y = torch.relu(exceed_y)
        
        # # 总违反量
        # total_violation = violation_x + violation_y
        
        # # 多层惩罚策略
        # # 1. 线性惩罚（远距离仍有梯度）
        # linear_penalty = torch.sum(total_violation) * 1e6
        
        # # 2. 平方惩罚（中等距离快速增长）
        # square_penalty = torch.sum(total_violation**2) * 1e7
        
        # # 3. 指数惩罚（近距离陡峭惩罚）
        # exp_penalty = torch.sum(torch.exp(torch.clamp(total_violation, max=10.0)) - 1.0) * 1e5
        
        # # 4. 违反点数惩罚（每个超界点都有基础惩罚）
        # num_violations = torch.sum((violation_x > 0) | (violation_y > 0)).float()
        # count_penalty = num_violations * 1e8

        # out_of_bounds_cost = linear_penalty + square_penalty + exp_penalty + count_penalty

        # k) 轨迹总时间损失
        # 计算轨迹的总弧长作为时间的代理
        eps = 1e-6
        dx_dense = Sx[1:] - Sx[:-1]  # (K-1,)
        dy_dense = Sy[1:] - Sy[:-1]  # (K-1,)
        segment_distances = torch.sqrt(dx_dense**2 + dy_dense**2 + eps)  # (K-1,)
        
        # 使用梯形法则
        # 这里直接用速度的倒数乘以距离，等价于 ∫(1/v)ds
        speed_dense = torch.sqrt(Sx_dot**2 + Sy_dot**2 + eps)  # (K,)
        min_speed = 0.1  # 最小速度 (m/s)
        speed_dense_safe = torch.clamp(speed_dense, min=min_speed)

        # 梯形积分: ∫(1/v)ds ≈ Σ[(1/v_i + 1/v_{i+1})/2 * ds_i]
        inv_speed_start = 1.0 / speed_dense_safe[:-1]  # (K-1,)
        inv_speed_end = 1.0 / speed_dense_safe[1:]     # (K-1,)
        inv_speed_avg = (inv_speed_start + inv_speed_end) / 2  # (K-1,)
        total_time_trapezoid = torch.sum(inv_speed_avg * segment_distances)

        time_cost_dynamic = total_time_trapezoid

        time_cost = time_cost_dynamic  # 目前只使用基于动态模型的时间估计
        
        # l) 小速度惩罚损失
        eps = 1e-6
        speed = torch.sqrt(Sx_dot**2 + Sy_dot**2 + eps)
        # 当速度接近0时，惩罚快速增长
        speed_threshold = 2  # 开始惩罚的速度阈值
        decay_rate = 5.0  # 衰减率，越大惩罚增长越快
        exponential_penalty = torch.mean(torch.exp(-decay_rate * torch.clamp(speed - speed_threshold, min=0.0)))
        slow_speed_cost = exponential_penalty

        # --- 组合成本 ---
        weights = {
            'obstacle': 3e-3,
            'smoothness': 1e-7,
            'curvature': 1e-3,
            'yaw_per_meter': 1e-2, # 惩罚原地大角度转动的权重
            'control': 0e-7,
            'angle_diff': 1e-0, # 惩罚角度与切线方向不一致的权重
            'endpoints': 1e0,    # 强力惩罚端点 yaw 对齐
            'control_smoothness': 0e-3,  # 控制点连线的光滑性损失
            'uniformity': 1e-1,  # 轨迹段长度均匀性损失
            'out_of_bounds': 1e2,  # 超出地图范围的惩罚
            'time': 0e-3,  # 时间损失
            'slow_speed': 0e10,  # 小速度惩罚
        }
        # weights = {
        #     'obstacle': 3e-3,
        #     'smoothness': 1e-7,
        #     'curvature': 1e-2,
        #     'yaw_per_meter': 0e-1, # 惩罚原地大角度转动的权重
        #     'control': 1e-7,
        #     'angle_diff': 0e-5, # 惩罚角度与切线方向不一致的权重
        #     'endpoints': 0e0,    # 强力惩罚端点 yaw 对齐
        #     'control_smoothness': 0e-3,  # 控制点连线的光滑性损失
        #     'uniformity': 1e1,  # 轨迹段长度均匀性损失
        #     'out_of_bounds': 1e22,  # 超出地图范围的惩罚
        #     'time': 0e100000,  # 时间损失
        #     'slow_speed': 0e10,  # 小速度惩罚
        # }
        # weights = {
        #     'obstacle': 0e-3,
        #     'smoothness': 0e-7,
        #     'curvature': 0e-3,
        #     'yaw_per_meter': 0e0, # 惩罚原地大角度转动的权重
        #     'control': 0e-8,
        #     'angle_diff': 0e-5, # 惩罚角度与切线方向不一致的权重
        #     'endpoints': 0e0,    # 强力惩罚端点 yaw 对齐
        #     'control_smoothness': 0e-3,  # 控制点连线的光滑性损失
        #     'uniformity': 1e1,  # 轨迹段长度均匀性损失
        #     'out_of_bounds': 0e90,  # 超出地图范围的惩罚
        #     'time': 0e100000,  # 时间损失
        #     'slow_speed': 0e10,  # 小速度惩罚
        # }
         
        total_cost = (
            weights['obstacle'] * obstacle_cost +
            weights['smoothness'] * smoothness_cost +
            weights['curvature'] * curvature_cost +
            weights['yaw_per_meter'] * yaw_per_meter_cost +
            weights['control'] * control_cost +
            weights['angle_diff'] * angle_diff_cost +
            weights['endpoints'] * yaw_endpoint_cost +
            # weights['control_smoothness'] * ctrl_smoothness_cost if self.variable_poses is not None else 0.0 +
            weights['uniformity'] * uniformity_cost +
            weights['out_of_bounds'] * out_of_bounds_cost +
            weights['time'] * time_cost +
            weights['slow_speed'] * slow_speed_cost
        ) * 1e-1
         
        return total_cost
    
    def cost_on_poses(self, ctrl_poses):
        """
        直接对外部提供的控制点(ctrl_poses: [N,3])计算代价，
        保留输入张量的计算图（不创建新的 Parameter / detach），
        以便梯度能回传到外部（例如模型输出）。
        """
        device = self.device
        # 确保 ctrl_poses 在正确设备和 dtype
        ctrl = ctrl_poses.to(device=device, dtype=torch.float32)

        # 插值得到密集轨迹（复用已有接口）
        Sx, Sy, Syaw, Sx_dot, Sy_dot, Syaw_dot, Sx_ddot, Sy_ddot, Syaw_ddot = self._evaluate_spline_se2(ctrl, self.t_dense)

        # occupancy / obstacle cost
        dense_traj_world = torch.stack([Sx, Sy, Syaw], dim=1)
        grid_coords = self.world_to_grid_normalized(dense_traj_world)
        # esdf_sample = F.grid_sample(
        #     self.occupancy_map,
        #     grid_coords,
        #     mode='bilinear',
        #     padding_mode='border',
        #     align_corners=True
        # )
        # esdf_flat = esdf_sample.reshape(-1)
        # d_safe = 0.15
        # kalpa = 0.08
        # z = (-(esdf_flat - d_safe) / (kalpa + 1e-12))
        # occupancy_values = torch.sigmoid(torch.clamp(z, min=-50.0, max=50.0))
        # obstacle_cost = torch.mean(occupancy_values)
        
        # 从稳定性代价地图中采样
        stability_sample = F.grid_sample(
            self.occupancy_map,    # 这里应该是稳定性地图，不是ESDF
            grid_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        stability_flat = stability_sample.reshape(-1)   # (K,) 每个点的稳定性成本 [0,1]
        
        # === 多层倾覆惩罚策略 ===
        
        # 1. 基础倾覆成本 - 直接使用稳定性地图的值
        base_instability_cost = torch.mean(stability_flat)
        
        # 2. 高风险区域的指数惩罚
        high_risk_threshold = 0.3  # 超过此值认为是高风险
        high_risk_mask = stability_flat > high_risk_threshold
        if torch.any(high_risk_mask):
            high_risk_values = stability_flat[high_risk_mask]
            # 对高风险区域施加指数惩罚
            exp_penalty = torch.mean(torch.exp(5.0 * (high_risk_values - high_risk_threshold)))
        else:
            exp_penalty = torch.tensor(0.0, device=self.device)
        
        # 3. 极高风险区域的硬约束
        critical_threshold = 0.7  # 极度危险阈值
        critical_mask = stability_flat > critical_threshold
        critical_penalty = torch.sum(critical_mask.float()) * 1e2  # 每个危险点都有巨大惩罚
        
        # 4. 连续高风险段的额外惩罚（避免长时间处于不稳定状态）
        if len(stability_flat) > 1:
            risk_diff = stability_flat[1:] - stability_flat[:-1]
            # 惩罚进入高风险区域的行为
            entering_risk = torch.relu(risk_diff) * (stability_flat[1:] > high_risk_threshold).float()
            continuous_risk_penalty = torch.sum(entering_risk**2) * 100.0
        else:
            continuous_risk_penalty = torch.tensor(0.0, device=self.device)
        
        # 5. 最大风险点的强力惩罚
        max_risk = torch.max(stability_flat)
        max_risk_penalty = torch.relu(max_risk - 0.2)**3 * 1e4
        
        # 组合倾覆相关的所有惩罚
        obstacle_cost = (
            base_instability_cost * 10.0 +           # 基础成本
            exp_penalty * 0.0 +                     # 高风险指数惩罚  
            critical_penalty*0 +                        # 极度危险硬约束
            continuous_risk_penalty*0 +                 # 连续风险惩罚
            max_risk_penalty*100                             # 最大风险惩罚
        )*1e-3
        
        # === 其他成本项保持不变 ===

        # smoothness
        smoothness_cost = torch.mean(1e-0*Sx_ddot**2 + 1e-0*Sy_ddot**2 + 1e-0*Syaw_ddot**2)

        # curvature with low-speed gate
        eps = 1e-6
        speed = torch.sqrt(Sx_dot**2 + Sy_dot**2 + eps)
        geom_curvature = torch.abs(Sx_dot * Sy_ddot - Sy_dot * Sx_ddot) / (speed**3 + 1e-6)
        v_thresh = 0.08
        gate_k = 80.0
        gate = torch.sigmoid((speed - v_thresh) * gate_k)
        curvature_limit = 1.4
        # curvature_violation = torch.relu(geom_curvature - curvature_limit)
        # curvature_cost = torch.mean(gate * curvature_violation)*1e3
        curvature_violation = geom_curvature - curvature_limit
        # 使用软ReLU，避免梯度突变
        curvature_cost = torch.mean(F.softplus(curvature_violation, beta=2.0))*1e3

        # yaw per meter
        v_eps = 1e-3
        yaw_per_meter = torch.abs(Syaw_dot) / (speed + v_eps)
        yaw_per_meter_limit = 1.4
        yaw_per_meter_violation = torch.relu(yaw_per_meter - yaw_per_meter_limit)
        yaw_per_meter_cost = torch.mean(yaw_per_meter_violation**2)*1e1

        # control cost
        w_angular_vel = 0.1
        control_cost = 0.0*torch.mean(Sx_dot**2 + Sy_dot**2) + w_angular_vel * torch.mean(Syaw_dot**2)

        # angle diff cost (sin^2 保持允许倒车)
        tangent_angle = torch.atan2(Sy_dot, Sx_dot)
        angle_diff = tangent_angle - Syaw
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        # angle_diff_cost = torch.mean(torch.sin(angle_diff)**2)
        angle_diff_cost = torch.mean(1.0 - torch.cos(angle_diff))
        # angle_diff_cost = torch.mean(angle_diff**2)

        # endpoints yaw cost (使用 ctrl 的首尾)
        start_yaw_opt = ctrl[0, 2]
        end_yaw_opt = ctrl[-1, 2]
        start_yaw_interp = Syaw[0]   # 插值轨迹的起点 yaw
        end_yaw_interp = Syaw[-1]    # 插值轨迹的终点 yaw
        yaw0 = self.start_pose[2]
        yawn = self.end_pose[2]
        # 确保插值轨迹的端点 yaw 与固定端点 yaw 一致
        yaw_diff_start = torch.atan2(torch.sin(start_yaw_interp - yaw0), torch.cos(start_yaw_interp - yaw0))
        yaw_diff_end = torch.atan2(torch.sin(end_yaw_interp - yawn), torch.cos(end_yaw_interp - yawn))
        yaw_endpoint_cost = torch.mean(yaw_diff_start**2) + torch.mean(yaw_diff_end**2)
        # yaw_diff_start = torch.atan2(torch.sin(start_yaw_opt - yaw0), torch.cos(start_yaw_opt - yaw0))
        # yaw_diff_end = torch.atan2(torch.sin(end_yaw_opt - yawn), torch.cos(end_yaw_opt - yawn))
        # yaw_endpoint_cost = torch.mean(torch.abs(yaw_diff_start)) + torch.mean(torch.abs(yaw_diff_end))

        # control smoothness on control points if available (use ctrl directly)
        ctrl_smoothness_cost = torch.tensor(0.0, device=device)
        if ctrl.shape[0] >= 3:
            ctrl_x = ctrl[:, 0]
            ctrl_y = ctrl[:, 1]
            ctrl_x_dot = ctrl_x[1:] - ctrl_x[:-1]
            ctrl_y_dot = ctrl_y[1:] - ctrl_y[:-1]
            if ctrl_x_dot.shape[0] >= 2:
                ctrl_x_ddot = ctrl_x_dot[1:] - ctrl_x_dot[:-1]
                ctrl_y_ddot = ctrl_y_dot[1:] - ctrl_y_dot[:-1]
                ctrl_smoothness_cost = torch.mean(ctrl_x_ddot**2 + ctrl_y_ddot**2)

        # i) 重新设计：基于控制点间直线距离的均匀性损失
        N = ctrl_poses.shape[0]  # 控制点数量
                
        if N >= 2:
            # 1. 计算控制点之间的直线距离
            dx = ctrl_poses[1:, 0] - ctrl_poses[:-1, 0]  # (N-1,)
            dy = ctrl_poses[1:, 1] - ctrl_poses[:-1, 1]  # (N-1,)
            segment_distances = torch.sqrt(dx**2 + dy**2 + eps)  # (N-1,) 相邻控制点间的直线距离
            
            # 2. 计算总路径长度（所有线段距离之和）
            total_length = torch.sum(segment_distances)
            
            # 修复：降低阈值判断，添加调试信息
            if total_length > 1e-10:  # 降低阈值从 1e-8 到 1e-10
                # 3. 计算理想的均匀段长度
                ideal_segment_length = total_length / (N - 1)  # 理想情况下每段的长度
                
                # 4. 计算各种均匀性损失
                
                # 方法1：段长度方差（标准化）
                length_deviations = segment_distances - ideal_segment_length  # (N-1,)
                uniformity_cost_var = torch.var(segment_distances) / (ideal_segment_length**2 + 1e-12)
                
                # 方法2：段长度的均方偏差
                uniformity_cost_mse = torch.mean(length_deviations**2) / (ideal_segment_length**2 + 1e-12)
                
                # 方法3：相对偏差的平方和
                relative_deviations = length_deviations / (ideal_segment_length + 1e-12)
                uniformity_cost_rel = torch.mean(relative_deviations**2)
                
                # 方法4：最大偏差的平方（关注最不均匀的段）
                max_deviation = torch.max(torch.abs(length_deviations))
                uniformity_cost_max = (max_deviation / (ideal_segment_length + 1e-12))**2
                
                # 方法5：基于比值的损失（避免某段过长或过短）
                # 计算每段与理想长度的比值，理想情况下所有比值都应该接近1
                length_ratios = segment_distances / (ideal_segment_length + 1e-12)  # (N-1,)
                # 使用对数来惩罚比值偏离1的情况（对称地惩罚过长和过短）
                log_ratios = torch.log(torch.clamp(length_ratios, min=1e-12))  # 添加clamp防止log(0)
                uniformity_cost_ratio = torch.mean(log_ratios**2)
                
                # 修复：调整权重，确保至少一个方法有非零权重
                uniformity_cost = (1.0 * uniformity_cost_mse + 
                                0.0 * uniformity_cost_var + 
                                0.0 * uniformity_cost_rel + 
                                0.0 * uniformity_cost_max + 
                                0.0 * uniformity_cost_ratio)
                
                # 添加调试信息 - 临时开启
                # print(f"Debug uniformity cost:")
                # print(f"  N={N}, total_length={total_length.item():.6f}")
                # print(f"  ideal_segment_length={ideal_segment_length.item():.6f}")
                # print(f"  segment_distances={segment_distances.detach().cpu().numpy()}")
                # print(f"  length_deviations={length_deviations.detach().cpu().numpy()}")
                # print(f"  uniformity_cost_mse={uniformity_cost_mse.item():.6f}")
                # print(f"  uniformity_cost_var={uniformity_cost_var.item():.6f}")
                # print(f"  uniformity_cost_rel={uniformity_cost_rel.item():.6f}")
                # print(f"  uniformity_cost_max={uniformity_cost_max.item():.6f}")
                # print(f"  uniformity_cost_ratio={uniformity_cost_ratio.item():.6f}")
                # print(f"  final uniformity_cost={uniformity_cost.item():.6f}")
                
            else:
                # print(f"Debug: total_length too small: {total_length.item():.10f}")
                uniformity_cost = torch.tensor(0.0, device=self.device)
        else:
            # print(f"Debug: N={N} < 2, skipping uniformity cost")
            uniformity_cost = torch.tensor(0.0, device=self.device)

        # out of bounds
        map_limit = 20.0
        overflow_x = torch.relu(torch.abs(Sx) - map_limit)
        overflow_y = torch.relu(torch.abs(Sy) - map_limit)
        out_of_bounds_cost = torch.mean(overflow_x + overflow_y)

        # weights = {
        #     'obstacle': 2e1,
        #     'smoothness': 1e-4,
        #     'curvature': 5e2,
        #     'yaw_per_meter': 1e2,
        #     'control': 1e-2,
        #     'angle_diff': 4e4,
        #     'endpoints': 1e4,
        #     'control_smoothness': 0e3,
        #     'out_of_bounds': 1e8,
        # }
        
        # weights = {
        #     'obstacle': 1e1,
        #     'smoothness': 1e-4,
        #     'curvature': 5e2,
        #     'yaw_per_meter': 1e2,
        #     'control': 1e-2,
        #     'angle_diff': 1e4,
        #     'endpoints': 1e4,
        #     'control_smoothness': 0e3,
        #     'out_of_bounds': 1e8,
        # }
        
        # weights = {
        #     'obstacle': 3e-3,
        #     'smoothness': 1e-7,
        #     'curvature': 1e-3,
        #     'yaw_per_meter': 1e-2, # 惩罚原地大角度转动的权重
        #     'control': 0e-7,
        #     'angle_diff': 1e-0, # 惩罚角度与切线方向不一致的权重
        #     'endpoints': 1e0,    # 强力惩罚端点 yaw 对齐
        #     'control_smoothness': 0e-3,  # 控制点连线的光滑性损失
        #     'uniformity': 1e-1,  # 轨迹段长度均匀性损失
        #     'out_of_bounds': 1e2,  # 超出地图范围的惩罚
        # }

        weights = {
            'obstacle': 3e-4,
            'smoothness': 1e-8,
            'curvature': 0e-3,
            'yaw_per_meter': 0e-2, # 惩罚原地大角度转动的权重
            'control': 0e-7,
            'angle_diff': 1e-0, # 惩罚角度与切线方向不一致的权重
            'endpoints': 0e0,    # 强力惩罚端点 yaw 对齐
            'control_smoothness': 0e-3,  # 控制点连线的光滑性损失
            'uniformity': 1e-1,  # 轨迹段长度均匀性损失
            'out_of_bounds': 0e2,  # 超出地图范围的惩罚
        }

        total_cost = (
            weights['obstacle'] * obstacle_cost +
            weights['smoothness'] * smoothness_cost +
            weights['curvature'] * curvature_cost +
            weights['yaw_per_meter'] * yaw_per_meter_cost +
            weights['control'] * control_cost +
            weights['angle_diff'] * angle_diff_cost +
            weights['endpoints'] * yaw_endpoint_cost +
            # weights['control_smoothness'] * ctrl_smoothness_cost +
            weights['uniformity'] * uniformity_cost +
            weights['out_of_bounds'] * out_of_bounds_cost
        )
        
        total_cost = total_cost * 1e-1  # 统一缩放，避免数值过大
        
        return total_cost

    def optimize(self, iterations=300, lr=0.01, verbose=True):
        """执行优化循环"""
        if self.variable_poses is None:
            print("No variable points to optimize.")
            # 即使没有可变点，也返回初始插值轨迹
            with torch.no_grad():
                Sx, Sy, Syaw, *_ = self._evaluate_spline_se2(self.initial_poses, self.t_dense)
                traj = torch.stack([Sx, Sy], dim=1).cpu().numpy()
                yaw_dense = Syaw.cpu().numpy()
            return traj, yaw_dense, []
        
        optimizer = optim.Adam([self.variable_poses], lr=lr)
        cost_history = []
        
        best_cost = float('inf')
        best_poses = self.variable_poses.detach().clone()

        for i in range(iterations):
            optimizer.zero_grad()
            cost = self.cost_function()
            # self.debug_cost_gradients()
            cost.backward()
            torch.nn.utils.clip_grad_norm_([self.variable_poses], max_norm=1.0)
            optimizer.step()
            
            current_cost = cost.item()
            cost_history.append(current_cost)

            if current_cost < best_cost:
                best_cost = current_cost
                best_poses = self.variable_poses.detach().clone()

            if verbose and (i + 1) % 50 == 0:
                grad_norm = self.variable_poses.grad.norm().item()
                print(f"[{i+1}/{iterations}] Cost: {current_cost:.6f}, Grad Norm: {grad_norm:.6f}")

        # 应用找到的最佳参数
        with torch.no_grad():
            self.variable_poses.copy_(best_poses)
            final_poses = self._assemble_control_poses()
            Sx, Sy, Syaw, *_ = self._evaluate_spline_se2(final_poses, self.t_dense)
        
        traj = torch.stack([Sx, Sy], dim=1).cpu().numpy()
        yaw_dense = Syaw.cpu().numpy()

        return traj, yaw_dense, cost_history


def visualize_terrain_trajectory(ax, trajectory, elev, nx, ny, nz, positions=None, yaws=None):
    """可视化带有地形信息和轨迹的绘图"""
    # 确保张量转为numpy数组
    if isinstance(nx, torch.Tensor):
        elev = elev.cpu().numpy()
        nx = nx.cpu().numpy()
        ny = ny.cpu().numpy()
        nz = nz.cpu().numpy()
    
    # 创建地形高度图（使用法向量的Z分量）
    height_map = elev
    
    # 显示地形
    # im = ax.imshow(height_map, cmap='terrain', extent=[-5, 5, -5, 5], origin='lower')
    im = ax.imshow(height_map, cmap='terrain', extent=[-20, 20, -20, 20], origin='lower')
    plt.colorbar(im, ax=ax, label='Terrain Height')
    
    # 绘制轨迹
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'y-', linewidth=2, label='Optimized Trajectory')
    
    if positions is not None:
        # 绘制控制点
        ax.scatter(positions[:, 0], positions[:, 1], c='red', s=60, label='Control Points')
        ax.plot(positions[:, 0], positions[:, 1], 'r--', alpha=0.5)
    
    if yaws is not None and positions is not None:
        # 绘制控制点的朝向箭头（较小）
        for i in range(len(positions)):
            if i < len(yaws):
                arrow_length = 0.15
                dx = arrow_length * np.cos(yaws[i])
                dy = arrow_length * np.sin(yaws[i])
                ax.arrow(positions[i, 0], positions[i, 1], dx, dy, 
                         head_width=0.05, head_length=0.07, fc='cyan', ec='cyan')
    
    # 设置标题和图例
    ax.set_title('Terrain with Optimized Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

from ESDF3d_atpoint import compute_esdf_batch, query_is_unreachable_by_match_batch

def generate_stability_cost_map(nx, ny, nz, map_info, device='cuda'):
    """
    阶段 A: 生成 (x, y, yaw) 稳定性代价地图.
    """
    map_size = map_info['size'] # (W, H, D)
    origin = map_info['origin'] # (x, y, yaw)
    resolution = map_info['resolution']
    
    W, H, D = map_size
    
    print(f"Generating stability cost map of size {W}x{H}x{D}...")
    
    # 1. 创建所有查询点的网格
    # a. 创建每个维度的坐标
    x_coords = torch.linspace(origin[0], origin[0] + (W-1)*resolution, W, device=device)
    y_coords = torch.linspace(origin[1], origin[1] + (H-1)*resolution, H, device=device)
    yaw_range = 2 * np.pi
    yaw_coords = torch.linspace(origin[2], origin[2] + yaw_range * (D-1)/D, D, device=device)
    
    # b. 使用 meshgrid 创建三维坐标网格
    grid_y, grid_x, grid_yaw = torch.meshgrid(y_coords, x_coords, yaw_coords, indexing='ij')
    
    # c. 将网格展平为查询点列表 (num_points, 3)
    queries = torch.stack([
        grid_x.flatten(),
        grid_y.flatten(),
        grid_yaw.flatten()
    ], dim=1)
    
    # 2. 批量计算稳定性 (这里可以利用你原来的批量处理函数)
    print(f"Evaluating stability for {queries.shape[0]} points...")
    # 注意：如果点太多导致显存爆炸，需要分块(chunk)处理
    # 这里采用外部分块调用 compute_esdf_batch，并兼容不同返回类型
    
    num_points = queries.shape[0]
    chunk_q = 12000  # 每次处理的查询点数，必要时减小（例如 2000 或 1000）
    capsize_parts = []
    # 保证 nx, ny, nz 在调用端设备上（compute_esdf_batch 内部可能也会移动）
    nx_dev = nx.to(device)
    ny_dev = ny.to(device)
    nz_dev = nz.to(device)
    
    # 使用 tqdm 显示分块进度
    total_chunks = (num_points + chunk_q - 1) // chunk_q
    for i in tqdm.tqdm(range(0, num_points, chunk_q), desc="compute_esdf_batch chunks", total=total_chunks):
        q_chunk = queries[i:i+chunk_q].to(device)
        try:
            res_chunk = compute_esdf_batch(nx_dev, ny_dev, nz_dev, q_chunk,
                                           resolution=resolution,
                                           origin=(origin[0], origin[1]),
                                           yaw_weight=1.4,
                                           search_radius=5.0,
                                           chunk_cells=1000,
                                           device=device)
        except RuntimeError as e:
            # GPU OOM 或其它错误，尝试清理并在 CPU 上重试（更慢但稳妥）
            print(f"Warning: compute_esdf_batch failed on GPU for chunk {i}-{i+chunk_q}, retrying on CPU: {e}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            nx_cpu = nx_dev.cpu()
            ny_cpu = ny_dev.cpu()
            nz_cpu = nz_dev.cpu()
            q_chunk_cpu = q_chunk.cpu()
            res_chunk = compute_esdf_batch(nx_cpu, ny_cpu, nz_cpu, q_chunk_cpu,
                                           resolution=resolution,
                                           origin=(origin[0], origin[1]),
                                           yaw_weight=1.4,
                                           search_radius=5.0,
                                           chunk_cells=500,
                                           device='cpu')

        # 调用 compute_esdf_batch 后，兼容各种返回类型并优先提取每个条目的第一个值 (esdf_val)
        tensor_res = None
        if torch.is_tensor(res_chunk):
            tensor_res = res_chunk
        elif isinstance(res_chunk, np.ndarray):
            tensor_res = torch.from_numpy(res_chunk)
        elif isinstance(res_chunk, (list, tuple)):
            # 处理 compute_esdf_batch 返回的 list/tuple，每项通常为 (esdf_val, (i,j) or None, yaw_dist or None)
            if len(res_chunk) == 0:
                raise RuntimeError("compute_esdf_batch returned empty list/tuple")
            esdf_list = []
            for item in res_chunk:
                # 若 item 本身是 (esdf_val, ij, yaw_dist)
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    v = item[0]
                else:
                    v = item
                if torch.is_tensor(v):
                    esdf_list.append(v.detach().cpu().float())
                elif isinstance(v, np.ndarray):
                    esdf_list.append(torch.from_numpy(v).float())
                else:
                    # 标量或其他可转为 float 的类型
                    try:
                        esdf_list.append(torch.tensor(float(v), dtype=torch.float32))
                    except Exception:
                        raise RuntimeError("Cannot convert element of compute_esdf_batch result to tensor: " + str(type(v)))
            # 将每个标量/标量tensor 堆叠为 1D tensor，长度应等于 q_chunk.shape[0]
            tensor_res = torch.stack(esdf_list).reshape(-1)
        elif isinstance(res_chunk, dict):
            # 先尝试常见字段名
            found = False
            for key in ('capsize', 'cap', 'esdf', 'distance', 'dist', 'value', 'values'):
                if key in res_chunk:
                    v = res_chunk[key]
                    if torch.is_tensor(v):
                        tensor_res = v
                    elif isinstance(v, np.ndarray):
                        tensor_res = torch.from_numpy(v)
                    found = True
                    break
            if not found:
                # 递归查找 dict 内的第一个 tensor/ndarray
                for v in res_chunk.values():
                    if torch.is_tensor(v):
                        tensor_res = v; break
                    if isinstance(v, np.ndarray):
                        tensor_res = torch.from_numpy(v); break
            if tensor_res is None:
                raise RuntimeError("compute_esdf_batch returned dict but contained no recognizable tensor/ndarray")
        else:
            raise RuntimeError("compute_esdf_batch returned unsupported type: " + str(type(res_chunk)))

        # 统一为 CPU 上的 float32 tensor，减少显存占用
        tensor_res = tensor_res.detach().cpu().float()

        # 尝试根据常见布局提取 capsize 值（第0通道或直接 1D 列表）
        expected_len = q_chunk.shape[0]
        if tensor_res.ndim == 1:
            cap_chunk = tensor_res.reshape(-1)
        elif tensor_res.ndim == 0:
            cap_chunk = tensor_res.reshape(-1).repeat(expected_len)[:expected_len]
        else:
            # 优先取最后一维的第0通道（例如 [..., C] -> [...,0]），否则展平
            try:
                if tensor_res.shape[-1] >= 1:
                    cap_chunk = tensor_res[..., 0].reshape(-1)
                else:
                    cap_chunk = tensor_res.reshape(-1)
            except Exception:
                cap_chunk = tensor_res.reshape(-1)

        # 保证长度与 q_chunk 匹配，否则截取或填充 Inf（后续会处理）
        if cap_chunk.numel() < expected_len:
            pad = torch.full((expected_len - cap_chunk.numel(),), float('inf'), dtype=cap_chunk.dtype)
            cap_chunk = torch.cat([cap_chunk, pad], dim=0)
        elif cap_chunk.numel() > expected_len:
            cap_chunk = cap_chunk[:expected_len]

        capsize_parts.append(cap_chunk)

    capsize_esdf_flat = torch.cat(capsize_parts, dim=0)  # (num_points,)

    # 如果有 inf（因填充），将其替换为一个很大的距离值
    if torch.isfinite(capsize_esdf_flat).all() is False:
        capsize_esdf_flat = torch.where(torch.isfinite(capsize_esdf_flat),
                                        capsize_esdf_flat,
                                        torch.full_like(capsize_esdf_flat, 1e3))

    # 恢复到原有成本计算流程
    d_safe = 0.
    kalpa = 0.6
    z = (-(capsize_esdf_flat - d_safe) / (kalpa + 1e-12))
    costs = torch.sigmoid(torch.clamp(z, min=-50.0, max=50.0))

    # 3. 将一维的成本列表重塑为三维地图 (D, H, W)
    cost_map = costs.reshape(H, W, D).permute(2, 0, 1) # reshape 成 (H, W, D)，然后 permute 成 (D, H, W)
    
    print("Stability cost map generated.")
    return cost_map

def check_trajectory_reachability(trajectory_points, yaw_values, yaw_stability):
    """
    使用与 data_clean.py 一致的方法检查轨迹点的可达性
    
    Args:
        trajectory_points: numpy array of shape (N, 2) - x, y coordinates
        yaw_values: numpy array of shape (N,) - yaw angles
        yaw_stability: torch tensor of shape (H, W, 36) - stability map
    
    Returns:
        capsize_mask: numpy array of shape (N,) - True for unreachable points
    """
    capsize_mask = []
    
    for i in range(len(trajectory_points)):
        x, y = trajectory_points[i]
        yaw = yaw_values[i]
        
        # 与 data_clean.py 完全一致的坐标转换
        x_idx = int((x + 20) / 0.4)
        y_idx = int((y + 20) / 0.4)
        yaw_idx = int((yaw + np.pi) / (2 * np.pi / 36)) % 36
        
        # 边界检查和稳定性判断
        if 0 <= x_idx < yaw_stability.shape[0] and 0 <= y_idx < yaw_stability.shape[1]:
            yaw_stability_value = yaw_stability[x_idx, y_idx, yaw_idx]
            is_unreachable = (yaw_stability_value == 0)
        else:
            is_unreachable = True  # 超出地图边界
            
        capsize_mask.append(is_unreachable)
    
    return np.array(capsize_mask, dtype=bool)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def verify_optimizer_consistency():
        """验证优化器的一致性"""
        # 创建测试轨迹
        test_traj = torch.randn(5, 3, requires_grad=True, device=device)
        
        # 创建优化器
        optimizer = TrajectoryOptimizerSE2(test_traj.detach(), stability_cost_map, map_info, device=device)
        
        # 设置相同的控制点
        optimizer.variable_poses.data = test_traj[1:-1].detach()
        
        # 计算两种损失
        cost1 = optimizer.cost_function()
        cost2 = optimizer.cost_on_poses(test_traj)
        
        print(f"cost_function: {cost1.item():.6e}")
        print(f"cost_on_poses: {cost2.item():.6e}")
        print(f"Difference: {abs(cost1.item() - cost2.item()):.6e}")
        
        return abs(cost1.item() - cost2.item()) < 1e-6

    # --- 0. 加载地形数据 ---
    from dataLoader_uneven import UnevenPathDataLoader
    env_list = ['env000000']
    dataFolder = '/home/yrf/MPT/data/sim_dataset/val'
    dataset = UnevenPathDataLoader(env_list, dataFolder)
    path_index = 7
    sample = dataset[path_index]
    if sample is None:
        raise ValueError(f"Sample at index {path_index} is invalid.")
    print("Loaded path index:", path_index)
    
    nx = sample['map'][0, :, :].to(device)
    ny = sample['map'][1, :, :].to(device)
    nz = sample['map'][2, :, :].to(device)

    # --- 1. 定义代价地图参数并生成地图 ---
    map_size = (100, 100, 36) # W, H, D for (x, y, yaw)
    resolution = 0.4
    origin = (-20.0, -20.0, -np.pi) # x, y, yaw
    map_info = {
        'resolution': resolution,
        'origin': origin,
        'size': map_size
    }

    # !! 核心步骤：生成稳定性代价地图 !!
    # stability_cost_map = generate_stability_cost_map(nx, ny, nz, map_info, device)
    stability_cost_map = sample['cost_map'].to(device)  # 使用预先计算的稳定性代价地图

    # --- 2. 定义初始轨迹控制点 ---
    initial_points = sample['trajectory'].cpu().numpy()  # (x, y, yaw)

    # --- 使用网络推理结果作为初始点 ---
    # 生成模型预测轨迹
    from dataLoader_uneven import get_encoder_input
    from eval_model_uneven import get_patch
    from transformer import Models
    # from vision_mamba import Models
    import os.path as osp
    import json
    
    best = True
    # best = False
    stage = 1
    # epoch = 39
    # stage = 2
    # epoch = 24

    modelFolder = 'data/sim'
    # modelFolder = 'data/uneven_old'
    modelFile = osp.join(modelFolder, f'model_params.json')
    model_param = json.load(open(modelFile))

    transformer = Models.UnevenTransformer(**model_param)
    _ = transformer.to(device)

    # checkpoint = torch.load(osp.join(modelFolder, f'model_epoch_{epoch}.pkl'))
    if stage == 1:
        if best:
            checkpoint = torch.load(osp.join(modelFolder, f'best_stage1_model.pkl'))
        else:
            checkpoint = torch.load(osp.join(modelFolder, f'stage1_model_epoch_{epoch}.pkl'))
    else:
        if best:
            checkpoint = torch.load(osp.join(modelFolder, f'best_stage2_model.pkl'))
        else:
            checkpoint = torch.load(osp.join(modelFolder, f'stage2_model_epoch_{epoch}.pkl'))
    transformer.load_state_dict(checkpoint['state_dict'])
    
    trajectory = sample['trajectory'].cpu().numpy()  # (N, 3)
    goal_pos = trajectory[-1, :]  # 终点位置
    start_pos = trajectory[0, :]  # 起点位置
    
    normal_x = nx.cpu().numpy()
    normal_y = ny.cpu().numpy()
    normal_z = nz.cpu().numpy()
    
    encoder_input = get_encoder_input(normal_z, goal_pos, start_pos, normal_x, normal_y)
    patch_maps, predProb_list, predTraj = get_patch(transformer, start_pos, goal_pos, normal_x, normal_y, normal_z)
    # 确保 predTraj 是一个连续的 numpy/torch Tensor，然后再拼接起止点
    if isinstance(predTraj, list):
        pred_arr = np.asarray(predTraj, dtype=np.float32) if len(predTraj) > 0 else np.zeros((0, 3), dtype=np.float32)
        pred_traj_t = torch.from_numpy(pred_arr).to(device)
    elif torch.is_tensor(predTraj):
        pred_traj_t = predTraj.to(device).float()
    else:
        pred_traj_t = torch.from_numpy(np.asarray(predTraj, dtype=np.float32)).to(device)

    start_t = torch.tensor(start_pos, dtype=torch.float32, device=device).unsqueeze(0)
    goal_t  = torch.tensor(goal_pos,  dtype=torch.float32, device=device).unsqueeze(0)
    predTraj = torch.cat((start_t, pred_traj_t, goal_t), dim=0)
    
    initial_points = predTraj.cpu().numpy()  # 使用模型预测轨迹作为初始点

    # # 在训练前验证
    # if not verify_optimizer_consistency():
    #     print("Warning: Optimizer consistency check failed!")

    # --- 3. 初始化并运行优化器 (使用 TrajectoryOptimizerSE2) ---
    optimizer = TrajectoryOptimizerSE2(initial_points, stability_cost_map, map_info, device=device)
    
    # 获取优化前的轨迹
    with torch.no_grad():
        initial_poses_torch = torch.tensor(initial_points, device=device, dtype=torch.float32)
        Sx_i, Sy_i, *_ = optimizer._evaluate_spline_se2(initial_poses_torch, optimizer.t_dense)
        initial_trajectory = torch.stack([Sx_i, Sy_i], dim=1).cpu().numpy()

    # # 执行优化
    # optimized_trajectory, optimized_yaw_dense, cost_history = optimizer.optimize(iterations=800, lr=0.1, verbose=True)

    # 设置优化参数
    iterations = 800
    lr = 0.1
    grad_clip_norm = 1.0

    # 只优化中间点，固定起点和终点
    start_point = torch.tensor(initial_points[0], device=device, dtype=torch.float32)  # 固定起点
    end_point = torch.tensor(initial_points[-1], device=device, dtype=torch.float32)   # 固定终点

    middle_points = torch.tensor(initial_points[1:-1], device=device, dtype=torch.float32, requires_grad=True)
    
    # 将初始轨迹转为可优化的参数
    # 注意：这里我们优化整个轨迹，包括起点和终点（如果需要固定起终点，可以只优化中间点）
    # trajectory_params = torch.tensor(initial_points, device=device, dtype=torch.float32, requires_grad=True)
    
    # 初始化优化器
    optimizer_adam = torch.optim.Adam([middle_points], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_adam, T_max=iterations, eta_min=lr * 0.01
    )
    
    # 记录训练过程
    cost_history = []
    best_cost = float('inf')
    best_middle_points = middle_points.detach().clone()
    
    print("Starting neural network style optimization...")
    print(f"Parameters: iterations={iterations}, lr={lr}, grad_clip={grad_clip_norm}")
    print()
    
    # 主训练循环
    for iteration in range(iterations):
        # 1. 清零梯度
        optimizer_adam.zero_grad()
        
        # 2. 构建完整的轨迹参数（起点 + 中间点 + 终点）
        full_trajectory = torch.cat([
            start_point.unsqueeze(0),  # 起点
            middle_points,             # 可优化的中间点
            end_point.unsqueeze(0)     # 终点
        ], dim=0)

        # 2. 每次迭代创建新的优化器实例（关键步骤）
        # 使用当前轨迹参数创建优化器，但不进行内部优化
        trajectory_optimizer = TrajectoryOptimizerSE2(
            full_trajectory.detach(),  # 用detach避免影响计算图
            stability_cost_map, 
            map_info, 
            device=device
        )
        
        # 3. 使用 cost_on_poses 计算损失（保持梯度）
        cost = trajectory_optimizer.cost_on_poses(full_trajectory)

        # 4. 数值稳定性检查
        if torch.isnan(cost) or torch.isinf(cost):
            print(f"Warning: Invalid cost at iteration {iteration}: {cost.item()}")
            break
        
        # 5. 反向传播
        cost.backward()
        
        # 6. 梯度裁剪
        original_grad_norm = torch.nn.utils.clip_grad_norm_([middle_points], grad_clip_norm)
        was_clipped = original_grad_norm > grad_clip_norm
        
        # 7. 参数更新
        optimizer_adam.step()
        
        # 8. 学习率调度
        scheduler.step()
        
        # 9. 记录和跟踪
        current_cost = cost.item()
        cost_history.append(current_cost)
        current_lr = optimizer_adam.param_groups[0]['lr']
        
        # 10. 最佳模型保存
        if current_cost < best_cost:
            best_cost = current_cost
            best_middle_points = middle_points.detach().clone()

        # 11. 打印训练进度
        if (iteration + 1) % 25 == 0 or iteration == 0:
            grad_info = f'{original_grad_norm:.2e}→{grad_clip_norm:.1e}' if was_clipped else f'{original_grad_norm:.2e}'
            
            print(f"Iter [{iteration+1:4d}/{iterations}] "
                  f"Cost: {current_cost:.6e} "
                  f"GradNorm: {grad_info} "
                  f"LR: {current_lr:.2e} "
                  f"Best: {best_cost:.6e}")
        
        # # 12. 早停检查（可选）
        # if len(cost_history) >= 100:
        #     recent_improvement = cost_history[-100] - cost_history[-1]
        #     if recent_improvement < 1e-8:
        #         print(f"Early stopping at iteration {iteration}: minimal improvement")
        #         break

    # --- 4. 恢复最佳轨迹并生成密集采样 ---
    print(f"\nOptimization completed!")
    print(f"Final cost: {best_cost:.6e}")
    print(f"Total iterations: {len(cost_history)}")
    if len(cost_history) > 0:
        print(f"Cost reduction: {cost_history[0]:.6e} → {best_cost:.6e} "
              f"({((cost_history[0] - best_cost) / cost_history[0] * 100):.2f}% improvement)")
        
    # 使用最佳轨迹生成密集采样
    with torch.no_grad():
        # 构建最终的完整控制点
        final_full_trajectory = torch.cat([
            start_point.unsqueeze(0),    # 固定的起点
            best_middle_points,          # 优化后的中间点
            end_point.unsqueeze(0)       # 固定的终点
        ], dim=0)

        final_trajectory_optimizer = TrajectoryOptimizerSE2(
            final_full_trajectory, 
            stability_cost_map, 
            map_info, 
            device=device
        )
        
        # 生成密集轨迹
        Sx, Sy, Syaw, *_ = final_trajectory_optimizer._evaluate_spline_se2(
            final_full_trajectory, 
            final_trajectory_optimizer.t_dense
        )
        
        optimized_trajectory = torch.stack([Sx, Sy], dim=1).cpu().numpy()
        optimized_yaw_dense = Syaw.cpu().numpy()
        final_control_poses = final_full_trajectory.cpu().numpy()

    # --- 4. 可视化结果 ---
    import mpl_toolkits.mplot3d  # 确保 3D 支持

    # 准备基础数据
    elev_map_np = sample['elevation'].cpu().numpy()
    # 计算要展示的 yaw 切片索引（优先根据初始控制点的第一个 yaw，回退到中间切片）
    D = map_size[2]
    try:
        init_yaw = float(initial_points[0, 2])
        rel = (init_yaw - origin[2]) % (2 * np.pi)
        yaw_idx = int(round(rel / (2 * np.pi) * D)) % D
    except Exception:
        yaw_idx = D // 2
    cost_slice_to_show = stability_cost_map[yaw_idx].cpu().numpy()  # (H, W)
    map_extent = [origin[0], origin[0] + map_size[0]*resolution, origin[1], origin[1] + map_size[1]*resolution]

    # 重新计算初始轨迹的 yawDense（用于 3D 可视化）
    with torch.no_grad():
        initial_poses_torch = torch.tensor(initial_points, device=device, dtype=torch.float32)
        Sx_i, Sy_i, Syaw_i, *_ = optimizer._evaluate_spline_se2(initial_poses_torch, optimizer.t_dense)
        initial_trajectory_full = torch.stack([Sx_i, Sy_i, Syaw_i], dim=1).cpu().numpy()  # (K,3)

    # 优化后轨迹（已经有 xy 和 yaw_dense）
    optimized_trajectory_full = np.concatenate([optimized_trajectory, optimized_yaw_dense.reshape(-1,1)], axis=1)  # (K,3)

    # final_control_poses = optimizer._assemble_control_poses().detach().cpu().numpy()

    # --- 修复：final_control_poses 是完整的控制点集合 (N,3)
    if optimizer.variable_indices:
        variable_control_poses = final_control_poses[optimizer.variable_indices]  # shape: (N-2, 3)
    else:
        variable_control_poses = np.zeros((0, 3), dtype=final_control_poses.dtype)

    optimized_positions = np.zeros_like(initial_points)
    optimized_positions[optimizer.fixed_indices] = initial_points[optimizer.fixed_indices]  # 起点和终点
    if len(optimizer.variable_indices) > 0:
        optimized_positions[optimizer.variable_indices] = final_control_poses[optimizer.variable_indices]

    # --- 先单独检测一下原轨迹的控制点，是否存在会倾覆的点 ---
    initial_yaw_dense = initial_trajectory_full[:, 2]  # 初始轨迹的 yaw_dense
    initial_points_tensor = torch.tensor(initial_points, device=device, dtype=torch.float32)
    # # 计算初始轨迹的中间点（去除首尾）
    # x_q = initial_points_tensor[1:-1, 0]
    # y_q = initial_points_tensor[1:-1, 1]
    # yaw_q = initial_points_tensor[1:-1, 2]
    # queries = torch.stack([x_q, y_q, yaw_q], dim=1)
    
    # res = compute_esdf_batch(
    #     nx, ny, nz, queries,
    #     resolution=0.4, origin=(-20.0, -20.0),
    #     yaw_weight=1.4, search_radius=5.0, chunk_cells=3000, device=device
    # )
    # is_unreachables, infos = query_is_unreachable_by_match_batch(res, queries, origin=(-20.0, -20.0), resolution=0.4)
    # # 打印 is_unreachables 的统计信息
    # if isinstance(is_unreachables, torch.Tensor):
    #     is_unreachables_np = is_unreachables.cpu().numpy()
    # else:
    #     is_unreachables_np = np.asarray(is_unreachables)
    # capsize_mask = is_unreachables_np > 0  # 将不可行点置为 True，其余为 False
    # # 安全检查掩码长度是否与中间点数一致，若不一致则尝试对齐或截断
    # mid_points = initial_trajectory[1:-1]
    # if capsize_mask.shape[0] != mid_points.shape[0]:
    #     # 尽量截断或扩展掩码到匹配长度（以 False 填充）
    #     mask = np.zeros(mid_points.shape[0], dtype=bool)
    #     L = min(mask.shape[0], capsize_mask.shape[0])
    #     mask[:L] = capsize_mask[:L]
    #     capsize_mask = mask
    # capsize_points = mid_points[capsize_mask]
    # num_capsize = np.sum(capsize_mask)
    # print(f"Number of points which is unreachable in <initial_control_poses>: {num_capsize} out of {len(is_unreachables)}, percentage: {num_capsize / len(is_unreachables) * 100:.2f}%")
    
    # 计算初始轨迹的中间点（去除首尾）- 使用控制点
    initial_control_mid_points = initial_points[1:-1, :2]  # (N-2, 2)
    initial_control_mid_yaws = initial_points[1:-1, 2]     # (N-2,)

    from dataLoader_uneven import compute_map_yaw_bins
    yaw_stability = compute_map_yaw_bins(nx, ny, nz, yaw_bins=36)  # [H, W, 36]

    def check_trajectory_reachability_consistent(trajectory_points, yaw_values, yaw_stability):
        """使用与 data_clean.py 完全一致的方法检查轨迹点的可达性"""
        capsize_mask = []
        
        for i in range(len(trajectory_points)):
            x, y = trajectory_points[i]
            yaw = yaw_values[i]
            
            # 与 data_clean.py 完全一致的坐标转换
            x_idx = int((x + 20) / 0.4)
            y_idx = int((y + 20) / 0.4)
            yaw_idx = int((yaw + np.pi) / (2 * np.pi / 36)) % 36
            
            # 边界检查和稳定性判断
            if 0 <= x_idx < yaw_stability.shape[0] and 0 <= y_idx < yaw_stability.shape[1]:
                yaw_stability_value = yaw_stability[x_idx, y_idx, yaw_idx]
                if torch.is_tensor(yaw_stability_value):
                    is_unreachable = bool((yaw_stability_value == 0).cpu().numpy())
                else:
                    is_unreachable = bool(yaw_stability_value == 0)
            else:
                is_unreachable = True  # 超出地图边界
                
            capsize_mask.append(is_unreachable)
        
        capsize_mask = np.array(capsize_mask, dtype=bool)
        capsize_points = trajectory_points[capsize_mask]
        
        return capsize_mask, capsize_points

    # 检查初始控制点（中间点）的可达性
    capsize_mask, capsize_points = check_trajectory_reachability_consistent(
        initial_control_mid_points, initial_control_mid_yaws, yaw_stability
    )
    num_capsize = np.sum(capsize_mask)
    print(f"Number of points which is unreachable in <initial_control_poses>: {num_capsize} out of {len(capsize_mask)}, percentage: {num_capsize / len(capsize_mask) * 100:.2f}%")
        
    # --- Figure 1: 两个子图（分别显示初始轨迹 / 优化后轨迹，各自叠在彩色高程上） ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 8))

    # 优化前轨迹
    visualize_terrain_trajectory(
        axes1[0], 
        initial_trajectory, 
        nz,
        nx, 
        ny, 
        nz,
        positions=initial_points[:, :2],  # 只传入 x, y
        yaws=initial_points[:, 2]  # 使用初始控制点的 yaw
    )
    axes1[0].set_title('Initial Trajectory')

    # 检查初始密集轨迹的可达性（去除端点）
    initial_traj_mid_points = initial_trajectory[1:-1]     # (K-2, 2)
    initial_traj_mid_yaws = initial_trajectory_full[1:-1, 2]       # (K-2,)

    capsize_mask, capsize_points = check_trajectory_reachability_consistent(
        initial_traj_mid_points, initial_traj_mid_yaws, yaw_stability
    )
    axes1[0].scatter(capsize_points[:, 0], capsize_points[:, 1], c='green', s=20, label='unreachable points')
    axes1[0].legend() # 添加图例

    # 添加对于可达性的统计信息
    num_capsize = np.sum(capsize_mask)
    print(f"Number of points which is unreachable in  <initial_trajectory> : {num_capsize} out of {len(capsize_mask)}, percentage: {num_capsize / len(capsize_mask) * 100:.2f}%")

    # 可视化优化后轨迹
    visualize_terrain_trajectory(
        axes1[1], 
        optimized_trajectory,
        nz,
        nx,
        ny,
        nz,
        positions=optimized_positions[:, :2],  # 只传入 x, y
        yaws=optimized_positions[:, 2]  # 使用优化后的 yaw
    )
    axes1[1].set_title('Optimized Trajectory')

    # 检查优化后密集轨迹的可达性（去除端点）
    optimized_traj_mid_points = optimized_trajectory[1:-1]     # (K-2, 2)
    optimized_traj_mid_yaws = optimized_yaw_dense[1:-1]       # (K-2,)

    capsize_mask, capsize_points = check_trajectory_reachability_consistent(
        optimized_traj_mid_points, optimized_traj_mid_yaws, yaw_stability
    )
    axes1[1].scatter(capsize_points[:, 0], capsize_points[:, 1], c='green', s=20, label='unreachable points')
    axes1[1].legend()
    
    # 添加对于可达性的统计信息
    num_capsize = np.sum(capsize_mask)
    print(f"Number of points which is unreachable in <optimized_trajectory>: {num_capsize} out of {len(capsize_mask)}, percentage: {num_capsize / len(capsize_mask) * 100:.2f}%")

    # --- Figure 2: 两个 3D 子图（流形空间：x, y, yaw）---
    # 构建占据体小量采样点以供 3D 可视化（下采样以减少点数）
    D, H, W = stability_cost_map.shape
    st_map_np = stability_cost_map.cpu().numpy()  # (D, H, W)
    xs = origin[0] + (np.arange(W) + 0.5) * resolution
    ys = origin[1] + (np.arange(H) + 0.5) * resolution
    yaws = origin[2] + (np.arange(D) * (2*np.pi / D))
    # 下采样网格
    sx = np.linspace(0, W-1, min(40, W)).astype(int)
    sy = np.linspace(0, H-1, min(40, H)).astype(int)
    sd = np.linspace(0, D-1, min(40, D)).astype(int)
    Xg, Yg, Zg = np.meshgrid(xs[sx], ys[sy], yaws[sd], indexing='xy')
    vals = st_map_np[np.ix_(sd, sy, sx)]  # (Sd, Sy, Sx)
    vals_flat = vals.reshape(-1)
    coords_flat = np.stack([Xg.reshape(-1), Yg.reshape(-1), Zg.reshape(-1)], axis=1)
    # 只绘制高代价点，用颜色映射显示 cost
    mask_vis = vals_flat > np.percentile(vals_flat, 60)
    vis_coords = coords_flat[mask_vis]
    vis_vals = vals_flat[mask_vis]

    fig2 = plt.figure(figsize=(16, 8))
    ax3d_1 = fig2.add_subplot(1, 2, 1, projection='3d')
    ax3d_2 = fig2.add_subplot(1, 2, 2, projection='3d')

    ax3d_1.plot(initial_trajectory_full[:,0], initial_trajectory_full[:,1], initial_trajectory_full[:,2], color='cyan', linestyle='--', linewidth=2, label='Initial Traj (SE2)')
    ax3d_1.scatter(initial_points[:,0], initial_points[:,1], initial_points[:,2], c='blue', s=40, marker='o', label='Initial Controls', zorder=6)
    ax3d_1.set_title("Initial Trajectory in SE(2) Manifold")
    ax3d_1.set_xlabel("X (m)"); ax3d_1.set_ylabel("Y (m)"); ax3d_1.set_zlabel("Yaw (rad)")
    ax3d_1.legend(); ax3d_1.grid(True, linestyle='--', alpha=0.3)

    ax3d_2.plot(optimized_trajectory_full[:,0], optimized_trajectory_full[:,1], optimized_trajectory_full[:,2], color='magenta', linestyle='-', linewidth=2, label='Optimized Traj (SE2)')
    ax3d_2.scatter(final_control_poses[:,0], final_control_poses[:,1], final_control_poses[:,2], c='red', s=40, marker='x', label='Final Controls', zorder=6)
    ax3d_2.set_title("Optimized Trajectory in SE(2) Manifold")
    ax3d_2.set_xlabel("X (m)"); ax3d_2.set_ylabel("Y (m)"); ax3d_2.set_zlabel("Yaw (rad)")
    ax3d_2.legend(); ax3d_2.grid(True, linestyle='--', alpha=0.3)

    # --- Figure 3: 两个子图（分别显示优化后的轨迹和通过优化后的控制点重新采样得到的轨迹），绘制在彩色高程图上 ---
    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 6))
    # 左：优化后的轨迹
    visualize_terrain_trajectory(
        ax3[0], 
        optimized_trajectory,
        nz,
        nx, 
        ny, 
        nz,
        positions=optimized_positions[:, :2],  # 只传入 x, y
        yaws=optimized_positions[:, 2]  # 使用优化后的 yaw
    )
    ax3[0].set_title('Optimized Trajectory')
    
    # 绘制优化后轨迹的不可达点（复用图1中的结果）
    optimized_traj_mid_points_fig3 = optimized_trajectory[1:-1]     # (K-2, 2)
    optimized_traj_mid_yaws_fig3 = optimized_yaw_dense[1:-1]       # (K-2,)

    capsize_mask, capsize_points = check_trajectory_reachability_consistent(
        optimized_traj_mid_points_fig3, optimized_traj_mid_yaws_fig3, yaw_stability
    )
    ax3[0].scatter(capsize_points[:, 0], capsize_points[:, 1], c='green', s=20, label='unreachable points')
    ax3[0].legend()  # 添加图例
    
    # 右：经过优化后的控制点重新采样得到的轨迹
    # 重新采样轨迹（使用优化后的控制点）
    resampled_control_poses = torch.tensor(optimized_positions, device=device, dtype=torch.float32)
    Sx_i, Sy_i, Syaw_i, *_ = optimizer._evaluate_spline_se2(resampled_control_poses, optimizer.t_dense)
    resampled_trajectory = torch.stack([Sx_i, Sy_i], dim=1).cpu().numpy()
    resampled_trajectory_full = torch.stack([Sx_i, Sy_i, Syaw_i], dim=1).cpu().numpy()  # (K,3)
    resampled_control_poses = resampled_control_poses.cpu().numpy()  # 转为 numpy

    visualize_terrain_trajectory(
        ax3[1], 
        resampled_trajectory,
        nz,
        nx,
        ny,
        nz,
        positions=resampled_control_poses[:, :2],  # 只传入 x, y
        yaws=resampled_control_poses[:, 2]  # 使用优化后的 yaw
    )
    ax3[1].set_title('Resampled Trajectory from Optimized Controls')
    
    # 统计 resampled 轨迹各点的可达性
    resampled_traj_mid_points = resampled_trajectory[1:-1]     # (K-2, 2)
    resampled_traj_mid_yaws = resampled_trajectory_full[1:-1, 2]       # (K-2,)

    capsize_mask, capsize_points = check_trajectory_reachability_consistent(
        resampled_traj_mid_points, resampled_traj_mid_yaws, yaw_stability
    )
    ax3[1].scatter(capsize_points[:, 0], capsize_points[:, 1], c='green', s=20, label='unreachable points')
    ax3[1].legend()  # 添加图例
    # 打印可达性统计信息
    num_capsize = np.sum(capsize_mask)
    print(f"Number of points which is unreachable in <resampled_trajectory>: {num_capsize} out of {len(capsize_mask)}, percentage: {num_capsize / len(capsize_mask) * 100:.2f}%")

    # --- Figure 4: 损失曲线 ---
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 4))
    ax4.plot(cost_history, marker='o', linewidth=1)
    ax4.set_title("Cost Function Convergence")
    ax4.set_xlabel("Iteration"); ax4.set_ylabel("Cost")
    ax4.grid(True); ax4.set_yscale('log')

    plt.tight_layout()
    
    # savefig
    fig1.savefig(f'figure_1_trajectory_comparison_{path_index}.png', dpi=300)
    fig2.savefig(f'figure_2_3d_manifold_{path_index}.png', dpi=300)
    fig3.savefig(f'figure_3_resampled_vs_optimized_{path_index}.png', dpi=300)
    fig4.savefig(f'figure_4_cost_convergence_{path_index}.png', dpi=300)
    
    plt.show()
