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
        # 把 (x,y,yaw) 当作 3D 空间坐标来参数化 t（不做角度 unwrap）
        # 注意：这里我们保留 rot_scale 作为可选项，但默认将其设为 0.0，让 t 只由平移 (x,y) 决定，避免 yaw 导致参数化不一致而使样条看起来不平滑。
        rot_scale = 0.0  # 将 yaw 差值映射为“等效距离”的缩放；设为 0 则 yaw 不影响 t
        xy = self.initial_poses[:, :2].to(self.device)
        yaw = self.initial_poses[:, 2].to(self.device)
        diffs_xy = xy[1:] - xy[:-1]
        trans_dists = torch.norm(diffs_xy, dim=1)  # (N-1,)

        # **注意**：此处如需把 yaw 也参与参数化，可将 rot_scale 设置为非零；
        # 但为了避免 yaw 的跳变（或 wrap）引起 t 不连续，默认不参与。
        yaw_diffs = (yaw[1:] - yaw[:-1]).abs()  # 直接绝对差，保留但默认不影响（rot_scale=0）

        total_dists = torch.sqrt(trans_dists**2 + (rot_scale * yaw_diffs)**2) + 1e-8
        t = torch.cat([torch.tensor([0.0], device=self.device), torch.cumsum(total_dists, dim=0)])
        t = (t / (t[-1] if t[-1] > 0 else 1.0)).cpu().numpy()
        self.t_points = torch.tensor(t, device=self.device, dtype=torch.float32).detach()

        # --- 4. 预计算参考轨迹 (用于 follow_cost) ---
        self.K_cost = 200
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

    def _evaluate_spline_se2(self, control_poses, t_eval):
        """对 SE(2) 控制点 (x, y, yaw) 进行样条插值，**把 yaw 视为普通标量**（不做周期/unwrap 处理）。
        关键修正：在插值前根据当前 control_poses 重新计算 t_ctrl，使参数化与控制点一致，避免因 t 固定而在控制点改变时造成不平滑。"""
        # control_poses: (N, 3)
        x_ctrl, y_ctrl, yaw_ctrl = control_poses.T

        # 根据当前 control_poses 重新计算 t_ctrl（默认仍使用 rot_scale=0，若需将 yaw 计入可手动调整）
        rot_scale = 0.0
        xy = control_poses[:, :2]
        diffs_xy = xy[1:] - xy[:-1]
        trans_dists = torch.norm(diffs_xy, dim=1)
        yaw_diffs = (yaw_ctrl[1:] - yaw_ctrl[:-1]).abs()
        total_dists = torch.sqrt(trans_dists**2 + (rot_scale * yaw_diffs)**2) + 1e-8
        t_local = torch.cat([torch.tensor([0.0], device=self.device), torch.cumsum(total_dists, dim=0)])
        t_local = (t_local / (t_local[-1] if t_local[-1] > 0 else 1.0)).to(device=self.device, dtype=torch.float32)

        # 对 x, y 直接插值（传入局部 t_ctrl）
        Sx, Sx_dot, Sx_ddot = self._evaluate_scalar_spline(x_ctrl, t_eval, t_ctrl=t_local)
        Sy, Sy_dot, Sy_ddot = self._evaluate_scalar_spline(y_ctrl, t_eval, t_ctrl=t_local)

        # 对 yaw：直接作为普通标量插值（无 unwrap），也使用局部 t_ctrl
        Syaw, Syaw_dot, Syaw_ddot = self._evaluate_scalar_spline(yaw_ctrl, t_eval, t_ctrl=t_local)

        return Sx, Sy, Syaw, Sx_dot, Sy_dot, Syaw_dot, Sx_ddot, Sy_ddot, Syaw_ddot

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
        
        # a) 占据/碰撞成本
        dense_traj_world = torch.stack([Sx, Sy, Syaw], dim=1)
        grid_coords = self.world_to_grid_normalized(dense_traj_world)
        occupancy_values = F.grid_sample(self.occupancy_map, grid_coords, mode='bilinear', padding_mode='border', align_corners=True)
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
        smoothness_cost = torch.mean(Sx_ddot**2 + Sy_ddot**2 + Syaw_ddot**2)

        # c) 几何曲率 + 低速门控（弱化低速曲率惩罚，以允许倒车）
        eps = 1e-9
        speed = torch.sqrt(Sx_dot**2 + Sy_dot**2 + eps)
        geom_curvature = torch.abs(Sx_dot * Sy_ddot - Sy_dot * Sx_ddot) / (speed**3 + 1e-9)
        geom_curvature = torch.abs(Sx_dot * Sy_ddot - Sy_dot * Sx_ddot) / (speed**3 + 1e-9)

        # 低速下逐渐弱化曲率惩罚
        v_thresh = 0.08
        gate_k = 80.0
        gate = torch.sigmoid((speed - v_thresh) * gate_k)
        # gate = torch.tensor(1.0, device=self.device)  # 直接使用 gate=1.0，表示不做门控（允许低速曲率）
        curvature_limit = 2.1
        curvature_violation = torch.relu(geom_curvature - curvature_limit)
        curvature_cost = torch.mean(gate * curvature_violation)

        # d) 禁止原地大角度转动（yaw per meter）
        # 衡量每米的朝向变化量：yaw_per_meter = |yaw_rate| / (speed + v_eps)
        v_eps = 1e-3
        yaw_per_meter = torch.abs(Syaw_dot) / (speed + v_eps)
        # 允许一定的角度/米（例如 2 rad/m），超过则惩罚；使用平滑平方惩罚
        yaw_per_meter_limit = 2.0
        yaw_per_meter_violation = torch.relu(yaw_per_meter - yaw_per_meter_limit)
        yaw_per_meter_cost = torch.mean(yaw_per_meter_violation**2)
        
        # e) 控制量最小化成本 (惩罚过大的速度)
        w_angular_vel = 0.1
        control_cost = torch.mean(Sx_dot**2 + Sy_dot**2) + w_angular_vel * torch.mean(Syaw_dot**2)
        
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
        tangent_angle = torch.atan2(y_dot, x_dot)
        angle_diff = tangent_angle - Syaw
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))  # 规范到 [-pi, pi]
        angle_diff_cost = torch.mean(torch.sin(angle_diff)**2)
        
        # g) 端点 yaw 对齐成本
        start_yaw_opt = ctrl_poses[0, 2]
        end_yaw_opt = ctrl_poses[-1, 2]
        yaw0 = self.start_pose[2]
        yawn = self.end_pose[2]
        yaw_diff_start = torch.atan2(torch.sin(start_yaw_opt - yaw0), torch.cos(start_yaw_opt - yaw0))
        yaw_diff_end = torch.atan2(torch.sin(end_yaw_opt - yawn), torch.cos(end_yaw_opt - yawn))
        yaw_endpoint_cost = torch.mean(torch.abs(yaw_diff_start)) + torch.mean(torch.abs(yaw_diff_end))
        
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
            ctrl_smoothness_cost = torch.mean(ctrl_x_ddot**2 + ctrl_y_ddot**2 \
                                    # + ctrl_yaw_ddot**2 \
                                    )
            
        # i) 超出地图范围的惩罚（修正：判断是否在 [-map_limit, map_limit] 外）
        map_limit = 5.0
        overflow_x = torch.relu(torch.abs(Sx) - map_limit)
        overflow_y = torch.relu(torch.abs(Sy) - map_limit)
        out_of_bounds_cost = torch.mean(overflow_x + overflow_y)
        
        

        # --- 组合成本 ---
        weights = {
            'obstacle': 1e1,
            'smoothness': 1e-4,
            'curvature': 5e2,
            'yaw_per_meter': 0e2, # 惩罚原地大角度转动的权重
            'control': 1e-2,
            'angle_diff': 5e2, # 惩罚角度与切线方向不一致的权重
            'endpoints': 1e4,    # 强力惩罚端点 yaw 对齐
            'control_smoothness': 0e3,  # 控制点连线的光滑性损失
            'out_of_bounds': 1e8,  # 超出地图范围的惩罚
        }
         
        total_cost = (
            weights['obstacle'] * obstacle_cost +
            weights['smoothness'] * smoothness_cost +
            weights['curvature'] * curvature_cost +
            weights['yaw_per_meter'] * yaw_per_meter_cost +
            weights['control'] * control_cost +
            weights['angle_diff'] * angle_diff_cost +
            weights['endpoints'] * yaw_endpoint_cost +
            weights['control_smoothness'] * ctrl_smoothness_cost if self.variable_poses is not None else 0.0 +
            weights['out_of_bounds'] * out_of_bounds_cost
        )
         
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
        occupancy_values = F.grid_sample(self.occupancy_map, grid_coords, mode='bilinear', padding_mode='border', align_corners=True)
        obstacle_cost = torch.mean(occupancy_values)

        # smoothness
        smoothness_cost = torch.mean(Sx_ddot**2 + Sy_ddot**2 + Syaw_ddot**2)

        # curvature with low-speed gate
        eps = 1e-9
        speed = torch.sqrt(Sx_dot**2 + Sy_dot**2 + eps)
        geom_curvature = torch.abs(Sx_dot * Sy_ddot - Sy_dot * Sx_ddot) / (speed**3 + 1e-9)
        v_thresh = 0.08
        gate_k = 80.0
        gate = torch.sigmoid((speed - v_thresh) * gate_k)
        curvature_limit = 2.1
        curvature_violation = torch.relu(geom_curvature - curvature_limit)
        curvature_cost = torch.mean(gate * curvature_violation)

        # yaw per meter
        v_eps = 1e-3
        yaw_per_meter = torch.abs(Syaw_dot) / (speed + v_eps)
        yaw_per_meter_limit = 2.0
        yaw_per_meter_violation = torch.relu(yaw_per_meter - yaw_per_meter_limit)
        yaw_per_meter_cost = torch.mean(yaw_per_meter_violation**2)

        # control cost
        w_angular_vel = 0.1
        control_cost = torch.mean(Sx_dot**2 + Sy_dot**2) + w_angular_vel * torch.mean(Syaw_dot**2)

        # angle diff cost (sin^2 保持允许倒车)
        tangent_angle = torch.atan2(Sy_dot, Sx_dot)
        angle_diff = tangent_angle - Syaw
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        # angle_diff_cost = torch.mean(torch.sin(angle_diff)**2)
        angle_diff_cost = torch.mean(1.0 - torch.cos(angle_diff))

        # endpoints yaw cost (使用 ctrl 的首尾)
        start_yaw_opt = ctrl[0, 2]
        end_yaw_opt = ctrl[-1, 2]
        yaw0 = self.start_pose[2]
        yawn = self.end_pose[2]
        yaw_diff_start = torch.atan2(torch.sin(start_yaw_opt - yaw0), torch.cos(start_yaw_opt - yaw0))
        yaw_diff_end = torch.atan2(torch.sin(end_yaw_opt - yawn), torch.cos(end_yaw_opt - yawn))
        yaw_endpoint_cost = torch.mean(torch.abs(yaw_diff_start)) + torch.mean(torch.abs(yaw_diff_end))

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

        # out of bounds
        map_limit = 5.0
        overflow_x = torch.relu(torch.abs(Sx) - map_limit)
        overflow_y = torch.relu(torch.abs(Sy) - map_limit)
        out_of_bounds_cost = torch.mean(overflow_x + overflow_y)

        weights = {
            'obstacle': 1e1,
            'smoothness': 1e-4,
            'curvature': 5e2,
            'yaw_per_meter': 1e2,
            'control': 1e-2,
            'angle_diff': 4e4,
            'endpoints': 1e4,
            'control_smoothness': 0e3,
            'out_of_bounds': 1e8,
        }

        total_cost = (
            weights['obstacle'] * obstacle_cost +
            weights['smoothness'] * smoothness_cost +
            weights['curvature'] * curvature_cost +
            weights['yaw_per_meter'] * yaw_per_meter_cost +
            weights['control'] * control_cost +
            weights['angle_diff'] * angle_diff_cost +
            weights['endpoints'] * yaw_endpoint_cost +
            weights['control_smoothness'] * ctrl_smoothness_cost +
            weights['out_of_bounds'] * out_of_bounds_cost
        )
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
    im = ax.imshow(height_map, cmap='terrain', extent=[-5, 5, -5, 5], origin='lower')
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


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 0. 加载地形数据 ---
    from dataLoader_uneven import UnevenPathDataLoader
    env_list = ['env000001']
    dataFolder = '/home/yrf/MPT/data/terrain_dataset/train'
    dataset = UnevenPathDataLoader(env_list, dataFolder)
    path_index = 41
    sample = dataset[path_index]
    if sample is None:
        raise ValueError(f"Sample at index {path_index} is invalid.")
    
    nx = sample['map'][0, :, :].to(device)
    ny = sample['map'][1, :, :].to(device)
    nz = sample['map'][2, :, :].to(device)

    # --- 1. 定义代价地图参数并生成地图 ---
    map_size = (100, 100, 36) # W, H, D for (x, y, yaw)
    resolution = 0.1
    origin = (-5.0, -5.0, -np.pi) # x, y, yaw
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
    # 注意：这里不对 yaw 做周期化或 unwrap，保持原样

    # --- 3. 初始化并运行优化器 (使用 TrajectoryOptimizerSE2) ---
    optimizer = TrajectoryOptimizerSE2(initial_points, stability_cost_map, map_info, device=device)
    
    # 获取优化前的轨迹
    with torch.no_grad():
        initial_poses_torch = torch.tensor(initial_points, device=device, dtype=torch.float32)
        Sx_i, Sy_i, *_ = optimizer._evaluate_spline_se2(initial_poses_torch, optimizer.t_dense)
        initial_trajectory = torch.stack([Sx_i, Sy_i], dim=1).cpu().numpy()

    # 执行优化
    optimized_trajectory, optimized_yaw_dense, cost_history = optimizer.optimize(iterations=400, lr=0.05, verbose=True)


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

    final_control_poses = optimizer._assemble_control_poses().detach().cpu().numpy()

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
    # 计算初始轨迹的中间点（去除首尾）
    x_q = initial_points_tensor[1:-1, 0]
    y_q = initial_points_tensor[1:-1, 1]
    yaw_q = initial_points_tensor[1:-1, 2]
    queries = torch.stack([x_q, y_q, yaw_q], dim=1)
    
    res = compute_esdf_batch(
        nx, ny, nz, queries,
        resolution=0.1, origin=(-5.0, -5.0),
        yaw_weight=1.4, search_radius=5.0, chunk_cells=3000, device=device
    )
    is_unreachables, infos = query_is_unreachable_by_match_batch(res, queries, origin=(-5.0, -5.0), resolution=0.1)
    # 打印 is_unreachables 的统计信息
    if isinstance(is_unreachables, torch.Tensor):
        is_unreachables_np = is_unreachables.cpu().numpy()
    else:
        is_unreachables_np = np.asarray(is_unreachables)
    capsize_mask = is_unreachables_np > 0  # 将不可行点置为 True，其余为 False
    # 安全检查掩码长度是否与中间点数一致，若不一致则尝试对齐或截断
    mid_points = initial_trajectory[1:-1]
    if capsize_mask.shape[0] != mid_points.shape[0]:
        # 尽量截断或扩展掩码到匹配长度（以 False 填充）
        mask = np.zeros(mid_points.shape[0], dtype=bool)
        L = min(mask.shape[0], capsize_mask.shape[0])
        mask[:L] = capsize_mask[:L]
        capsize_mask = mask
    capsize_points = mid_points[capsize_mask]
    num_capsize = np.sum(capsize_mask)
    print(f"Number of points which is unreachable in <initial_control_poses>: {num_capsize} out of {len(is_unreachables)}, percentage: {num_capsize / len(is_unreachables) * 100:.2f}%")

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

    # 去除端点以匹配后续对中间点的处理
    x_q = Sx_i[1:-1]
    y_q = Sy_i[1:-1]
    yaw_q = Syaw_i[1:-1]

    queries = torch.stack([x_q, y_q, yaw_q], dim=1)
    
    res = compute_esdf_batch(
        nx, ny, nz, queries,
        resolution=0.1, origin=(-5.0, -5.0),
        yaw_weight=1.4, search_radius=5.0, chunk_cells=3000, device=device
    )
    is_unreachables, infos = query_is_unreachable_by_match_batch(res, queries, origin=(-5.0, -5.0), resolution=0.1)
    
    # 绘制出 is_unreachables 对应的xy点
    if isinstance(is_unreachables, torch.Tensor):
        is_unreachables_np = is_unreachables.cpu().numpy()
    else:
        is_unreachables_np = np.asarray(is_unreachables)
    capsize_mask = is_unreachables_np > 0  # 将不可行点置为 True，其余为 False
    # 安全检查掩码长度是否与中间点数一致，若不一致则尝试对齐或截断
    mid_points = initial_trajectory[1:-1]
    if capsize_mask.shape[0] != mid_points.shape[0]:
        # 尽量截断或扩展掩码到匹配长度（以 False 填充）
        mask = np.zeros(mid_points.shape[0], dtype=bool)
        L = min(mask.shape[0], capsize_mask.shape[0])
        mask[:L] = capsize_mask[:L]
        capsize_mask = mask
    capsize_points = mid_points[capsize_mask]
    axes1[0].scatter(capsize_points[:, 0], capsize_points[:, 1], c='green', s=20, label='unreachable points')
    axes1[0].legend() # 添加图例

    # 添加对于 is_unreachables 的统计信息
    num_capsize = np.sum(capsize_mask)
    print(f"Number of points which is unreachable in  <initial_trajectory> : {num_capsize} out of {len(is_unreachables)}, percentage: {num_capsize / len(is_unreachables) * 100:.2f}%")

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

    # 使用 optimizer 返回的密集 yaw 配合轨迹做不可达判断
    x_q = torch.tensor(optimized_trajectory[:, 0], device=device, dtype=torch.float32)
    y_q = torch.tensor(optimized_trajectory[:, 1], device=device, dtype=torch.float32)
    yaw_q = torch.tensor(optimized_yaw_dense, device=device, dtype=torch.float32)
    # 剪裁端点以匹配之前的 "中间点" 处理（如果需要）
    x_q_mid = x_q[1:-1]
    y_q_mid = y_q[1:-1]
    yaw_q_mid = yaw_q[1:-1]
    queries = torch.stack([x_q_mid, y_q_mid, yaw_q_mid], dim=1)

    res = compute_esdf_batch(
        nx, ny, nz, queries,
        resolution=0.1, origin=(-5.0, -5.0),
        yaw_weight=1.4, search_radius=5.0, chunk_cells=3000, device=device
    )
    is_unreachables, infos = query_is_unreachable_by_match_batch(res, queries, origin=(-5.0, -5.0), resolution=0.1)
    
    # 绘制出 is_unreachables 对应的xy点
    if isinstance(is_unreachables, torch.Tensor):
        is_unreachables_np = is_unreachables.cpu().numpy()
    else:
        is_unreachables_np = np.asarray(is_unreachables)
    capsize_mask = is_unreachables_np > 0
    mid_points = optimized_trajectory[1:-1]
    if capsize_mask.shape[0] != mid_points.shape[0]:
        mask = np.zeros(mid_points.shape[0], dtype=bool)
        L = min(mask.shape[0], capsize_mask.shape[0])
        mask[:L] = capsize_mask[:L]
        capsize_mask = mask
    capsize_points = mid_points[capsize_mask]
    axes1[1].scatter(capsize_points[:, 0], capsize_points[:, 1], c='green', s=20, label='unreachable points')
    axes1[1].legend()
    
    # 添加对于 is_unreachables 的统计信息
    num_capsize = np.sum(capsize_mask)
    print(f"Number of points which is unreachable in <optimized_trajectory>: {num_capsize} out of {len(is_unreachables)}, percentage: {num_capsize / len(is_unreachables) * 100:.2f}%")

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
    x_q = torch.tensor(optimized_trajectory[:, 0], device=device, dtype=torch.float32)
    y_q = torch.tensor(optimized_trajectory[:, 1], device=device, dtype=torch.float32)
    yaw_q = torch.tensor(optimized_yaw_dense, device=device, dtype=torch.float32)
    # 剪裁端点以匹配之前的 "中间点"
    x_q_mid = x_q[1:-1]
    y_q_mid = y_q[1:-1]
    yaw_q_mid = yaw_q[1:-1]
    queries = torch.stack([x_q_mid, y_q_mid, yaw_q_mid], dim=1)
    res = compute_esdf_batch(
        nx, ny, nz, queries,
        resolution=0.1, origin=(-5.0, -5.0),
        yaw_weight=1.4, search_radius=5.0, chunk_cells=3000, device=device
    )
    is_unreachables, infos = query_is_unreachable_by_match_batch(res, queries, origin=(-5.0, -5.0), resolution=0.1)
    # 绘制出 is_unreachables 对应的xy点
    if isinstance(is_unreachables, torch.Tensor):
        is_unreachables_np = is_unreachables.cpu().numpy()
    else:
        is_unreachables_np = np.asarray(is_unreachables)
    capsize_mask = is_unreachables_np > 0  # 将不可行点置为 True，其余为 False
    # 安全检查掩码长度是否与中间点数一致，若不一致则尝试对齐或截断
    mid_points = optimized_trajectory[1:-1]
    if capsize_mask.shape[0] != mid_points.shape[0]:
        # 尽量截断或扩展掩码到匹配长度（以 False 填充）
        mask = np.zeros(mid_points.shape[0], dtype=bool)
        L = min(mask.shape[0], capsize_mask.shape[0])
        mask[:L] = capsize_mask[:L]
        capsize_mask = mask
    capsize_points = mid_points[capsize_mask]
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
    x_q = torch.tensor(resampled_trajectory[:, 0], device=device, dtype=torch.float32)
    y_q = torch.tensor(resampled_trajectory[:, 1], device=device, dtype=torch.float32)
    yaw_q = torch.tensor(resampled_trajectory_full[:, 2], device=device, dtype=torch.float32)
    # 剪裁端点以匹配之前的 "中间点" 处理（如果需要）
    x_q_mid = x_q[1:-1]
    y_q_mid = y_q[1:-1]
    yaw_q_mid = yaw_q[1:-1]
    queries = torch.stack([x_q_mid, y_q_mid, yaw_q_mid], dim=1)
    res = compute_esdf_batch(
        nx, ny, nz, queries,
        resolution=0.1, origin=(-5.0, -5.0),
        yaw_weight=1.4, search_radius=5.0, chunk_cells=3000, device=device
    )
    is_unreachables, infos = query_is_unreachable_by_match_batch(res, queries, origin=(-5.0, -5.0), resolution=0.1)
    # 绘制出 is_unreachables 对应的xy点
    if isinstance(is_unreachables, torch.Tensor):
        is_unreachables_np = is_unreachables.cpu().numpy()
    else:
        is_unreachables_np = np.asarray(is_unreachables)
    capsize_mask = is_unreachables_np > 0  # 将不可行点置为 True，其余为 False
    # 安全检查掩码长度是否与中间点数一致，若不一致则尝试对齐或截断
    mid_points = resampled_trajectory[1:-1]
    if capsize_mask.shape[0] != mid_points.shape[0]:
        # 尽量截断或扩展掩码到匹配长度（以 False 填充）
        mask = np.zeros(mid_points.shape[0], dtype=bool)
        L = min(mask.shape[0], capsize_mask.shape[0])
        mask[:L] = capsize_mask[:L]
        capsize_mask = mask
    capsize_points = mid_points[capsize_mask]
    ax3[1].scatter(capsize_points[:, 0], capsize_points[:, 1], c='green', s=20, label='unreachable points')
    ax3[1].legend()  # 添加图例
    # 打印 is_unreachables 的统计信息
    num_capsize = np.sum(capsize_mask)
    print(f"Number of points which is unreachable in <resampled_trajectory>: {num_capsize} out of {len(is_unreachables)}, percentage: {num_capsize / len(is_unreachables) * 100:.2f}%")

    # --- Figure 4: 损失曲线 ---
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 4))
    ax4.plot(cost_history, marker='o', linewidth=1)
    ax4.set_title("Cost Function Convergence")
    ax4.set_xlabel("Iteration"); ax4.set_ylabel("Cost")
    ax4.grid(True); ax4.set_yscale('log')

    plt.tight_layout()
    plt.show()
