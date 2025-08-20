import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from scipy.interpolate import make_interp_spline

class TrajectoryOptimizer:
    def __init__(self, points, nx, ny, nz, device='cuda'):
        """
        points: (N,3) numpy or tensor -> x,y,yaw
        nx,ny,nz: terrain normal channels (torch or numpy). Not used in detail here.
        """
        self.device = torch.device(device)

        # 接受 numpy 或 torch.Tensor 作为输入；保留一个 numpy 副本用于可视化兼容性
        if torch.is_tensor(points):
            pts_t = points.to(device=self.device, dtype=torch.float32)
        else:
            pts_t = torch.tensor(np.array(points), device=self.device, dtype=torch.float32)

        # 确保初始化参考量不保留来自外部 points 的计算图（避免多余梯度）
        pts_t = pts_t.detach()
        self.N = pts_t.shape[0]
        # 用于计算/反向传播的 tensor 版本
        self.initial_positions_tensor = pts_t[:, :2]        # (N,2) tensor on device
        self.yaws_tensor = pts_t[:, 2]                      # (N,) tensor on device
        # 兼容旧代码（可视化等）保留 numpy 副本
        self.initial_positions = self.initial_positions_tensor.detach().cpu().numpy()
        self.yaws = self.yaws_tensor.detach().cpu().numpy()

        # fixed endpoints indices
        self.fixed_indices = [0, self.N - 1]
        self.variable_indices = [i for i in range(1, self.N - 1)]

        # 将起点/终点的位姿显式存为不需要梯度的 torch 常量（detach）
        self.start_pos = torch.tensor(self.initial_positions[self.fixed_indices[0]], device=self.device, dtype=torch.float32)
        self.end_pos = torch.tensor(self.initial_positions[self.fixed_indices[1]], device=self.device, dtype=torch.float32)
        self.start_yaw = torch.tensor(self.yaws[self.fixed_indices[0]], device=self.device, dtype=torch.float32)
        self.end_yaw = torch.tensor(self.yaws[self.fixed_indices[1]], device=self.device, dtype=torch.float32)

        # variable positions as a torch parameter
        var_pos = self.initial_positions[self.variable_indices].copy()
        self.variable_positions = torch.nn.Parameter(torch.tensor(var_pos, device=self.device, dtype=torch.float32))

        # variable yaws for internal control points (allow yaw optimization)
        if len(self.variable_indices) > 0:
            var_yaws = self.yaws[self.variable_indices].copy()
            self.variable_yaws = torch.nn.Parameter(torch.tensor(var_yaws, device=self.device, dtype=torch.float32))
        else:
            # no variable yaws (only two control points)
            self.variable_yaws = None

        # store terrain (kept as provided)
        self.nx = nx.to(self.device) if isinstance(nx, torch.Tensor) else torch.tensor(nx, device=self.device)
        self.ny = ny.to(self.device) if isinstance(ny, torch.Tensor) else torch.tensor(ny, device=self.device)
        self.nz = nz.to(self.device) if isinstance(nz, torch.Tensor) else torch.tensor(nz, device=self.device)

        # prepare a fixed parameterization t_points based on initial_positions (chord length)
        dists = np.linalg.norm(np.diff(self.initial_positions, axis=0), axis=1)
        t = np.concatenate([[0.0], np.cumsum(dists)])
        if t[-1] == 0:
            t = np.linspace(0.0, 1.0, self.N)
        else:
            t = t / t[-1]
        # keep as torch tensor on device but detached (treated as constants for indexing)
        self.t_points = torch.tensor(t, device=self.device, dtype=torch.float32)
        
        # -----------------------
        # 预计算：cost 中会用到的参考样条（只计算一次）
        # -----------------------
        # 用于 cost 的采样点数（可调）
        self.K_cost = 200
        self.t_dense = torch.linspace(0.0, 1.0, self.K_cost, device=self.device, dtype=torch.float32)

        # 参考样条、参考yaw、端点参考速度均为常数：使用 no_grad 并 detach 保存，避免构造多余计算图
        control_yaws_ref = torch.cat(
            [self.start_yaw.unsqueeze(0), self.yaws_tensor[1:-1], self.end_yaw.unsqueeze(0)]
        ) if self.N > 2 else torch.cat([self.start_yaw.unsqueeze(0), self.end_yaw.unsqueeze(0)])

        with torch.no_grad():
            orig_pos_torch = self.initial_positions_tensor.detach()
            Sx_ref, Sy_ref, Sx_dot_ref, Sy_dot_ref = self._evaluate_spline(orig_pos_torch, self.t_dense)
            cos_ref = torch.cos(control_yaws_ref.detach())
            sin_ref = torch.sin(control_yaws_ref.detach())
            Scos_ref, _ = self._evaluate_scalar_spline(cos_ref, self.t_dense)
            Ssin_ref, _ = self._evaluate_scalar_spline(sin_ref, self.t_dense)
            yaw_ref_dense = torch.atan2(Ssin_ref, Scos_ref)

            speed_start = torch.sqrt(Sx_dot_ref[0] ** 2 + Sy_dot_ref[0] ** 2) + 1e-6
            speed_end   = torch.sqrt(Sx_dot_ref[-1] ** 2 + Sy_dot_ref[-1] ** 2) + 1e-6
            yaw0 = self.start_yaw
            yawn = self.end_yaw

        # 把参考量保成 detached 常量（无梯度）
        self.Sx_ref = Sx_ref.detach()
        self.Sy_ref = Sy_ref.detach()
        self.Sx_dot_ref = Sx_dot_ref.detach()
        self.Sy_dot_ref = Sy_dot_ref.detach()
        self.yaw_ref_dense = yaw_ref_dense.detach()

        self.d0x = (torch.cos(yaw0) * speed_start).to(device=self.device, dtype=torch.float32).detach()
        self.d0y = (torch.sin(yaw0) * speed_start).to(device=self.device, dtype=torch.float32).detach()
        self.dnx = (torch.cos(yawn) * speed_end).to(device=self.device, dtype=torch.float32).detach()
        self.dny = (torch.sin(yawn) * speed_end).to(device=self.device, dtype=torch.float32).detach()

    # ------------------------------
    # ---- Natural Cubic Spline ----
    # ------------------------------
    def _solve_natural_cubic_M(self, y, d0=None, dn=None):
        """
        Solve for second derivatives M for cubic spline.
        If d0 and dn are None -> natural boundary.
        If d0/dn provided -> clamped boundary with S'(t0)=d0 and S'(tn)=dn.
        y: (N,) torch tensor
        d0, dn: optional scalars or 0-dim tensors
        returns M: (N,) torch tensor
        """
        N = self.N
        t = self.t_points
        h = t[1:] - t[:-1]  # (N-1,)

        A = torch.zeros((N, N), dtype=torch.float32, device=self.device)
        rhs = torch.zeros((N,), dtype=torch.float32, device=self.device)

        if d0 is None and dn is None:
            # natural spline boundary
            A[0, 0] = 1.0
            A[-1, -1] = 1.0
        else:
            # make sure d0/dn are tensors on the same device/dtype
            if not torch.is_tensor(d0):
                d0 = torch.tensor(d0, dtype=torch.float32, device=self.device)
            else:
                d0 = d0.to(device=self.device, dtype=torch.float32)
            if not torch.is_tensor(dn):
                dn = torch.tensor(dn, dtype=torch.float32, device=self.device)
            else:
                dn = dn.to(device=self.device, dtype=torch.float32)

            # clamped boundary rows
            A[0, 0] = 2.0 * h[0]
            A[0, 1] = h[0]
            rhs[0] = 6.0 * ( (y[1] - y[0]) / (h[0] + 1e-12) - d0 )

            A[-1, -2] = h[-1]
            A[-1, -1] = 2.0 * h[-1]
            rhs[-1] = 6.0 * ( dn - (y[-1] - y[-2]) / (h[-1] + 1e-12) )

        # interior rows (vectorized)
        if N > 2:
            idx = torch.arange(1, N-1, device=self.device, dtype=torch.long)  # indices 1..N-2
            A[idx, idx - 1] = h[idx - 1]
            A[idx, idx]     = 2.0 * (h[idx - 1] + h[idx])
            A[idx, idx + 1] = h[idx]
            rhs[idx] = 6.0 * ( (y[idx + 1] - y[idx]) / (h[idx] + 1e-12) - (y[idx] - y[idx - 1]) / (h[idx - 1] + 1e-12) )

        # 在求解前对矩阵 A 添加小的对角抖动，提升数值稳定性
        jitter = 1e-6
        A = A + torch.eye(N, device=self.device, dtype=A.dtype) * jitter

        M = torch.linalg.solve(A, rhs.unsqueeze(1)).squeeze(1)
        return M


    def _evaluate_spline(self, positions, t_eval, d0=None, dn=None):
        """
        Evaluate cubic spline with optional clamped endpoint derivatives.
        d0: tuple or list (d0x, d0y) or None
        dn: tuple or list (dnx, dny) or None
        Returns Sx, Sy, Sx_dot, Sy_dot
        """
        t = self.t_points
        h = t[1:] - t[:-1]
        # axis x
        if d0 is None:
            Mx = self._solve_natural_cubic_M(positions[:, 0])
            My = self._solve_natural_cubic_M(positions[:, 1])
        else:
            # d0 and dn expected as tuples (d0x,d0y), (dnx,dny)
            d0x, d0y = d0
            dnx, dny = dn
            Mx = self._solve_natural_cubic_M(positions[:, 0], d0=d0x, dn=dnx)
            My = self._solve_natural_cubic_M(positions[:, 1], d0=d0y, dn=dny)
            
        # 增加最小步长保护，避免除零或超大梯度
        h = torch.clamp(h, min=1e-6)

        # rest of original evaluation (unchanged) ...
        idx = torch.searchsorted(t, t_eval, right=False)
        idx = torch.clamp(idx - 1, 0, self.N - 2)

        t_k = t[idx]
        t_k1 = t[idx + 1]
        # 更稳健的 h_k（防止局部为0）
        h_k = torch.clamp((t_k1 - t_k), min=1e-6)
        dt = (t_eval - t_k)

        # fetch per-segment values
        yk_x = positions[idx, 0]
        yk1_x = positions[idx + 1, 0]
        Mk_x = Mx[idx]
        Mk1_x = Mx[idx + 1]

        yk_y = positions[idx, 1]
        yk1_y = positions[idx + 1, 1]
        Mk_y = My[idx]
        Mk1_y = My[idx + 1]

        one_over_h = 1.0 / (h_k)  # 已保证 h_k 非零

        term1_x = Mk_x * (t_k1 - t_eval) ** 3 * (1.0 / (6.0 * h_k))
        term2_x = Mk1_x * (dt ** 3) * (1.0 / (6.0 * h_k))
        term3_x = (yk_x - Mk_x * (h_k ** 2) / 6.0) * ((t_k1 - t_eval) * one_over_h)
        term4_x = (yk1_x - Mk1_x * (h_k ** 2) / 6.0) * (dt * one_over_h)
        Sx = term1_x + term2_x + term3_x + term4_x

        term1_y = Mk_y * (t_k1 - t_eval) ** 3 * (1.0 / (6.0 * h_k))
        term2_y = Mk1_y * (dt ** 3) * (1.0 / (6.0 * h_k))
        term3_y = (yk_y - Mk_y * (h_k ** 2) / 6.0) * ((t_k1 - t_eval) * one_over_h)
        term4_y = (yk1_y - Mk1_y * (h_k ** 2) / 6.0) * (dt * one_over_h)
        Sy = term1_y + term2_y + term3_y + term4_y

        Sx_dot = -Mk_x * (t_k1 - t_eval) ** 2 / (2.0 * h_k) + Mk1_x * (dt ** 2) / (2.0 * h_k) \
                 - (yk_x - Mk_x * (h_k ** 2) / 6.0) * one_over_h + (yk1_x - Mk1_x * (h_k ** 2) / 6.0) * one_over_h

        Sy_dot = -Mk_y * (t_k1 - t_eval) ** 2 / (2.0 * h_k) + Mk1_y * (dt ** 2) / (2.0 * h_k) \
                 - (yk_y - Mk_y * (h_k ** 2) / 6.0) * one_over_h + (yk1_y - Mk1_y * (h_k ** 2) / 6.0) * one_over_h

        return Sx, Sy, Sx_dot, Sy_dot

    def _evaluate_scalar_spline(self, values, t_eval, d0=None, dn=None):
        """
        Evaluate a scalar cubic spline (1D) defined at control points `values`.
        values: (N,) tensor
        returns S (len(t_eval),) and S_dot (len(t_eval),)
        """
        t = self.t_points
        # solve for M
        if d0 is None:
            M = self._solve_natural_cubic_M(values)
        else:
            M = self._solve_natural_cubic_M(values, d0=d0, dn=dn)

        h = torch.clamp(t[1:] - t[:-1], min=1e-6)
        idx = torch.searchsorted(t, t_eval, right=False)
        idx = torch.clamp(idx - 1, 0, self.N - 2)

        t_k = t[idx]
        t_k1 = t[idx + 1]
        h_k = torch.clamp((t_k1 - t_k), min=1e-6)
        dt = (t_eval - t_k)

        yk = values[idx]
        yk1 = values[idx + 1]
        Mk = M[idx]
        Mk1 = M[idx + 1]

        one_over_h = 1.0 / (h_k)
        term1 = Mk * (t_k1 - t_eval) ** 3 * (1.0 / (6.0 * h_k))
        term2 = Mk1 * (dt ** 3) * (1.0 / (6.0 * h_k))
        term3 = (yk - Mk * (h_k ** 2) / 6.0) * ((t_k1 - t_eval) * one_over_h)
        term4 = (yk1 - Mk1 * (h_k ** 2) / 6.0) * (dt * one_over_h)
        S = term1 + term2 + term3 + term4

        S_dot = -Mk * (t_k1 - t_eval) ** 2 / (2.0 * h_k) + Mk1 * (dt ** 2) / (2.0 * h_k) \
                - (yk - Mk * (h_k ** 2) / 6.0) * one_over_h + (yk1 - Mk1 * (h_k ** 2) / 6.0) * one_over_h

        return S, S_dot

    # ------------------------------
    # ---- Utility to get full positions tensor ----
    # ------------------------------
    def _assemble_control_positions(self):
        """Return a torch tensor (N,2) of control positions with current variable positions"""
        fixed0 = self.start_pos
        fixed1 = self.end_pos
        parts = [fixed0.unsqueeze(0), self.variable_positions, fixed1.unsqueeze(0)]
        positions = torch.cat(parts, dim=0)
        return positions  # shape (N,2)

    def _assemble_control_yaws(self):
        """Return a torch tensor (N,) of control yaws with current variable yaws (endpoints fixed)"""
        if self.variable_yaws is None:
            return torch.cat([self.start_yaw.unsqueeze(0), self.end_yaw.unsqueeze(0)])
        return torch.cat([self.start_yaw.unsqueeze(0), self.variable_yaws, self.end_yaw.unsqueeze(0)])

    # ------------------------------
    # ---- Cost function based on interpolated dense trajectory ----
    # ------------------------------
    def cost_function(self):
        # assemble current control positions (requires_grad)
        ctrl_pos = self._assemble_control_positions()  # (N,2)
        ctrl_yaws = self._assemble_control_yaws()     # (N,)

        # 使用 __init__ 中预先生成的自变量采样点和参考样条（避免每次重复构造）
        t_dense = self.t_dense  # 已在 __init__ 缓存
        # 使用预计算的参考样条
        Sx_ref = self.Sx_ref
        Sy_ref = self.Sy_ref
        Sx_dot_ref = self.Sx_dot_ref
        Sy_dot_ref = self.Sy_dot_ref
        traj_ref = torch.stack([Sx_ref, Sy_ref], dim=1)

        # 只评估一次优化样条 —— 使用带端点导数（clamped）以确保端点 yaw 固定
        Sx_opt, Sy_opt, Sx_dot_opt, Sy_dot_opt = self._evaluate_spline(
            ctrl_pos, 
            t_dense, 
            d0=(self.d0x, self.d0y), 
            dn=(self.dnx, self.dny)
        )
        traj_opt = torch.stack([Sx_opt, Sy_opt], dim=1)

        # 计算基于控制点 yaw 的插值（通过对 cos/sin 值做样条，避免角度环绕问题）
        cos_ctrl = torch.cos(ctrl_yaws)
        sin_ctrl = torch.sin(ctrl_yaws)
        Scos_opt, _ = self._evaluate_scalar_spline(cos_ctrl, t_dense)
        Ssin_opt, _ = self._evaluate_scalar_spline(sin_ctrl, t_dense)
        yaw_ctrl_dense = torch.atan2(Ssin_opt, Scos_opt)

        # 后续成本计算（与原来一致）
        # 1) follow cost
        follow_cost = torch.mean(torch.norm(traj_opt - traj_ref, dim=1))

        # 2) smoothness
        vx = Sx_dot_opt
        vy = Sy_dot_opt
        dt = (t_dense[1] - t_dense[0])
        ax = torch.diff(vx, dim=0) / dt
        ay = torch.diff(vy, dim=0) / dt
        accel_mags = torch.sqrt(ax ** 2 + ay ** 2 + 1e-12)
        smooth_cost = torch.mean(accel_mags)

        # 3) yaw consistency: compare interpolated control-yaw with derivative-based yaw
        yaw_opt = torch.atan2(vy, vx)
        yaw_diff = yaw_ctrl_dense - yaw_opt
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        yaw_cost = torch.mean(torch.abs(yaw_diff))

        # 4) curvature
        xdot = vx[:-1]
        ydot = vy[:-1]
        xdd = ax
        ydd = ay
        num = torch.abs(xdot * ydd - ydot * xdd)
        denom = (xdot ** 2 + ydot ** 2) ** 1.5 + 1e-9
        # 更稳健的分母下界，避免速度接近0时曲率爆炸
        denom = torch.clamp(denom, min=1e-3)
        curvature = num / denom
        curvature_threshold = 2.1
        curvature_cost = torch.mean(torch.relu(curvature - curvature_threshold))

        # 5) acceleration cost
        acceleration_cost = torch.mean((ax ** 2 + ay ** 2))

        # 6) terrain-related cost: sample normals/elevation at traj_opt and compute penalty
        terrain_cost = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        from ESDF3d_atpoint import compute_esdf_batch
        if self.nx is not None and self.ny is not None and self.nz is not None:
            # 将坐标和角度按列堆叠为 [steps_to_process, 3]
            queries = torch.stack([
                traj_opt[:, 0],  # x坐标
                traj_opt[:, 1],  # y坐标
                yaw_ctrl_dense    # 使用 control-yaw 插值而不是由位置导数得到的 yaw
            ], dim=1).to(dtype=torch.float32, device=self.device)  # [steps_to_process, 3]
            # 将queries重排为 [steps_to_process, 3]
            queries = queries.t()  # [3, steps_to_process] if compute_esdf_batch expects that; keep as before
            esdf_results = compute_esdf_batch(self.nx, self.ny, self.nz, queries,
                                              resolution=0.1, origin=(-5.0, -5.0),
                                              yaw_weight=4, search_radius=5.0, chunk_cells=3000, device=self.device)
            # 提取并转成tensor（原逻辑）
            if isinstance(esdf_results, (list, tuple)):
                first_elems = []
                for item in esdf_results:
                    if isinstance(item, (list, tuple)):
                        first_val = item[0]
                    else:
                        first_val = item
                    if not isinstance(first_val, torch.Tensor):
                        first_val = torch.tensor(first_val, device=self.device, dtype=torch.float32)
                    else:
                        first_val = first_val.to(device=self.device, dtype=torch.float32)
                    first_elems.append(first_val)
                if len(first_elems) == 0:
                    capsize_esdf = torch.tensor([], device=self.device, dtype=torch.float32)
                else:
                    capsize_esdf = torch.stack(first_elems)
            else:
                capsize_esdf = esdf_results if isinstance(esdf_results, torch.Tensor) else torch.tensor(esdf_results, device=self.device, dtype=torch.float32)

            d_safe = 0.  # 安全距离（米）
            kalpa = 0.6  # 安全损失衰减速率

            # 数值检查并稳健化：避免极端值导致 exp 爆炸
            if capsize_esdf.numel() > 0:
                # 将 capsize_esdf 限制在合理范围，防止异常极值
                capsize_esdf = torch.nan_to_num(capsize_esdf, nan=1e6, posinf=1e6, neginf=-1e6)
                # # z = -(capsize_esdf - d_safe) / kalpa
                # z = (-(capsize_esdf - d_safe) / (kalpa + 1e-12))
                # # clamp z 再用 sigmoid 保证输出有界 [0,1]
                # z_clamped = torch.clamp(z, min=-50.0, max=50.0)
                # cap_penalty = torch.sigmoid(z_clamped)  # bounded penalty
                cap_penalty = torch.exp(-(capsize_esdf - d_safe) / kalpa)
                terrain_cost = cap_penalty.mean()
        
        # 7) endpoints yaw penalty（冗余强约束）
        # endpoints are fixed; we may still penalize mismatch between derivative-derived yaw and control yaw at endpoints
        start_vx = Sx_dot_opt[0]
        start_vy = Sy_dot_opt[0]
        end_vx = Sx_dot_opt[-1]
        end_vy = Sy_dot_opt[-1]
        start_yaw_opt = torch.atan2(start_vy, start_vx)
        end_yaw_opt   = torch.atan2(end_vy, end_vx)
        yaw0 = self.start_yaw
        yawn = self.end_yaw
        yaw_diff_start = torch.atan2(torch.sin(start_yaw_opt - yaw0), torch.cos(start_yaw_opt - yaw0))
        yaw_diff_end   = torch.atan2(torch.sin(end_yaw_opt - yawn),     torch.cos(end_yaw_opt - yawn))
        yaw_endpoint_cost = torch.mean(torch.abs(yaw_diff_start)) + torch.mean(torch.abs(yaw_diff_end))
        
        # 8) 超出地图惩罚
        x_min, x_max = -5.0, 5.0
        y_min, y_max = -5.0, 5.0
        out_points = (traj_opt[:, 0] < x_min) | (traj_opt[:, 0] > x_max) | \
                     (traj_opt[:, 1] < y_min) | (traj_opt[:, 1] > y_max)
        out_of_bounds_cost = torch.sum(out_points.float())  # count

        # combine weights
        # weights = {
        #     'follow': 0.,        # 方向跟随
        #     'smooth': 3e-2,      # 平滑度正则化 2e-1
        #     'yaw': 1e2,          # 朝向一致性 3e-1
        #     'curvature': 5e2,    # 最大曲率约束 5e2
        #     'accelerate': 0e-1,  # 大加速度惩罚
        #     'terrain': 1e2,      # 地形惩罚
        #     'endpoints': 1e4     # 强力惩罚端点 yaw 对齐
        # }
        
        weights = {
            'follow': 0.,        # 方向跟随
            'smooth': 0e-1,      # 平滑度正则化 2e-1
            'yaw': 1e1,         # 朝向一致性 3e-1
            'curvature': 5e2,    # 最大曲率约束 1e1
            'accelerate': 0e-1,  # 大加速度惩罚
            'terrain': 1e2,     # 地形惩罚
            'endpoints': 1e4,    # 强力惩罚端点 yaw 对齐
            'out_of_bounds': 1e3 # 超出地图惩罚
        }

        total_cost = (
            weights['follow'] * follow_cost +
            weights['smooth'] * smooth_cost +
            weights['yaw'] * yaw_cost +
            weights['curvature'] * curvature_cost +
            weights['accelerate'] * acceleration_cost +
            weights['terrain'] * terrain_cost +
            weights['endpoints'] * yaw_endpoint_cost +
            weights['out_of_bounds'] * out_of_bounds_cost
        )
        
        # 再对 total_cost 进行比例调整
        total_cost = total_cost * 1e-1

        return total_cost

    # ------------------------------
    # ---- Optimization loop ----
    # ------------------------------
    def optimize(self, iterations=500, lr=0.01, verbose=True):
        params = [self.variable_positions]
        if self.variable_yaws is not None:
            params.append(self.variable_yaws)
        optimizer = optim.Adam(params, lr=lr)
        best_cost = float('inf')
        best_positions = None
        best_yaws = None
        history = []

        for i in range(iterations):
            optimizer.zero_grad()
            cost = self.cost_function()
            cost.backward()
            # optional gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            history.append(cost.item())
            if verbose and ((i+1) % 20 == 0):
                gnorm_pos = self.variable_positions.grad.norm().item() if self.variable_positions.grad is not None else 0.0
                gnorm_yaw = self.variable_yaws.grad.norm().item() if (self.variable_yaws is not None and self.variable_yaws.grad is not None) else 0.0
                print(f"[{i+1}/{iterations}] cost={cost.item():.6f} grad_pos={gnorm_pos:.6f} grad_yaw={gnorm_yaw:.6f}")

            if cost.item() < best_cost:
                best_cost = cost.item()
                best_positions = self.variable_positions.detach().clone()
                if self.variable_yaws is not None:
                    best_yaws = self.variable_yaws.detach().clone()

        # set best solution
        if best_positions is not None:
            with torch.no_grad():
                self.variable_positions.copy_(best_positions)
        if best_yaws is not None and self.variable_yaws is not None:
            with torch.no_grad():
                self.variable_yaws.copy_(best_yaws)

        # return optimized dense trajectory and history and dense yaw profile
        ctrl_pos_opt = self._assemble_control_positions().detach()
        ctrl_yaw_opt = self._assemble_control_yaws().detach()
        t_dense = torch.linspace(0.0, 1.0, 200, device=self.device, dtype=torch.float32)
        Sx, Sy, Sx_dot, Sy_dot = self._evaluate_spline(ctrl_pos_opt, t_dense, d0=(self.d0x, self.d0y), dn=(self.dnx, self.dny))
        traj = torch.stack([Sx, Sy], dim=1).cpu().numpy()

        # compute dense yaw profile from control yaws
        cos_ctrl = torch.cos(ctrl_yaw_opt)
        sin_ctrl = torch.sin(ctrl_yaw_opt)
        Scos_dense, _ = self._evaluate_scalar_spline(cos_ctrl, t_dense)
        Ssin_dense, _ = self._evaluate_scalar_spline(sin_ctrl, t_dense)
        yaw_dense = torch.atan2(Ssin_dense, Scos_dense).cpu().numpy()

        return traj, yaw_dense, history

# 修改后的轨迹生成和可视化函数
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

if __name__ == "__main__":
    # 从数据集中加载法向量场
    from dataLoader_uneven import UnevenPathDataLoader
    
    # 测试加载数据集
    env_list = ['env000001']
    dataFolder = '/home/yrf/MPT/data/terrain_test/train'
    dataset = UnevenPathDataLoader(env_list, dataFolder)
    path_index = 0
    sample = dataset[path_index]
    
    # 提取高程和法向量场
    elev = sample['elevation']  # 高程图
    nx = sample['map'][0, :, :]  # X方向法向量
    ny = sample['map'][1, :, :]  # Y方向法向量
    nz = sample['map'][2, :, :]  # Z方向法向量
    
    # 提取控制点（起点 + 10个中间点 + 终点）
    points = sample['trajectory'][:12, :].cpu().numpy()  # (x, y, yaw)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = TrajectoryOptimizer(points, nx, ny, nz, device=device)
    
    # 优化前轨迹
    # build dense evaluation parameter (K samples)
    K = 200  # 采样点数，可改
    t_dense = torch.linspace(0.0, 1.0, K, device=device, dtype=torch.float32)
    points_gpu = torch.tensor(points, device=device, dtype=torch.float32)  # (N,3) -> x,y,yaw

    # evaluate optimized spline (differentiable wrt ctrl_pos)
    Sx_opt, Sy_opt, Sx_dot_opt, Sy_dot_opt = optimizer._evaluate_spline(points_gpu, t_dense)
    initial_trajectory = torch.stack([Sx_opt, Sy_opt], dim=1).detach().cpu().numpy()  # (K,2)

    # 执行优化
    optimized_trajectory, optimized_yaw_dense, cost_history = optimizer.optimize(iterations=300, lr=0.1)
    
    # 可视化结果
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    # 优化前轨迹
    visualize_terrain_trajectory(
        ax[0], 
        initial_trajectory, 
        nz,
        nx, 
        ny, 
        nz,
        positions=optimizer.initial_positions,
        yaws=optimizer.yaws
    )
    ax[0].set_title('Initial Trajectory')

    # 使用中间差分计算初始不可达点（保持原有逻辑，建议以后使用 yaw 插值替代）
    x_q = Sx_opt
    y_q = Sy_opt
    
    # 计算中心差分
    x_dot_q = (x_q[2:] - x_q[:-2]) / 2
    y_dot_q = (y_q[2:] - y_q[:-2]) / 2
    yaw_q = torch.atan2(y_dot_q, x_dot_q)
    x_q = x_q[1:-1]
    y_q = y_q[1:-1]
    
    queries = torch.stack([x_q, y_q, yaw_q], dim=1)
    
    from ESDF3d_atpoint import compute_esdf_batch, query_is_unreachable_by_match_batch
    res = compute_esdf_batch(
        nx, ny, nz, queries,
        resolution=0.1, origin=(-5.0, -5.0),
        yaw_weight=1.4, search_radius=5.0, chunk_cells=3000, device=device
    )
    is_unreachables, infos = query_is_unreachable_by_match_batch(res, queries, origin=(-5.0, -5.0), resolution=0.1)
    
    # 绘制出 is_unreachables 对应的xy点
    # is_unreachables 可能为 list/tuple/np.ndarray/torch.Tensor -> 统一转为 numpy 布尔掩码
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
    ax[0].scatter(capsize_points[:, 0], capsize_points[:, 1], c='green', s=20, label='unreachable points')
    ax[0].legend() # 添加图例

    # 添加对于 is_unreachables 的统计信息
    num_capsize = np.sum(capsize_mask)
    print(f"Number of points which is unreachable: {num_capsize} out of {len(is_unreachables)}, percentage: {num_capsize / len(is_unreachables) * 100:.2f}%")

    # 构建完整的优化后控制点数组（包括起点和终点）
    optimized_positions = np.zeros_like(optimizer.initial_positions)
    optimized_positions[optimizer.fixed_indices] = optimizer.initial_positions[optimizer.fixed_indices]  # 起点和终点
    optimized_positions[optimizer.variable_indices] = optimizer.variable_positions.detach().cpu().numpy()  # 优化后的中间点

    # 使用优化后的 control yaws（包含端点）
    ctrl_yaws_opt = optimizer._assemble_control_yaws().detach().cpu().numpy()

    # 可视化优化后轨迹
    visualize_terrain_trajectory(
        ax[1], 
        optimized_trajectory,
        nz,
        nx,
        ny,
        nz,
        positions=optimized_positions,
        yaws=ctrl_yaws_opt
    )
    ax[1].set_title('Optimized Trajectory')

    # 使用 optimizer 返回的密集 yaw 配合轨迹做不可达判断（比中心差分更一致）
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
    ax[1].scatter(capsize_points[:, 0], capsize_points[:, 1], c='green', s=20, label='unreachable points')
    ax[1].legend()
    
    # 添加对于 is_unreachables 的统计信息
    num_capsize = np.sum(capsize_mask)
    print(f"Number of points which is unreachable: {num_capsize} out of {len(is_unreachables)}, percentage: {num_capsize / len(is_unreachables) * 100:.2f}%")

    # 添加代价函数收敛曲线
    fig_cost, ax_cost = plt.subplots(figsize=(10, 5))
    ax_cost.plot(cost_history, 'b-', linewidth=1.5)
    ax_cost.set_title('Optimization Cost History')
    ax_cost.set_xlabel('Iteration')
    ax_cost.set_ylabel('Cost')
    ax_cost.grid(True)
    
    plt.tight_layout()
    plt.show()
