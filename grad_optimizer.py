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
        pts = np.array(points)
        self.N = pts.shape[0]
        self.initial_positions = pts[:, :2].astype(np.float32)  # numpy
        self.yaws = pts[:, 2].astype(np.float32)

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

        # 参考控制点（初始位置） -> 计算一次参考样条（natural spline）
        orig_pos_torch = torch.tensor(self.initial_positions, device=self.device, dtype=torch.float32)
        Sx_ref, Sy_ref, Sx_dot_ref, Sy_dot_ref = self._evaluate_spline(orig_pos_torch.detach(), self.t_dense)

        # 缓存参考轨迹及导数（detach，避免额外梯度运算）
        self.Sx_ref = Sx_ref.detach()
        self.Sy_ref = Sy_ref.detach()
        self.Sx_dot_ref = Sx_dot_ref.detach()
        self.Sy_dot_ref = Sy_dot_ref.detach()

        # 预计算端点的参考速度幅值并根据固定 yaw 构造端点一阶导数 d0/dn（tensor）
        speed_start = torch.sqrt(self.Sx_dot_ref[0] ** 2 + self.Sy_dot_ref[0] ** 2) + 1e-6
        speed_end   = torch.sqrt(self.Sx_dot_ref[-1] ** 2 + self.Sy_dot_ref[-1] ** 2) + 1e-6

        yaw0 = self.start_yaw
        yawn = self.end_yaw

        self.d0x = (torch.cos(yaw0) * speed_start).to(device=self.device, dtype=torch.float32)
        self.d0y = (torch.sin(yaw0) * speed_start).to(device=self.device, dtype=torch.float32)
        self.dnx = (torch.cos(yawn) * speed_end).to(device=self.device, dtype=torch.float32)
        self.dny = (torch.sin(yawn) * speed_end).to(device=self.device, dtype=torch.float32)


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

        # interior rows
        for i in range(1, N-1):
            A[i, i-1] = h[i-1]
            A[i, i]   = 2.0 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            rhs[i] = 6.0 * ( (y[i+1] - y[i]) / (h[i] + 1e-12) - (y[i] - y[i-1]) / (h[i-1] + 1e-12) )

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

        # rest of original evaluation (unchanged) ...
        idx = torch.searchsorted(t, t_eval, right=False)
        idx = torch.clamp(idx - 1, 0, self.N - 2)

        t_k = t[idx]
        t_k1 = t[idx + 1]
        h_k = (t_k1 - t_k)
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

        one_over_h = 1.0 / (h_k + 1e-12)

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

    # ------------------------------
    # ---- Utility to get full positions tensor ----
    # ------------------------------
    def _assemble_control_positions(self):
        """Return a torch tensor (N,2) of control positions with current variable positions"""
        # 使用已存的常量张量（已 detach），避免每次重建可能带来的梯度通道
        fixed0 = self.start_pos
        fixed1 = self.end_pos
        parts = [fixed0.unsqueeze(0), self.variable_positions, fixed1.unsqueeze(0)]
        positions = torch.cat(parts, dim=0)
        return positions  # shape (N,2)

    # ------------------------------
    # ---- Cost function based on interpolated dense trajectory ----
    # ------------------------------
    def cost_function(self):
        # assemble current control positions (requires_grad)
        ctrl_pos = self._assemble_control_positions()  # (N,2)

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

        # 3) yaw consistency
        yaw_opt = torch.atan2(vy, vx)
        yaw_ref = torch.atan2(Sy_dot_ref, Sx_dot_ref)
        yaw_diff = yaw_opt - yaw_ref
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        yaw_cost = torch.mean(torch.abs(yaw_diff))

        # 4) curvature
        xdot = vx[:-1]
        ydot = vy[:-1]
        xdd = ax
        ydd = ay
        num = torch.abs(xdot * ydd - ydot * xdd)
        denom = (xdot ** 2 + ydot ** 2) ** 1.5 + 1e-9
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
                yaw_opt    # 预测角度
            ], dim=1).to(dtype=torch.float32, device=nx.device)  # [steps_to_process, 3]
            # 将queries重排为 [steps_to_process, 3]
            queries = queries.t()  # [steps_to_process, 3]
            esdf_results = compute_esdf_batch(nx, ny, nz, queries,
                                              resolution=0.1, origin=(-5.0, -5.0),
                                              yaw_weight=1.4, search_radius=5.0, chunk_cells=3000, device=nx.device)
            # 提取每个返回项的第一个元素
            if isinstance(esdf_results, (list, tuple)):
                first_elems = []
                for item in esdf_results:
                    if isinstance(item, (list, tuple)):
                        first_val = item[0]
                    else:
                        first_val = item
                    # 将非 tensor 转为 tensor，并放到正确设备、类型
                    if not isinstance(first_val, torch.Tensor):
                        first_val = torch.tensor(first_val, device=nx.device, dtype=torch.float32)
                    else:
                        first_val = first_val.to(device=nx.device, dtype=torch.float32)
                    first_elems.append(first_val)
                if len(first_elems) == 0:
                    # 兜底：避免空列表导致后续错误
                    capsize_esdf = torch.tensor([], device=nx.device, dtype=torch.float32)
                else:
                    capsize_esdf = torch.stack(first_elems)  # [steps, ...]
            else:
                # 如果直接返回单个值或单个 tensor，直接使用
                capsize_esdf = esdf_results if isinstance(esdf_results, torch.Tensor) else torch.tensor(esdf_results, device=nx.device, dtype=torch.float32)
            
            d_safe = 0.  # 安全距离（米）
            kalpa = 0.6  # 安全损失衰减速率
            
            # 如果 capsize_esdf 为空则跳过；否则按元素计算并取平均作为损失（可按需改为逐点加权）
            if capsize_esdf.numel() > 0:
                # 将倾覆损失的梯度设置为可训练
                terrain_cost = torch.tensor(0.0, device=nx.device, requires_grad=True)
                capsize_loss_per_point = torch.exp(-(capsize_esdf - d_safe) / kalpa)
                terrain_cost = capsize_loss_per_point.mean()
        
        # 7) endpoints yaw penalty（冗余强约束）
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

        # combine weights
        weights = {
            'follow': 0.,       # 方向跟随
            'smooth': 1e-1,     # 平滑度正则化 2e-1
            'yaw': 0e-1,        # 朝向一致性 3e-1
            'curvature': 1e1,   # 最大曲率约束 1e1
            'accelerate': 0.,   # 大加速度惩罚
            'terrain': 1e3,     # 地形惩罚
            'endpoints': 1e4    # 强力惩罚端点 yaw 对齐
        }

        total_cost = (
            weights['follow'] * follow_cost +
            weights['smooth'] * smooth_cost +
            weights['yaw'] * yaw_cost +
            weights['curvature'] * curvature_cost +
            weights['accelerate'] * acceleration_cost +
            weights['terrain'] * terrain_cost +
            weights['endpoints'] * yaw_endpoint_cost
        )
        
        # 再对 total_cost 进行比例调整
        total_cost = total_cost * 1e-3

        return total_cost

    # ------------------------------
    # ---- Optimization loop ----
    # ------------------------------
    def optimize(self, iterations=500, lr=0.01, verbose=True):
        optimizer = optim.Adam([self.variable_positions], lr=lr)
        best_cost = float('inf')
        best_positions = None
        history = []

        for i in range(iterations):
            optimizer.zero_grad()
            cost = self.cost_function()
            cost.backward()
            # optional gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([self.variable_positions], max_norm=1.0)
            optimizer.step()

            history.append(cost.item())
            if verbose and ((i+1) % 20 == 0):
                gnorm = self.variable_positions.grad.norm().item() if self.variable_positions.grad is not None else 0.0
                print(f"[{i+1}/{iterations}] cost={cost.item():.6f} grad_norm={gnorm:.6f}")

            if cost.item() < best_cost:
                best_cost = cost.item()
                best_positions = self.variable_positions.detach().clone()

        # set best solution
        if best_positions is not None:
            with torch.no_grad():
                self.variable_positions.copy_(best_positions)

        # return optimized dense trajectory and history
        ctrl_pos_opt = self._assemble_control_positions().detach()
        t_dense = torch.linspace(0.0, 1.0, 500, device=self.device, dtype=torch.float32)
        Sx, Sy, _, _ = self._evaluate_spline(ctrl_pos_opt, t_dense)
        traj = torch.stack([Sx, Sy], dim=1).cpu().numpy()
        return traj, history

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
    env_list = ['env000009']
    dataFolder = '/home/yrf/MPT/data/terrain_dataset/val'
    dataset = UnevenPathDataLoader(env_list, dataFolder)
    path_index = 0
    sample = dataset[path_index]
    
    # 提取高程和法向量场
    elev = sample['elevation']  # 高程图
    nx = sample['map'][0, :, :]  # X方向法向量
    ny = sample['map'][1, :, :]  # Y方向法向量
    nz = sample['map'][2, :, :]  # Z方向法向量
    # nz = torch.abs(nz)  # 取Z方向法向量的绝对值作为高度
    
    # 提取控制点（起点 + 10个中间点 + 终点）
    points = sample['trajectory'][:12, :].cpu().numpy()  # (x, y, yaw)

    # # 将10个中间点的yaw值设置为起点到终点的朝向
    # start_yaw = points[0, 2]
    # end_yaw = points[-1, 2]
    # points[1:-1, 2] = np.linspace(start_yaw, end_yaw, num=10)
    
    # # 将10个中间点的位置设置为起点和终点之间的线性插值
    # start_pos = points[0, :2]
    # end_pos = points[-1, :2]
    # for i in range(1, len(points) - 1):
    #     t = i / (len(points) - 1)
    #     points[i, 0] = start_pos[0] + t * (end_pos[0] - start_pos[0])
    #     points[i, 1] = start_pos[1] + t * (end_pos[1] - start_pos[1])

    # 创建轨迹优化器
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
    optimized_trajectory, cost_history = optimizer.optimize(iterations=300, lr=0.05)
    
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

    # 使用相邻段方向：内部点用前后段平均，端点用相邻段方向
    diffs = optimized_positions[1:] - optimized_positions[:-1]  # (N-1, 2)
    seg_yaws = np.arctan2(diffs[:, 1], diffs[:, 0])  # 每段的航向 (N-1,)
    N = optimized_positions.shape[0]
    optimized_yaws = np.zeros(N, dtype=float)
    if N >= 2:
        optimized_yaws[0] = seg_yaws[0]
        optimized_yaws[-1] = seg_yaws[-1]
    if N > 2:
        # 内部点取相邻段航向平均并归一化
        for i in range(1, N-1):
            a = seg_yaws[i-1]
            b = seg_yaws[i]
            # 平均角度，考虑环绕
            diff = np.arctan2(np.sin(b - a), np.cos(b - a))
            optimized_yaws[i] = a + diff * 0.5
        # 归一化到[-pi, pi]
        optimized_yaws = (optimized_yaws + np.pi) % (2 * np.pi) - np.pi

    # 优化后轨迹（传入更新后的朝向）
    visualize_terrain_trajectory(
        ax[1], 
        optimized_trajectory,
        # elev,
        nz,
        nx,
        ny,
        nz,
        positions=optimized_positions,
        yaws=optimized_yaws
    )
    ax[1].set_title('Optimized Trajectory')

    x_q = torch.tensor(optimized_trajectory[:, 0], device=device, dtype=torch.float32)
    y_q = torch.tensor(optimized_trajectory[:, 1], device=device, dtype=torch.float32)
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
    mid_points = optimized_trajectory[1:-1]
    if capsize_mask.shape[0] != mid_points.shape[0]:
        # 尽量截断或扩展掩码到匹配长度（以 False 填充）
        mask = np.zeros(mid_points.shape[0], dtype=bool)
        L = min(mask.shape[0], capsize_mask.shape[0])
        mask[:L] = capsize_mask[:L]
        capsize_mask = mask
    capsize_points = mid_points[capsize_mask]
    ax[1].scatter(capsize_points[:, 0], capsize_points[:, 1], c='green', s=20, label='unreachable points')
    ax[1].legend() # 添加图例

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