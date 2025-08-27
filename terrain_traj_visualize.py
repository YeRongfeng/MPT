"""
terrain_visualize.py - 绘制3d地形图的脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3D
from scipy.interpolate import griddata

import os.path as osp
import pickle

def plot_terrain(elevation, nx, ny, nz, pred_traj=None, true_traj=None,
                 pred_color="#CD1F4D", true_color='#0072B2'):
    """
    绘制3D地形图
    :param terrain_data: 2D numpy array, 每个元素表示地形高度
    :param title: 图表标题
    """
    h, w = elevation.shape
    
    elevation = elevation / 0.1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 使用单元格（cells）生成相连的方形切平面（在 XY 投影上无缝拼接）
    plane_size = 1.0
    polys = []
    for i in range(0, h - 1):
        for j in range(0, w - 1):
            # 单元格四角的平均法线和单元格中心高度作为该单元的平面法线与中心高度
            n = np.array([
                np.mean(nx[i:i+2, j:j+2]),
                np.mean(ny[i:i+2, j:j+2]),
                np.mean(nz[i:i+2, j:j+2])
            ], dtype=np.float32)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-6:
                continue
            n = n / n_norm

            # 单元格中心高度（用于平面方程）
            center_h = float(np.mean(elevation[i:i+2, j:j+2]))
            center = np.array([j + 0.5, i + 0.5, center_h], dtype=np.float32)

            # 单元格在 XY 平面上的四个角 (x,y)
            corners_xy = [(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)]

            # 为了保证相邻单元严格拼接（共享相同顶点高度），使用格点高度作为四角高度
            zs = [float(elevation[y, x]) for (x, y) in corners_xy]

            # 构造四角三维点（顺序为 cell 四角，保证相邻单元共享边/顶点）
            corners = [np.array([float(x), float(y), z], dtype=np.float32)
                       for (x, y), z in zip(corners_xy, zs)]

            polys.append(corners)

    # 按单元格顺序采样颜色（高度映射），使用简约且感知均匀的配色
    if len(polys) > 0:
        zmin = np.nanmin(elevation)
        zmax = np.nanmax(elevation)
        denom = zmax - zmin if (zmax - zmin) > 1e-8 else 1.0

        # 选择配色（保留原有自定义配色，并额外添加多组更深色调供挑选）
        # 把 palette_name 设为下列任意键（示例： 'deep_ocean' 或新增 'rust' 等）
        palette_name = 'deep_moss_r'

        from matplotlib.colors import LinearSegmentedColormap

        palettes = {
            # 原有自定义配色（保留）
            'sepia':        ['#ffffff', '#f3ead8', '#cda66a', '#6b3d12'],
            'greens':       ['#ffffff', '#eaf6ea', '#9fcc9f', '#256828'],
            'grayscale':    ['#ffffff', '#e0e0e0', '#808080', '#141414'],
            'redbrown':     ['#ffffff', '#f7eae8', '#e07a5f', '#7a2e2e'],
            'deep_ocean':   ['#0b2745', '#16325a', '#2b557a', '#4a7aa6'],
            'slate':        ['#0f1720', '#22303f', '#44535f', '#6b8099'],
            'charcoal':     ['#0b0b0b', '#2b2b2b', '#555555', '#8a8a8a'],
            'forest_dark':  ['#081f12', '#0f3a24', '#1f5838', '#2e7a50'],
            'wine':         ['#2a0a0f', '#541423', '#7b2431', '#a33b4b'],
            'ochre_deep':   ['#2f1a00', '#5a3510', '#8a5b21', '#b9893d'],
            'science_neutral': ['#ffffff', '#f0efe9', '#d5cbb8', '#8f6f4a'],
            'warm_rock':    ['#fff7f0', '#f2d7c9', '#d99b6c', '#7f3f2a'],
            'coal':         ['#f7f7f7', '#d9d9d9', '#969696', '#252525'],

            # 新增深色调（轻色不要接近纯白，保证印刷/背景对比）
            'deep_moss':    ['#d7e6d9', '#8fbf90', '#4a7a4c', '#183a1f'],
            'deep_moss_r':    ["#385334", "#578c57", "#69ad5f", "#8ec18d"],
            'rust':         ['#f0d0c6', '#d38b71', '#a94f30', '#4f1b14'],
            'sandstone':    ['#eadccb', '#d0b187', '#a06f46', '#5a3a28'],
            'moody_teal':   ['#d9e9ea', '#79b3b6', '#2f7c7f', '#0f3f40'],
            'indigo_evening':['#d6d9ef','#8a90d6','#39459a','#0b1c4a'],
            'indigo_evening_r':['#0b1c4a','#39459a','#8a90d6',"#96a1f2"],
            'plum_deep':    ['#e5d7e8', '#b47fbf', '#7a3b88', '#3a103f'],
            'brick':        ['#f0d3cf', '#d07a6b', '#9b3f2f', '#501811'],
            'sepia_deep':   ['#e8d7c5', '#c9996a', '#8a5a34', '#402616'],
            'midnight':     ['#cfd7e6', '#99a9d1', '#41518f', '#071034'],
            'volcanic':     ['#e6d6d0', '#c47b6a', '#8d3d2f', '#3a0f0b'],
            'olive_deep':   ['#dfe6d4', '#9fb07a', '#5f6b3a', '#2b3518'],
            'copper':       ['#ead7cf', '#c58b6b', '#8f4f33', '#4a2a16'],
            'smoke':        ['#d9dbe0', '#a3a7b0', '#6a6f78', '#2b2f34'],
            'ash':          ['#dcdfe1', '#acb3b5', '#6f7577', '#2e3233'],
            'maroon':       ['#e6d8d9', '#b77a85', '#7a2f3c', '#3b0f12'],
            'pine':         ['#dbe6de', '#8fb09a', '#3f6b53', '#183a2a'],
            'clay':         ['#e8d9d4', '#c79b86', '#8b5f4b', '#4b2a22'],
            'slate_deep':   ['#dbe0e6', '#99a6b6', '#566675', '#22313b'],

            'custom':      ["#012911", "#0D592F", "#219136", "#40B759"],  # 备用：自定义配色（浅灰到深灰）
        }

        builtin_palettes = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'cubehelix', 'terrain']

        if palette_name in palettes:
            cmap = LinearSegmentedColormap.from_list('terrain_pub', palettes[palette_name], N=256)
        elif palette_name in builtin_palettes:
            cmap = plt.get_cmap(palette_name)
        else:
            # 回退：深色调 deep_ocean
            cmap = LinearSegmentedColormap.from_list('terrain_pub', palettes['deep_ocean'], N=256)

        colors = []
        # polys 顺序与单元格遍历一致，按单元格平均高度采样
        for i in range(0, h - 1):
            for j in range(0, w - 1):
                if len(colors) >= len(polys):
                    break
                zval = float(np.mean(elevation[i:i+2, j:j+2]))
                t = (zval - zmin) / denom
                # 稍微压暗中高位以增强对比
                t = np.clip(t**0.9, 0.0, 1.0)
                colors.append(cmap(t))

        # 若数量不够，填充中性深灰而不是浅灰
        while len(colors) < len(polys):
            colors.append((0.78, 0.78, 0.78, 1.0))

        # 增强边缘对比但仍保持细微（便于印刷与后续叠加）
        coll = Poly3DCollection(polys, facecolors=colors,
                                edgecolors=(0.0, 0.0, 0.0, 0.12), linewidths=0.25)
        # 将地形集合的 zorder 调低，优先显示轨迹
        try:
            coll.set_zorder(0)
        except Exception:
            pass
        ax.add_collection3d(coll)
        
    # 设置连接线颜色为黑色
    coll.set_edgecolor((0.0, 0.0, 0.0, 1.0))
    coll.set_linewidth(0.1)

    # 设置范围并彻底隐藏坐标系
    ax.set_xlim(0, w - 1)
    ax.set_ylim(0, h - 1)
    ax.set_zlim(np.nanmin(elevation), np.nanmax(elevation))

    try:
        ax.set_axis_off()
    except Exception:
        pass
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    # 设置范围并使三轴比例一致（保证 z 轴与地面单位一致）
    ax.set_xlim(0, w - 1)
    ax.set_ylim(0, h - 1)
    zmin = np.nanmin(elevation)
    zmax = np.nanmax(elevation)
    ax.set_zlim(zmin, zmax)

    # 计算 z 的范围，防止为 0
    z_range = float(zmax - zmin) if (zmax - zmin) > 1e-8 else 1.0

    try:
        # 为了保证 x,y,z 三个方向的相对单位长度一致，
        # 先取三轴范围的最大值作为归一化尺度，然后传入 set_box_aspect
        sx = float(w - 1)
        sy = float(h - 1)
        sz = float(z_range)
        scale = max(sx, sy, sz, 1e-8)
        ax.set_box_aspect((sx / scale, sy / scale, sz / scale))
    except Exception:
        # 旧版本 matplotlib 没有该接口，保持原样（可考虑升级 matplotlib）
        pass

    for waxis in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
        try:
            waxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            waxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        except Exception:
            pass

    ax.grid(False)
    ax.view_init(elev=45, azim=-120)

    # -------------------------
    # Trajectory 绘制部分（可选）
    # pred_traj / true_traj: array-like of shape (N,3) or (N,2) in (x,y[,theta]) 格式，坐标以栅格索引为单位
    # pred_traj 将被绘制在 true_traj 之上，并且两者都将使用平滑插值
    # -------------------------
    def smooth_traj_xy(traj_xy, num=300):
        """对 (N,2) 的轨迹做样条或线性插值并返回 (num,2) 点集和切向角theta"""
        traj_xy = np.asarray(traj_xy, dtype=np.float64)
        if traj_xy.ndim != 2 or traj_xy.shape[0] == 0:
            return np.zeros((0, 3))
        x = traj_xy[:, 0]
        y = traj_xy[:, 1]
        if len(x) < 2:
            theta = np.zeros_like(x)
            pts = np.stack([np.repeat(x, num), np.repeat(y, num), np.repeat(theta, num)], axis=1)
            return pts
        try:
            from scipy import interpolate as si
            k = min(3, len(x) - 1)
            tck, u = si.splprep([x, y], s=0.0, k=k)
            u_fine = np.linspace(0, 1, num)
            x_fine, y_fine = si.splev(u_fine, tck)
            dx, dy = si.splev(u_fine, tck, der=1)
            theta_fine = np.arctan2(dy, dx)
            return np.stack([x_fine, y_fine, theta_fine], axis=1)
        except Exception:
            # fallback: 累积弧长线性插值 + 简单平滑
            dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
            u = np.zeros(len(x))
            if len(dist) > 0:
                u[1:] = np.cumsum(dist)
            if u[-1] == 0:
                u = np.linspace(0, 1, len(x))
            else:
                u = u / u[-1]
            u_fine = np.linspace(0, 1, num)
            x_fine = np.interp(u_fine, u, x)
            y_fine = np.interp(u_fine, u, y)
            # 平滑
            def smooth_arr(a, win=7):
                if len(a) < win:
                    return a
                pad = win // 2
                a_pad = np.pad(a, (pad, pad), mode='edge')
                kernel = np.ones(win) / win
                return np.convolve(a_pad, kernel, mode='valid')
            x_s = smooth_arr(x_fine)
            y_s = smooth_arr(y_fine)
            dx = np.gradient(x_s)
            dy = np.gradient(y_s)
            theta_fine = np.arctan2(dy, dx)
            return np.stack([x_s, y_s, theta_fine], axis=1)

    def sample_elevation_at_xy(xy_pts):
        """给定 (M,2) 的 (x,y)（对应列,行），用 griddata 从 elevation 中插值返回 z 值数组"""
        if xy_pts is None or len(xy_pts) == 0:
            return np.array([])
        # grid points: X=j (cols), Y=i (rows)
        h, w = elevation.shape
        Xi, Yi = np.meshgrid(np.arange(w), np.arange(h))
        points = np.stack([Xi.ravel(), Yi.ravel()], axis=1)
        values = elevation.ravel()
        pts = np.asarray(xy_pts)[:, :2]
        try:
            z = griddata(points, values, pts, method='linear')
            # 对 NaN 使用最近邻填补
            nan_mask = np.isnan(z)
            if nan_mask.any():
                z_nn = griddata(points, values, pts[nan_mask], method='nearest')
                z[nan_mask] = z_nn
            return z
        except Exception:
            # 简单边界裁切
            pts_clipped = np.copy(pts)
            pts_clipped[:, 0] = np.clip(pts_clipped[:, 0], 0, w - 1)
            pts_clipped[:, 1] = np.clip(pts_clipped[:, 1], 0, h - 1)
            z = []
            for xx, yy in pts_clipped:
                z.append(float(elevation[int(round(yy)), int(round(xx))]))
            return np.array(z)

    def world_to_grid_coords(xy_pts, h, w):
        """自动检测并把世界坐标系(-5..5)转换为栅格索引(0..w-1,0..h-1)。
        如果输入已经接近栅格索引范围则直接返回原值。
        xy_pts: (N,2) array-like (x,y)
        返回 (N,2) 的浮点数组，列(x)->[0,w-1], 行(y)->[0,h-1]
        """
        pts = np.asarray(xy_pts, dtype=np.float64)
        if pts.size == 0:
            return pts
        # 记录原始范围
        xmin, xmax = np.nanmin(pts[:, 0]), np.nanmax(pts[:, 0])
        ymin, ymax = np.nanmin(pts[:, 1]), np.nanmax(pts[:, 1])

        # 如果看起来像世界坐标（落在 -5..5 以内），则进行转换
        if xmin >= -5.5 and xmax <= 5.5 and ymin >= -5.5 and ymax <= 5.5:
            # world x in [-5,5] -> col = (x+5)/0.1 ; similarly for row
            scale = 0.1
            cols = (pts[:, 0] + 5.0) / scale
            rows = (pts[:, 1] + 5.0) / scale
            cols = np.clip(cols, 0, w - 1)
            rows = np.clip(rows, 0, h - 1)
            return np.stack([cols, rows], axis=1)

        # 否则假定已经是栅格索引（x->col, y->row），但仍裁剪到边界
        cols = np.clip(pts[:, 0], 0, w - 1)
        rows = np.clip(pts[:, 1], 0, h - 1)
        return np.stack([cols, rows], axis=1)

    # 绘制轨迹（先绘制真实，再绘制预测，使预测在上方）
    zmin = np.nanmin(elevation)
    zmax = np.nanmax(elevation)
    z_range = max((zmax - zmin), 1e-8)

    # 真是轨迹
    if true_traj is not None:
        try:
            tr = np.asarray(true_traj)
            if tr.ndim == 1:
                tr = tr.reshape(-1, tr.shape[0])
            # 只取 x,y
            tr_xy = tr[:, :2]
            # 坐标转换：如果轨迹为 world 坐标 (-5..5)，将其转换为栅格索引
            tr_xy_grid = world_to_grid_coords(tr_xy, h, w)
            smooth = smooth_traj_xy(tr_xy_grid, num=300)
            zs = sample_elevation_at_xy(smooth[:, :2])
            if zs.size == 0:
                # 当无法采样高度时，将轨迹放在地形顶部
                zs = np.full(smooth.shape[0], zmax)
            # 小幅悬空避免与面片精确重合：提高到相对于地形高度的可见偏移
            # 将真实轨迹抬高为较小的正值，确保可见且不过于“漂浮”
            offset = 0.012 * z_range
            xs = smooth[:, 0]
            ys = smooth[:, 1]
            zs_plot = zs + offset
            # 阴影底层线以增加对比
            try:
                line_shadow = Line3D(xs, ys, zs_plot, color=(0, 0, 0, 0.18), linewidth=6.0, solid_capstyle='round')
                line_shadow.set_zorder(10)
                ax.add_line(line_shadow)
            except Exception:
                ax.plot(xs, ys, zs_plot, color=(0, 0, 0, 0.18), linewidth=6.0, zorder=10)
            # 主线
            try:
                line_main = Line3D(xs, ys, zs_plot, color=true_color, linewidth=3.0, linestyle='--', solid_capstyle='round')
                line_main.set_zorder(20)
                ax.add_line(line_main)
            except Exception:
                ax.plot(xs, ys, zs_plot, color=true_color, linewidth=3.0, linestyle='--', zorder=20)
            # 起点/终点标记（带黑色边框）
            if xs.size > 0:
                ax.scatter(xs[0], ys[0], zs_plot[0], color=true_color, s=140, edgecolors='k', linewidths=0.8, zorder=5)
                ax.scatter(xs[-1], ys[-1], zs_plot[-1], color=true_color, s=140, edgecolors='k', linewidths=0.8, zorder=5)
        except Exception:
            pass

    # 预测轨迹（覆盖在上方，颜色不同）
    if pred_traj is not None:
        try:
            pr = np.asarray(pred_traj)
            if pr.ndim == 1:
                pr = pr.reshape(-1, pr.shape[0])
            pr_xy = pr[:, :2]
            # 如果存在真实轨迹，从真实轨迹中取起点和终点并加入预测轨迹两端
            if true_traj is not None:
                try:
                    tt = np.asarray(true_traj)
                    if tt.ndim == 1:
                        tt = tt.reshape(-1, tt.shape[0])
                    start_pt = tt[0, :2].astype(np.float64)
                    goal_pt = tt[-1, :2].astype(np.float64)
                    # 若预测点已包含起点/终点则不重复（基于距离判断）
                    def near(a, b, tol=1e-3):
                        return np.linalg.norm(np.asarray(a) - np.asarray(b)) < tol
                    # 如果第一个预测点非常接近 start，则不插入 start
                    if pr_xy.shape[0] > 0 and near(pr_xy[0], start_pt):
                        prefix = np.empty((0, 2))
                    else:
                        prefix = start_pt.reshape(1, 2)
                    if pr_xy.shape[0] > 0 and near(pr_xy[-1], goal_pt):
                        suffix = np.empty((0, 2))
                    else:
                        suffix = goal_pt.reshape(1, 2)
                    pr_xy = np.vstack((prefix, pr_xy, suffix)) if pr_xy.size else np.vstack((prefix, suffix))
                except Exception:
                    # 若真实轨迹解析失败，继续使用原始预测点
                    pass
            pr_xy_grid = world_to_grid_coords(pr_xy, h, w)
            smooth_p = smooth_traj_xy(pr_xy_grid, num=300)
            zs_p = sample_elevation_at_xy(smooth_p[:, :2])
            if zs_p.size == 0:
                zs_p = np.full(smooth_p.shape[0], zmax)
            # 将预测轨迹抬高以显示在上方（比真实轨迹更高以保证覆盖）
            offset_p = 0.035 * z_range
            xs_p = smooth_p[:, 0]
            ys_p = smooth_p[:, 1]
            zs_plot_p = zs_p + offset_p
            # 预测轨迹底层柔和阴影
            try:
                line_shadow_p = Line3D(xs_p, ys_p, zs_plot_p, color=(0, 0, 0, 0.12), linewidth=8.0, solid_capstyle='round')
                line_shadow_p.set_zorder(30)
                ax.add_line(line_shadow_p)
            except Exception:
                ax.plot(xs_p, ys_p, zs_plot_p, color=(0, 0, 0, 0.12), linewidth=8.0, zorder=30)
            # 主预测线（更粗以突出）
            try:
                line_main_p = Line3D(xs_p, ys_p, zs_plot_p, color=pred_color, linewidth=4.2, solid_capstyle='round')
                line_main_p.set_zorder(40)
                ax.add_line(line_main_p)
            except Exception:
                ax.plot(xs_p, ys_p, zs_plot_p, color=pred_color, linewidth=4.2, zorder=40)
            # 起点/终点标记（菱形以示区别）
            if xs_p.size > 0:
                ax.scatter(xs_p[0], ys_p[0], zs_plot_p[0], color=pred_color, s=180, marker='D', edgecolors='k', linewidths=1.0, zorder=7)
                ax.scatter(xs_p[-1], ys_p[-1], zs_plot_p[-1], color=pred_color, s=180, marker='D', edgecolors='k', linewidths=1.0, zorder=7)
        except Exception:
            pass

    # # 尝试全窗口显示（兼容不同后端）
    # try:
    #     mng = plt.get_current_fig_manager()
    #     try:
    #         # 常见后端：TkAgg/Qt5Agg 等
    #         mng.full_screen_toggle()
    #     except Exception:
    #         try:
    #             mng.window.showMaximized()
    #         except Exception:
    #             pass
    # except Exception:
    #     pass

    # 保存为 PNG / SVG / PDF（保存在当前工作目录，文件名前缀 terrain_render）
    out_base = osp.join('.', 'terrain_render')
    try:
        fig.savefig(out_base + '.png', dpi=1000, bbox_inches='tight')
        fig.savefig(out_base + '.svg', bbox_inches='tight')
        fig.savefig(out_base + '.pdf', bbox_inches='tight')
    except Exception:
        # 保存失败时不阻塞显示
        pass

    plt.show()
    
if __name__ == "__main__":
    envNum = np.random.randint(0, 99)  # 随机选择环境id
    envList = ['env000009']  # 生成环境列表，格式为 env000000, env000001, ..., env000009
    dataset_path = 'data/terrain/train'
    
    env_path = osp.join(dataset_path, envList[0])
    
    # 获取地图信息
    map_file = osp.join(env_path, 'map.p')
    with open(map_file, 'rb') as f:
        env_data = pickle.load(f)
    map_tensor = env_data['tensor']  # shape (H, W, B)
    elevation = map_tensor[:, :, 0]  # 高度图 (H, W)
    normal_x = map_tensor[:, :, 1]  # 法向量 x 分量 (H, W)
    normal_y = map_tensor[:, :, 2]  # 法向量 y 分量 (H, W)
    normal_z = map_tensor[:, :, 3]  # 法向量 z 分量 (H, W)

    # 尝试加载一条训练样本轨迹（path_*.p），若不存在则只绘制地形
    true_traj = None
    pred_traj = None
    try:
        # 选择一个示例路径编号（可修改为所需编号）
        example_paths = [30, 47, 48, 49, 44, 45]
        chosen = example_paths[0]
        path_file = osp.join(env_path, f'path_{chosen}.p')
        with open(path_file, 'rb') as pf:
            path_data = pickle.load(pf)
            true_traj = np.asarray(path_data.get('path', None))
    except Exception:
        # 若加载失败，保持 true_traj 为 None
        true_traj = None

    # 尝试使用已有的 eval_model_uneven.get_patch 来生成预测轨迹（若依赖缺失则跳过）
    try:
        import torch
        import json
        from eval_model_uneven import get_patch
        from transformer import Models

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        modelFolder = 'data/uneven'
        modelFile = osp.join(modelFolder, f'model_params.json')
        if osp.exists(modelFile):
            model_param = json.load(open(modelFile))
            transformer = Models.UnevenTransformer(**model_param)
            transformer = transformer.to(device)
            # 尝试加载常见检查点文件（stage2 优先）
            ckpt = None
            for stage in (2, 1):
                fname = osp.join(modelFolder, f'stage{stage}_model_epoch_79.pkl')
                if osp.exists(fname):
                    try:
                        ckpt = torch.load(fname, map_location=device)
                        break
                    except Exception:
                        ckpt = None
            if ckpt is not None and 'state_dict' in ckpt:
                transformer.load_state_dict(ckpt['state_dict'])
                transformer.eval()
                # 若有 true_traj，则用其起终点做一次推理，否则跳过
                if true_traj is not None and len(true_traj) >= 2:
                    start_pose = true_traj[0]
                    goal_pose = true_traj[-1]
                    try:
                        _, _, pred = get_patch(transformer, start_pose, goal_pose, normal_x, normal_y, normal_z)
                        # pred 可能是 list of tuples
                        pred_traj = np.asarray(pred)
                    except Exception:
                        pred_traj = None
    except Exception:
        pred_traj = None

    plot_terrain(elevation, normal_x, normal_y, normal_z, pred_traj=pred_traj, true_traj=true_traj)