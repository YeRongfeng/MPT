"""
terrain_visualize.py - 绘制3d地形图的脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata

import os.path as osp
import pickle

def plot_terrain(elevation, nx, ny, nz):
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
        palette_name = 'indigo_evening_r'

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

    plot_terrain(elevation, normal_x, normal_y, normal_z)