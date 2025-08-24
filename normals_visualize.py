"""
normals_visualize.py - 仅绘制 nz 并在其上叠加由 nx, ny 组成的归一化坡度方向箭头
已包含多种期刊友好配色（可直接选择名字），并提供一个白中心自定义发散配色用于正负值表达。
"""

import os.path as osp
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# 10 个柔和渐变配色（面向期刊插图，强调平滑与可印刷性，避免强烈对比）
CUSTOM_CMAP_HEX = {
    "soft_ocean": [
        "#eaf6f6", "#cfeff0", "#b7e8ea", "#97e0e3", "#74d7db", "#4fcbd0"
    ],
    "misty_rose": [
        "#fff7f8", "#ffeef0", "#ffdfe6", "#ffcfd9", "#ffbfc9", "#ffafba"
    ],
    "sage_mint": [
        "#f3f8f4", "#e0f1e7", "#cfe9da", "#bde0cc", "#a7d6bf", "#89c9a8"
    ],
    "dawn_petal": [
        "#fffaf6", "#fff0ea", "#ffe4dd", "#ffd6cf", "#ffc6bf", "#ffb5ad"
    ],
    "pearl_blue": [
        "#f6fbff", "#eaf6ff", "#d8eeff", "#c4e5ff", "#abcffb", "#88b7f2"
    ],
    "sand_blush": [
        "#fbf8f6", "#f7efe8", "#f1e6d9", "#e9d9c6", "#dfcbb2", "#d3baa0"
    ],
    "lavender_haze": [
        "#fbf8ff", "#f3efff", "#eadfff", "#decfff", "#cfbff6", "#bfaeea"
    ],
    "soft_sunset": [
        "#fff9f6", "#fff1ec", "#ffe6df", "#ffd9cf", "#ffc9bd", "#ffb6a7"
    ],
    "pale_cedar": [
        "#fbfcfb", "#eef7f2", "#e0f0e9", "#d1e9df", "#c0e0d3", "#aee6c7"
    ],
    "cool_mist": [
        "#fbfdfe", "#eef8fb", "#e0f1f6", "#d0e9ef", "#bfdfe6", "#a8d3db"
    ]
}

# 构建 LinearSegmentedColormap 对象并列出可选名称
CUSTOM_CMAPS = {
    name: LinearSegmentedColormap.from_list(name, cols, N=256)
    for name, cols in CUSTOM_CMAP_HEX.items()
}

CMAP_OPTIONS = list(CUSTOM_CMAPS.keys())


def get_cmap(name):
    """
    仅返回上面自定义的 colormap。name 必须是 CUSTOM_CMAPS 的键。
    """
    if isinstance(name, str) and name in CUSTOM_CMAPS:
        return CUSTOM_CMAPS[name]
    raise ValueError("cmap 必须为自定义配色之一。可选：" + ", ".join(CMAP_OPTIONS))

def plot_nz_with_arrows(nx, ny, nz,
                        cmap='viridis',
                        vmin=None, vmax=None,
                        out_base='normals_nz_arrows',
                        save=True, show=True,
                        arrow_step=None,
                        arrow_color='k',
                        arrow_width=0.003,
                        arrow_scale=1.0,
                        arrow_edge_color='black',
                        arrow_edge_width_factor=2.5):
    """
    仅绘制 nz（2D 图），并在上面绘制坡度方向箭头（由 nx, ny 归一化得到）。
    新增参数：
      - arrow_edge_color: 箭头描边颜色
      - arrow_edge_width_factor: 描边宽度相对于 arrow_width 的倍数（通常取 ~2.0-3.0）
    """
    # 参数检查与 colormap 获取
    cmap_obj = get_cmap(cmap) if isinstance(cmap, str) else get_cmap('viridis')

    # 确保数组为 float 并替换非法值
    nx = np.array(nx, dtype=np.float32)
    ny = np.array(ny, dtype=np.float32)
    nz = np.array(nz, dtype=np.float32)
    nx = np.nan_to_num(nx, nan=0.0, posinf=np.nanmax(nx[np.isfinite(nx)]) if np.any(np.isfinite(nx)) else 1.0,
                      neginf=np.nanmin(nx[np.isfinite(nx)]) if np.any(np.isfinite(nx)) else -1.0)
    ny = np.nan_to_num(ny, nan=0.0, posinf=np.nanmax(ny[np.isfinite(ny)]) if np.any(np.isfinite(ny)) else 1.0,
                      neginf=np.nanmin(ny[np.isfinite(ny)]) if np.any(np.isfinite(ny)) else -1.0)
    nz = np.nan_to_num(nz, nan=0.0, posinf=np.nanmax(nz[np.isfinite(nz)]) if np.any(np.isfinite(nz)) else 1.0,
                      neginf=np.nanmin(nz[np.isfinite(nz)]) if np.any(np.isfinite(nz)) else -1.0)

    H, W = nz.shape

    # 自动决定 vmin/vmax（常为对称中心 0，便于观察正负）
    if vmin is None or vmax is None:
        finite = nz[np.isfinite(nz)]
        if finite.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            # 使用 99.5 百分位裁剪极端值
            m = np.nanpercentile(np.abs(finite), 99.5)
            vmin, vmax = -m, m

    # 归一化 nx, ny（仅 XY 平面），避免除以 0
    mag = np.sqrt(nx**2 + ny**2)
    eps = 1e-8
    U = nx / (mag + eps)
    V = ny / (mag + eps)

    # 在图像坐标系上，imshow origin='upper' 会使 y 轴向下，因此 quiver 的 V 需取反以直观显示方向
    V_plot = -V

    # 决定箭头采样步长，默认尽量让箭头数量适中（长边约 30~60 个）
    if arrow_step is None:
        long_side = max(H, W)
        target = 40  # 目标栅格数
        arrow_step = max(1, long_side // target)

    # 采样用于绘箭头的网格
    ys = np.arange(0, H, arrow_step)
    xs = np.arange(0, W, arrow_step)
    X, Y = np.meshgrid(xs, ys)  # X: 列索引, Y: 行索引

    U_sub = U[Y, X]
    V_sub = V_plot[Y, X]

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(nz, origin='upper', cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation='nearest')

    # 去掉坐标轴、边框、网格
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 用 FancyArrowPatch 绘制带描边的箭头（替代 quiver）
    # arrow_scale 控制箭头长度，相对于采样步长 arrow_step
    arrow_base_len = max(1.0, arrow_step)  # 基准长度（数据坐标）
    arrow_len = arrow_base_len * 0.6 * arrow_scale

    # 将子采样数组摊平，逐个添加箭头补丁
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    U_flat = U_sub.ravel()
    V_flat = V_sub.ravel()

    # 将 arrow_width（原为 quiver width 的小值）映射到补丁的 linewidth
    lw = max(0.5, arrow_width * 200.0)  # 可调整映射比例以获得合适线宽
    edge_lw = lw * arrow_edge_width_factor

    for xi, yi, ui, vi in zip(X_flat, Y_flat, U_flat, V_flat):
        dx = ui * arrow_len
        dy = vi * arrow_len
        # 终点略微缩短以防箭头超出像素边界（可选）
        endx = xi + dx
        endy = yi + dy

        face_mutation = 10.0 * arrow_scale * 1.5
        arr_face = mpatches.FancyArrowPatch((xi, yi), (endx, endy),
                                           arrowstyle='-|>',
                                           mutation_scale=face_mutation,
                                           linewidth=6*max(0.1, lw*0.6),
                                           facecolor=arrow_color,
                                           edgecolor=arrow_color,
                                           zorder=2,
                                           alpha=1.0)
        ax.add_patch(arr_face)

    # 保存与显示（高分辨率，紧边距）
    if save:
        try:
            fig.savefig(f"{out_base}_nz.png", dpi=600, bbox_inches='tight', pad_inches=0)  # 期刊图常用高 DPI
            fig.savefig(f"{out_base}_nz.svg", bbox_inches='tight', pad_inches=0)
        except Exception:
            pass

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    # 配置：根据需要修改 dataset_path 与 env 列表
    dataset_path = 'data/terrain/train'
    env_list = ['env000009']

    env_path = osp.join(dataset_path, env_list[0])
    map_file = osp.join(env_path, 'map.p')

    if not osp.exists(map_file):
        raise FileNotFoundError(f"找不到 map 文件: {map_file}")

    with open(map_file, 'rb') as f:
        env_data = pickle.load(f)

    # 支持 map_tensor 存储格式 (H,W,B) 或直接包含键 'tensor'
    if isinstance(env_data, dict) and 'tensor' in env_data:
        map_tensor = env_data['tensor']
    else:
        map_tensor = env_data

    # 期望通道顺序: elevation, nx, ny, nz 或 H,W,4
    if map_tensor.ndim == 3 and map_tensor.shape[2] >= 4:
        elevation = map_tensor[:, :, 0]
        nx = map_tensor[:, :, 1]
        ny = map_tensor[:, :, 2]
        nz = map_tensor[:, :, 3]
    else:
        raise ValueError("map_tensor 形状不符合预期，期望 (H, W, >=4)")

    # 列出可选配色供挑选
    print("可选配色：", ", ".join(CMAP_OPTIONS))
    # 示例选择（期刊常用推荐：'viridis', 'cividis', 或中心为0时用 'blue_white_red' / 'RdBu' 等
    chosen_cmap = 'sage_mint'  # 改为 'viridis' 或 'cividis' 或 'RdBu' 等

    # 可调参数示例：arrow_step（采样步长), arrow_color, arrow_scale（控制箭头相对显示长度）
    plot_nz_with_arrows(nx, ny, nz,
                        cmap=chosen_cmap,
                        out_base='normals_nz_arrows',
                        save=True,
                        show=True,
                        arrow_step=8,      # None 时自动决定
                        arrow_color="#5a5732", # 箭头自定义颜色
                        arrow_width=0.006,
                        arrow_scale=1.00,
                        arrow_edge_width_factor=1.5)