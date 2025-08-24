import matplotlib.pyplot as plt
import os
from os import path as osp
import numpy as np
import pickle

import sys
sys.modules['numpy._core'] = np
sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
sys.modules['numpy._core.multiarray'] = np.core.multiarray

from dataLoader_uneven import UnevenPathDataLoader

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# added imports for 3D mesh rendering
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.colors import Normalize

if __name__ == "__main__":
    envlist = ['env000009']
    dataset_path = 'data/terrain/train'
    dataset = UnevenPathDataLoader(envlist, dataset_path)

    esdf = dataset[0]['cost_map']

    # 绘制 3D ESDF（替换 TODO）
    # ensure numpy array
    vals = np.asarray(esdf, dtype=float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

    # 数据维度为 (x, y, z)
    nx, ny, nz = vals.shape  # (x, y, z)

    # 选择若干等值面，避开极端噪声（使用 5-7 层）
    vmin = float(np.percentile(vals, 2))
    vmax = float(np.percentile(vals, 98))
    if not (vmax > vmin):
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
    num_levels = 6
    levels = np.linspace(vmin + 1e-6, vmax - 1e-6, num=num_levels)

    # 选用丰富而论文友好的配色（turbo / magma / viridis 可选）
    cmap = cm.get_cmap('turbo')
    norm = Normalize(vmin=vmin, vmax=vmax)

    # 准备画布：白底、无坐标、紧凑无边距
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 渲染多层等值面，靠近中心的面更不透明以增强层次感
    # marching_cubes 返回的 verts 与输入数组轴顺序一致（这里输入为 (x,y,z)），因此无需翻转轴
    center = np.array([nx, ny, nz], dtype=float) / 2.0

    inner_idx = max(0, num_levels - 1)

    # 目标统一长度（取最长维度），并计算每轴缩放因子以使三个轴的总长度相同
    L = float(max(nx, ny, nz))
    sx = L / nx if nx > 0 else 1.0
    sy = L / ny if ny > 0 else 1.0
    sz = L / nz if nz > 0 else 1.0
    scales = np.array([sx, sy, sz], dtype=float)  # 对应 (x, y, z) 缩放因子

    inner_mesh = None
    for i, level in enumerate(levels):
        try:
            verts, faces, normals, values_mc = measure.marching_cubes(vals, level=level)
        except Exception:
            # 如果某层无法生成网格（例如过小），跳过
            continue

        # 顶点为 (x, y, z)（与 vals.shape 对应），先居中再按轴缩放到统一总长度 L
        verts_centered = verts - center
        verts_plot = verts_centered * scales

        # 翻转 y 轴：将 y 分量取反以实现竖直方向的翻转
        verts_plot[:, 1] *= -1

        mesh = Poly3DCollection(verts_plot[faces], linewidths=0)
        c = np.array(cmap(norm(level)))  # RGBA
        if i == inner_idx:
            # 构建最内层但先不添加，最后统一绘制以保证覆盖
            solid_color = (c[:3] * 0.85).tolist() + [1.0]
            mesh.set_facecolor(solid_color)
            mesh.set_edgecolor('none')
            mesh.set_alpha(1.0)
            inner_mesh = mesh
            continue

        # 其他层保持半透明分层效果
        alpha = 0.12 + 0.7 * (i / max(1, num_levels - 1))
        mesh.set_facecolor(c[:3].tolist() + [alpha])
        mesh.set_edgecolor('none')
        ax.add_collection3d(mesh)

    # 最后把最内层（若存在）添加到图上以确保为实心覆盖
    if inner_mesh is not None:
        ax.add_collection3d(inner_mesh)

    # 缩放后各轴总长度均为 L，设置对称范围以居中显示
    half = L / 2.0
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    try:
        # 强制等尺度显示（modern matplotlib）
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        # 旧版 matplotlib 可能没有 set_box_aspect，忽略
        pass

    # 去除坐标轴、刻度、边框（简约论文风）
    ax.axis('off')
    ax.view_init(elev=20, azim=-50)
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 保存高分辨率 PNG
    out_path = osp.join(os.getcwd(), "esdf_3d_multilevel.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print("Saved", out_path)

    # 可视化（可注释以在无交互环境直接保存）
    # plt.show()
