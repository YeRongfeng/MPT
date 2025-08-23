"""
terrain_visualize.py - 绘制3d地形图的脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def plot_terrain(terrain_data, title="3D Terrain Visualization"):
    """
    绘制3D地形图
    :param terrain_data: 2D numpy array, 每个元素表示地形高度
    :param title: 图表标题
    """
    # 创建网格数据
    x = np.linspace(0, terrain_data.shape[1]-1, terrain_data.shape[1])
    y = np.linspace(0, terrain_data.shape[0]-1, terrain_data.shape[0])
    x, y = np.meshgrid(x, y)
    z = terrain_data

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面
    surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # 关闭坐标系
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # 关闭网格线
    ax.grid(False)

    plt.show()
    
if __name__ == "__main__":
    # 示例地形数据
    terrain_data = np.random.rand(100, 100) * 100  # 生成随机地形高度数据
    plot_terrain(terrain_data, title="Sample 3D Terrain")