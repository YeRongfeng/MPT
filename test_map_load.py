import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# 添加兼容性处理
import sys
sys.modules['numpy._core'] = np
sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
sys.modules['numpy._core.multiarray'] = np.core.multiarray

if __name__ == '__main__':
    # 加载地图数据
    with open('/home/yrf/MPT/data/desert/map.p', 'rb') as f:
        map_data = pickle.load(f)
    
    tensor = map_data['tensor']  # [H, W, 4]
    elevation = tensor[:, :, 0]
    normal_x = tensor[:, :, 1]
    normal_y = tensor[:, :, 2]
    normal_z = tensor[:, :, 3]
    
    # 可视化
    # 创建坐标网格
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制高程热力图
    elevation_masked = np.ma.masked_invalid(elevation)
    
    im = ax.imshow(elevation_masked, extent=[-5, 5, -5, 5],
                    origin='lower', cmap='terrain', aspect='equal')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Elevation (m)', fontsize=12)
    
    # 绘制法向量箭头（只显示部分，避免过于密集）
    step = max(1, min(100, 100) // 20)  # 自适应步长
    
    for i in range(0, 100, step):
        for j in range(0, 100, step):
            if not (np.isnan(elevation[i, j]) or
                    np.isnan(normal_x[i, j]) or
                    np.isnan(normal_y[i, j])):
                
                # 栅格中心坐标
                cx = x[j]
                cy = y[i]
                
                # 法向量的XY分量合成向量对应的cos和sin
                nx = normal_x[i, j]
                ny = normal_y[i, j]
                
                # 箭头长度（根据地图尺寸自适应）
                arrow_scale = min(10, 10) / 30
                
                # 绘制箭头
                ax.arrow(cx, cy, nx * arrow_scale, ny * arrow_scale,
                        head_width=arrow_scale*0.3, head_length=arrow_scale*0.2,
                        fc='red', ec='red', alpha=0.7, linewidth=1)
    
    # 设置网格
    ax.grid(True, alpha=0.3)
    
    plt.show()
