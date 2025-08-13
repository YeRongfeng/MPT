import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import noise
import random
import math
import time
from scipy.ndimage import gaussian_filter, zoom
from scipy.interpolate import CubicSpline

# 参数
MAP_SIZE = 10.0          # 地图尺寸（米）
RESOLUTION = 0.02        # 网格分辨率（米），500x500点
MAX_HEIGHT = 1.5
MIN_HEIGHT = 0.0
# RNG_SEED = 42
RNG_SEED = time.time_ns() % (2**32)  # 使用当前时间的纳秒部分作为种子，确保每次运行不同
print(f"Using RNG seed: {RNG_SEED}")
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

num_pts = int(MAP_SIZE / RESOLUTION)
x = np.linspace(0, MAP_SIZE, num_pts)
y = np.linspace(0, MAP_SIZE, num_pts)
xx, yy = np.meshgrid(x, y, indexing='xy')

# 生成多尺度Perlin噪声（适配10m尺度）
def perlin_noise_grid(size, cycles, octaves=4, seed=0):
    grid = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            nx = i / (size - 1) * cycles
            ny = j / (size - 1) * cycles
            grid[i, j] = noise.pnoise2(nx + seed, ny + seed, octaves=octaves, persistence=0.5, lacunarity=2.0,
                                       repeatx=1024, repeaty=1024, base=0)
    mn, mx = grid.min(), grid.max()
    if mx - mn > 1e-9:
        grid = 2 * (grid - mn) / (mx - mn) - 1
    return grid

def upsample(field, target_shape):
    return zoom(field, (target_shape[0] / field.shape[0], target_shape[1] / field.shape[1]), order=3)

start = time.time()

# 多尺度噪声，基础起伏
global_cycles = random.uniform(0.7, 1.2)
meso_cycles = random.uniform(2.0, 3.5)
micro_cycles = random.uniform(2.0, 4.0)
low_res = 128

global_noise = perlin_noise_grid(low_res, global_cycles, octaves=4, seed=random.random() * 10)
meso_noise = perlin_noise_grid(low_res, meso_cycles, octaves=3, seed=random.random() * 10 + 20)
micro_noise = perlin_noise_grid(low_res, micro_cycles, octaves=2, seed=random.random() * 10 + 40)

global_up = upsample(global_noise, (num_pts, num_pts))
meso_up = upsample(meso_noise, (num_pts, num_pts))
micro_up = upsample(micro_noise, (num_pts, num_pts))

# 叠加噪声，保持细节但不要压缩归一化，保持较大振幅
base = 0.8 * global_up + 0.5 * meso_up + 0.15 * micro_up

# ---------------------------
# 这里加入一个宏观弧度起伏，模拟缓坡/大波动
# 以地图中心为原点，构造一个抛物面或者二维正弦曲面弧度
# 例如抛物面：z = A * ( (x - center_x)^2 + (y - center_y)^2 )
center_x, center_y = MAP_SIZE / 2, MAP_SIZE / 2
A = random.uniform(-0.05, 0.05)  # 控制弧度幅度和方向，正负决定凸起或凹陷

paraboloid = A * ((xx - center_x)**2 + (yy - center_y)**2)

# 或者用二维正弦波（替代抛物面，波动更自然）
# wave_amp = 0.1
# wave_freq = 2 * math.pi / MAP_SIZE  # 一个周期跨整个地图
# sinusoid = wave_amp * (np.sin(wave_freq * xx) + np.sin(wave_freq * yy))

# 将宏观弧度加到基础噪声中
base += paraboloid

# 轻微倾斜，增加自然感
tilt_angle = random.uniform(-math.pi / 6, math.pi / 6)
tilt = ((xx - MAP_SIZE / 2) * math.cos(tilt_angle) + (yy - MAP_SIZE / 2) * math.sin(tilt_angle)) / MAP_SIZE
base += 0.05 * tilt

# 归一化到0~1
base = (base - base.min()) / (base.max() - base.min())
heightmap = base * MAX_HEIGHT * 0.6  # 基础起伏占最大高的60%


def add_elliptical_peak(hmap, cx, cy, a, b, angle_deg, height, exponent=1.3):
    th = math.radians(angle_deg)
    xr = (xx - cx) * math.cos(th) + (yy - cy) * math.sin(th)
    yr = -(xx - cx) * math.sin(th) + (yy - cy) * math.cos(th)
    r2 = (xr / a) ** 2 + (yr / b) ** 2
    influence = np.exp(-r2)
    influence = influence ** exponent
    hmap += height * influence
    return hmap


def add_elliptical_valley(hmap, cx, cy, a, b, angle_deg, depth, exponent=1.0):
    th = math.radians(angle_deg)
    xr = (xx - cx) * math.cos(th) + (yy - cy) * math.sin(th)
    yr = -(xx - cx) * math.sin(th) + (yy - cy) * math.cos(th)
    r2 = (xr / a) ** 2 + (yr / b) ** 2
    influence = np.exp(-r2)
    influence = influence ** exponent
    hmap += depth * influence
    return hmap


def generate_smooth_ridge_path(length=10, num_points=40, max_curve=2.5):
    """
    生成带宏观弧度且连续平滑的山脊路径
    用多个控制点构建三次样条，避免断层和突变
    """
    xs = np.linspace(0, length, num_points)
    # 产生随机控制点的Y方向偏移，保持平滑
    ctrl_pts_x = np.linspace(0, length, 7)
    ctrl_pts_y = max_curve * np.sin(np.linspace(0, 2 * math.pi, 7) + random.uniform(0, 2*math.pi))
    # 加随机扰动增强自然感
    ctrl_pts_y += np.random.uniform(-0.3, 0.3, size=7)
    
    cs = CubicSpline(ctrl_pts_x, ctrl_pts_y, bc_type='natural')
    ys = cs(xs)
    
    path = np.vstack((xs, ys)).T
    
    # 随机旋转
    theta = random.uniform(0, 2 * math.pi)
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta),  math.cos(theta)]])
    path_rot = path @ rot
    
    # 平移到地图中心附近，保证路径完全在地图内
    min_xy = path_rot.min(axis=0)
    max_xy = path_rot.max(axis=0)
    shift = np.array([MAP_SIZE/2, MAP_SIZE/2]) - (min_xy + max_xy)/2
    shift += np.random.uniform(-MAP_SIZE/2, MAP_SIZE/2, size=2)  # 随机偏移增强自然感
    path_final = path_rot + shift
    
    # 裁剪，防止越界
    path_final = np.clip(path_final, 0, MAP_SIZE)
    return path_final

def add_continuous_ridge(hmap, path, width, height):
    """
    沿路径添加连续山脊，使用高斯核横向衰减，纵向用样条控制高度起伏，使得山脊连贯且自然
    """
    dist_along = np.zeros(len(path))
    for i in range(1, len(path)):
        dist_along[i] = dist_along[i-1] + np.linalg.norm(path[i]-path[i-1])
    total_len = dist_along[-1]
    norm_dist = dist_along / total_len
    
    # 纵向高度变化用多个波段叠加模拟山脊自然起伏
    long_profile = (np.sin(norm_dist * math.pi * 2 * random.uniform(1.5, 3)) * 0.3 + 
                    np.sin(norm_dist * math.pi * 4 * random.uniform(2,4)) * 0.15 + 1)
    long_profile /= long_profile.max()
    long_profile *= height
    
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    ridge_heights = np.zeros(points.shape[0], dtype=np.float32)
    
    # 逐点计算离路径距离，横向高斯权重和纵向高度加权
    # 为提速，分批处理
    batch_size = 50000
    for start_idx in range(0, points.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, points.shape[0])
        batch_points = points[start_idx:end_idx]
        # 计算批量点到所有路径点距离矩阵 shape=(batch, path_len)
        dists = np.linalg.norm(batch_points[:, None, :] - path[None, :, :], axis=2)
        min_idx = np.argmin(dists, axis=1)
        min_dist = dists[np.arange(len(batch_points)), min_idx]
        lat_w = np.exp(-(min_dist / width) ** 2)
        ridge_heights[start_idx:end_idx] = long_profile[min_idx] * lat_w

    ridge_layer = ridge_heights.reshape(xx.shape)
    hmap += ridge_layer
    return hmap


# 决定是否生成山脊，如果生成山脊则无其他主要特征
# generate_ridge = True  # 强制测试山脊，你可以改成 random.random() < 0.5
generate_ridge = random.random() < 0.5
# generate_ridge = False

if generate_ridge:
    ridge_path = generate_smooth_ridge_path(length=random.uniform(8.0, 10.0),
                                        num_points=200,
                                        max_curve=random.uniform(1.0, 2.5))
    ridge_width = random.uniform(0.7, 1.0)
    ridge_height = random.uniform(0.5, 1.4)
    heightmap = add_continuous_ridge(heightmap, ridge_path, ridge_width, ridge_height)

    heightmap = gaussian_filter(heightmap, sigma=1.2)
    peaks_num = 0
else:
    peaks_num = random.randint(1, 3)
# 主要地形特征 - 1~2个椭圆峰 + 0~1个椭圆谷
placed_centers = []
min_dist = 2.5  # 峰间距米

# peaks_num = random.randint(1, 3)
for _ in range(peaks_num):
    for _ in range(50):
        cx = random.uniform(1.5, MAP_SIZE - 1.5)
        cy = random.uniform(1.5, MAP_SIZE - 1.5)
        if all(math.hypot(cx - px, cy - py) > min_dist for (px, py) in placed_centers):
            break
    placed_centers.append((cx, cy))
    a = random.uniform(2.0, 3.0)
    b = random.uniform(1.6, 2.4)
    angle = random.uniform(0, 360)
    height = random.uniform(0.4, 1.3)
    heightmap = add_elliptical_peak(heightmap, cx, cy, a, b, angle, height, exponent=random.uniform(1.1, 1.5))

if random.random() < 0.5:
    for _ in range(50):
        cx = random.uniform(2.0, MAP_SIZE - 2.0)
        cy = random.uniform(2.0, MAP_SIZE - 2.0)
        if all(math.hypot(cx - px, cy - py) > min_dist for (px, py) in placed_centers):
            break
    placed_centers.append((cx, cy))
    a = random.uniform(2.5, 4.5)
    b = random.uniform(2.0, 3.6)
    angle = random.uniform(0, 360)
    depth = -random.uniform(0.15, 0.3)
    heightmap = add_elliptical_valley(heightmap, cx, cy, a, b, angle, depth)

# 轻微侵蚀/平滑，减少尖锐
heightmap = gaussian_filter(heightmap, sigma=1.0)

# 石子细节调整，减小振幅，改为更细微纹理
micro_detail_cycles = 20
micro_detail_noise = perlin_noise_grid(low_res, micro_detail_cycles, octaves=2, seed=random.random() * 100)
micro_detail = upsample(micro_detail_noise, (num_pts, num_pts))
heightmap += micro_detail * 0.01  # 振幅缩小3倍

# 再次归一化限制范围
heightmap = np.clip(heightmap, MIN_HEIGHT, MAX_HEIGHT)
heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min()) * (MAX_HEIGHT - MIN_HEIGHT) + MIN_HEIGHT

print(f"Terrain generated in {time.time() - start:.2f} seconds")
print(f"Generated ridge: {generate_ridge}")

# 保存高度图
plt.figure(figsize=(6, 6))
plt.imshow(heightmap.T, origin='lower', extent=[0, MAP_SIZE, 0, MAP_SIZE], cmap='terrain', interpolation='bilinear')
plt.colorbar(label='Height (m)')
plt.title('10m x 10m Terrain with Ridge' if generate_ridge else '10m x 10m Terrain')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.tight_layout()
plt.savefig("terrain_10m_10m.png", dpi=300)
plt.show()

# 创建点云并着色
points = np.stack([xx.ravel(), yy.ravel(), heightmap.ravel()], axis=-1).astype(np.float32)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

import matplotlib.cm as cm

norm_heights = (heightmap.ravel() - MIN_HEIGHT) / (MAX_HEIGHT - MIN_HEIGHT)
colors = cm.terrain(norm_heights)[:, :3]
pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))

# 保存点云
o3d.io.write_point_cloud("terrain_10m_10m.pcd", pcd)

# 可视化（点云）
o3d.visualization.draw_geometries([pcd], window_name="10m Terrain", width=900, height=700, point_show_normal=False)
