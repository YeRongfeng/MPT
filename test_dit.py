# import torch
# from dit.Models import PathDiffusionTransformer
# import numpy as np

# # 加载模型
# model_args = dict(
#     n_layers=6,
#     n_heads=8,
#     d_k=192,
#     d_v=96,
#     d_model=512,
#     d_inner=2048,  # 扩散模型需要更大的MLP
#     pad_idx=None,
#     n_position=15*15,
#     dropout=0.1,
#     train_shape=[12, 12],
#     n_path_steps=20,  # 起点 + 10个中间点
#     diffusion_steps=1000
# )
# model = PathDiffusionTransformer(**model_args).cuda()
# model.load_state_dict(torch.load('/home/yrf/MPT/data/sim/best_model.pth')['model_state_dict'])
# model.eval()

# # 生成路径
# test_map = torch.randn(1, 6, 100, 100).cuda()
# paths = model.sample(test_map, num_samples=10)

# print("归一化输出范围检查:")
# print(f"  X 范围: [{paths[:, :, 0].min():.4f}, {paths[:, :, 0].max():.4f}]")
# print(f"  Y 范围: [{paths[:, :, 1].min():.4f}, {paths[:, :, 1].max():.4f}]")
# print(f"  Yaw 范围: [{paths[:, :, 2].min():.4f}, {paths[:, :, 2].max():.4f}]")
# print(f"  期望: 所有值在 [-1.0, 1.0] 范围内")

# # 反归一化
# paths[:, :, :2] = paths[:, :, :2] * 20.0
# paths[:, :, 2] = paths[:, :, 2] * np.pi

# print("\n反归一化后范围检查:")
# print(f"  X 范围: [{paths[:, :, 0].min():.2f}, {paths[:, :, 0].max():.2f}]")
# print(f"  Y 范围: [{paths[:, :, 1].min():.2f}, {paths[:, :, 1].max():.2f}]")
# print(f"  Yaw 范围: [{paths[:, :, 2].min():.2f}, {paths[:, :, 2].max():.2f}]")
# print(f"  期望: X,Y ∈ [-20, 20], Yaw ∈ [-π, π]")


import torch
import numpy as np
import pickle
import os
from glob import glob

# --- 配置 ---
dataFolder = "/home/yrf/MPT/data/sim_dataset/train"  # 修改为你的路径
env_list = ["env000004", "env000005"]

# 检查路径是否存在
if not os.path.exists(dataFolder):
    print(f"错误：数据文件夹不存在于路径: {dataFolder}")
    exit()

# --- 统计逻辑 ---
x_min, x_max = float('inf'), float('-inf')
y_min, y_max = float('inf'), float('-inf')
yaw_min, yaw_max = float('inf'), float('-inf')

print("📢 开始直接读取 .p 文件并统计数据...")

total_files = 0
for env_name in env_list:
    env_path = os.path.join(dataFolder, env_name)
    if not os.path.exists(env_path):
        print(f"警告：环境路径不存在: {env_path}")
        continue
    
    # 获取该环境下所有的 .p 文件
    p_files = glob(os.path.join(env_path, "*.p"))
    print(f"环境 {env_name}: 找到 {len(p_files)} 个 .p 文件")
    
    for i, p_file in enumerate(p_files):
        # 文件名不能为map.p
        if os.path.basename(p_file) == "map.p":
            continue
        try:
            # 直接读取 .p 文件
            with open(p_file, 'rb') as f:
                data = pickle.load(f)
            
            # 假设数据结构中包含轨迹信息，根据实际数据结构调整
            # 常见的可能是 data['trajectory'] 或 data['path'] 等
            if 'trajectory' in data:
                traj = data['trajectory']  # 形状应该是 (n_points, 3)
            elif 'path' in data:
                traj = data['path']
            else:
                # 如果不确定结构，可以打印查看
                print(f"文件 {p_file} 的数据结构: {data.keys()}")
                continue
            
            # 转换为 numpy 数组（如果不是的话）
            if isinstance(traj, list):
                traj = np.array(traj)
            
            traj[:, 2] = (traj[:, 2] + np.pi) % (2 * np.pi) - np.pi
            
            # 更新统计信息
            x_min = min(x_min, traj[:, 0].min())
            x_max = max(x_max, traj[:, 0].max())
            y_min = min(y_min, traj[:, 1].min())
            y_max = max(y_max, traj[:, 1].max())
            yaw_min = min(yaw_min, traj[:, 2].min())
            yaw_max = max(yaw_max, traj[:, 2].max())
            
            total_files += 1
            
        except Exception as e:
            print(f"读取文件 {p_file} 时出错: {e}")
            continue

print(f"✅ 数据统计完成。共处理 {total_files} 个文件。")

# --- 结果输出 ---
print("\n--- 训练数据原始范围统计 ---")
print(f"  X 范围: [{x_min:.2f}, {x_max:.2f}]")
print(f"  Y 范围: [{y_min:.2f}, {y_max:.2f}]")
print(f"  Yaw 范围: [{yaw_min:.2f}, {yaw_max:.2f}]")

# 归一化后的范围
x_norm_min = x_min / 20.0
x_norm_max = x_max / 20.0
y_norm_min = y_min / 20.0
y_norm_max = y_max / 20.0
yaw_norm_min = yaw_min / np.pi
yaw_norm_max = yaw_max / np.pi

print("\n--- 归一化后统计 (除以 20.0 和 π) ---")
print(f"  X_norm 范围: [{x_norm_min:.4f}, {x_norm_max:.4f}]")
print(f"  Y_norm 范围: [{y_norm_min:.4f}, {y_norm_max:.4f}]")
print(f"  Yaw_norm 范围: [{yaw_norm_min:.4f}, {yaw_norm_max:.4f}]")
print("  --- 提示 ---")
print("  1. 如果 X/Y 范围接近 ±20.0，表示轨迹可能覆盖了整个地图边界。")
print("  2. 如果 Yaw 范围接近 ±1.0，表示角度覆盖了 [-π, π] 的完整范围。")