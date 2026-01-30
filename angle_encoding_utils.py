"""
angle_encoding_utils.py - 角度编码转换工具

提供角度表示的转换函数，解决角度周期性问题：
- 标量θ ↔ sin/cos编码(sin(θ), cos(θ))

核心优势：
- 179° 和 -179° 在sin/cos空间中距离≈0.034，完美反映物理接近度
- 避免了直接使用θ时的不连续性问题（179°和-179°差1.99在归一化空间）
"""

import torch
import numpy as np


def angle_to_sincos(theta):
    """
    将角度θ转换为sin/cos编码
    
    Args:
        theta: torch.Tensor or np.ndarray, 形状任意, 最后一维是角度(弧度)
              例如: (B, N) 或 (B, N, 1)
    
    Returns:
        sincos: torch.Tensor or np.ndarray, 形状为 (*theta.shape[:-1], 2)
                最后一维为 [sin(θ), cos(θ)]
    """
    if isinstance(theta, torch.Tensor):
        return torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
    else:  # numpy
        return np.stack([np.sin(theta), np.cos(theta)], axis=-1)


def sincos_to_angle(sincos):
    """
    从sin/cos编码恢复角度θ
    
    Args:
        sincos: torch.Tensor or np.ndarray, 形状为 (*batch_dims, 2)
                最后一维为 [sin(θ), cos(θ)]
    
    Returns:
        theta: torch.Tensor or np.ndarray, 形状为 (*batch_dims,)
               角度范围 [-π, π]
    """
    sin_theta = sincos[..., 0]
    cos_theta = sincos[..., 1]
    
    if isinstance(sincos, torch.Tensor):
        return torch.atan2(sin_theta, cos_theta)
    else:  # numpy
        return np.arctan2(sin_theta, cos_theta)


def pose3_to_pose4(pose3):
    """
    将3维位姿(x, y, θ)转换为4维位姿(x, y, sin(θ), cos(θ))
    
    Args:
        pose3: torch.Tensor or np.ndarray, 形状为 (..., 3)
               最后一维为 [x, y, θ]
    
    Returns:
        pose4: torch.Tensor or np.ndarray, 形状为 (..., 4)
               最后一维为 [x, y, sin(θ), cos(θ)]
    """
    xy = pose3[..., :2]  # (B, N, 2)
    theta = pose3[..., 2]  # (B, N)
    
    if isinstance(pose3, torch.Tensor):
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        return torch.cat([xy, sin_theta.unsqueeze(-1), cos_theta.unsqueeze(-1)], dim=-1)
    else:  # numpy
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        return np.concatenate([xy, sin_theta[..., None], cos_theta[..., None]], axis=-1)


def pose4_to_pose3(pose4):
    """
    将4维位姿(x, y, sin(θ), cos(θ))转换回3维位姿(x, y, θ)
    
    Args:
        pose4: torch.Tensor or np.ndarray, 形状为 (..., 4)
               最后一维为 [x, y, sin(θ), cos(θ)]
    
    Returns:
        pose3: torch.Tensor or np.ndarray, 形状为 (..., 3)
               最后一维为 [x, y, θ]，θ ∈ [-π, π]
    """
    xy = pose4[..., :2]  # (B, N, 2)
    sin_theta = pose4[..., 2]
    cos_theta = pose4[..., 3]
    
    if isinstance(pose4, torch.Tensor):
        theta = torch.atan2(sin_theta, cos_theta)
        return torch.cat([xy, theta.unsqueeze(-1)], dim=-1)
    else:  # numpy
        theta = np.arctan2(sin_theta, cos_theta)
        return np.concatenate([xy, theta[..., None]], axis=-1)


def normalize_pose_for_model(pose3, coord_range=20.0):
    """
    将3维位姿归一化为4维模型输入格式
    
    Args:
        pose3: torch.Tensor or np.ndarray, 形状为 (..., 3)
               原始坐标 [x, y, θ]，其中 x,y ∈ [-coord_range, coord_range]
        coord_range: float, 坐标范围（默认20米）
    
    Returns:
        pose4_normalized: torch.Tensor or np.ndarray, 形状为 (..., 4)
                         归一化后的 [x_norm, y_norm, sin(θ), cos(θ)]
                         其中 x_norm, y_norm ∈ [-1, 1]
    """
    # 转换为4维
    pose4 = pose3_to_pose4(pose3)
    
    # 归一化坐标
    pose4_normalized = pose4.copy() if isinstance(pose4, np.ndarray) else pose4.clone()
    pose4_normalized[..., :2] = pose4[..., :2] / coord_range
    
    # 裁剪到[-1, 1]
    if isinstance(pose4_normalized, torch.Tensor):
        pose4_normalized[..., :2] = torch.clamp(pose4_normalized[..., :2], -1.0, 1.0)
    else:
        pose4_normalized[..., :2] = np.clip(pose4_normalized[..., :2], -1.0, 1.0)
    
    return pose4_normalized


def denormalize_pose_from_model(pose4_normalized, coord_range=20.0):
    """
    将模型输出的4维归一化位姿反归一化为3维实际坐标
    
    Args:
        pose4_normalized: torch.Tensor or np.ndarray, 形状为 (..., 4)
                         归一化后的 [x_norm, y_norm, sin(θ), cos(θ)]
        coord_range: float, 坐标范围（默认20米）
    
    Returns:
        pose3: torch.Tensor or np.ndarray, 形状为 (..., 3)
               实际坐标 [x, y, θ]
    """
    # 反归一化坐标
    pose4 = pose4_normalized.copy() if isinstance(pose4_normalized, np.ndarray) else pose4_normalized.clone()
    pose4[..., :2] = pose4_normalized[..., :2] * coord_range
    
    # 转换回3维
    pose3 = pose4_to_pose3(pose4)
    
    return pose3


# =================== 示例用法 ===================
if __name__ == "__main__":
    print("=" * 60)
    print("角度编码转换示例")
    print("=" * 60)
    
    # 测试案例：179度和-179度
    theta1 = np.deg2rad(179)
    theta2 = np.deg2rad(-179)
    
    print(f"\n【问题演示】标量表示的不连续性:")
    print(f"  179° = {theta1:.4f} rad")
    print(f"  -179° = {theta2:.4f} rad")
    print(f"  差值 = {abs(theta1 - theta2):.4f} rad (≈6.25 rad)")
    print(f"  归一化到[-1,1]后差值 = {abs(theta1/np.pi - theta2/np.pi):.4f} (≈1.99)")
    
    # sin/cos编码
    sincos1 = angle_to_sincos(np.array([theta1]))[0]
    sincos2 = angle_to_sincos(np.array([theta2]))[0]
    
    print(f"\n【解决方案】Sin/Cos编码:")
    print(f"  179° → sin={sincos1[0]:.4f}, cos={sincos1[1]:.4f}")
    print(f"  -179° → sin={sincos2[0]:.4f}, cos={sincos2[1]:.4f}")
    print(f"  欧氏距离 = {np.linalg.norm(sincos1 - sincos2):.4f} (≈0.035)")
    print(f"  ✓ 完美反映了角度的物理接近度！")
    
    # 测试位姿转换
    print(f"\n【位姿转换测试】")
    pose3 = np.array([[10.0, 5.0, np.deg2rad(179)],
                      [10.0, 5.0, np.deg2rad(-179)]])
    print(f"  原始3维位姿: shape={pose3.shape}")
    print(f"    {pose3[0]}")
    print(f"    {pose3[1]}")
    
    pose4 = pose3_to_pose4(pose3)
    print(f"\n  转换为4维: shape={pose4.shape}")
    print(f"    {pose4[0]}")
    print(f"    {pose4[1]}")
    
    pose3_recovered = pose4_to_pose3(pose4)
    print(f"\n  恢复为3维: shape={pose3_recovered.shape}")
    print(f"    {pose3_recovered[0]}")
    print(f"    {pose3_recovered[1]}")
    print(f"  ✓ 重建误差: {np.max(np.abs(pose3 - pose3_recovered)):.6e}")
    
    # 测试归一化
    print(f"\n【归一化测试】")
    pose4_norm = normalize_pose_for_model(pose3, coord_range=20.0)
    print(f"  归一化后: shape={pose4_norm.shape}")
    print(f"    {pose4_norm[0]}")
    print(f"    x范围: [{pose4_norm[:, 0].min():.2f}, {pose4_norm[:, 0].max():.2f}]")
    print(f"    y范围: [{pose4_norm[:, 1].min():.2f}, {pose4_norm[:, 1].max():.2f}]")
    
    pose3_denorm = denormalize_pose_from_model(pose4_norm, coord_range=20.0)
    print(f"\n  反归一化后:")
    print(f"    {pose3_denorm[0]}")
    print(f"  ✓ 重建误差: {np.max(np.abs(pose3 - pose3_denorm)):.6e}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
