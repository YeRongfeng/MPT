"""
测试脚本：生成初始轨迹控制点并可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from train_fine import generate_initial_trajectory_control_points
from bspline_utils import DifferentiableBSpline


def visualize_initial_trajectories(num_samples=4, map_bounds=(-20, 20)):
    """
    Generate and visualize initial trajectories
    
    Args:
        num_samples: Number of trajectories to generate
        map_bounds: Map range (min, max)
    """
    device = 'cpu'
    
    # Generate random start and goal poses
    np.random.seed(42)
    torch.manual_seed(42)
    
    start_poses = []
    goal_poses = []
    
    # Scenario 1: Bottom-left to top-right
    start_poses.append(torch.tensor([-15.0, -15.0, 0.0]))
    goal_poses.append(torch.tensor([15.0, 15.0, np.pi*3/2]))
    
    # Scenario 2: Top-right to bottom-left
    start_poses.append(torch.tensor([15.0, 15.0, np.pi]))
    goal_poses.append(torch.tensor([-15.0, -15.0, 0.0]))
    
    # Scenario 3: Corner case - departing from corner
    start_poses.append(torch.tensor([-16.0, -13.0, -np.pi/3]))
    goal_poses.append(torch.tensor([18.0, 18.0, np.pi/2]))
    
    # Scenario 4: Corner case - heading outside map bounds
    start_poses.append(torch.tensor([20.0, 0.0, 0.0]))
    goal_poses.append(torch.tensor([-15.0, 15.0, np.pi/2]))
    
    # Prepare figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # B-spline reconstruction layer
    bspline_layer = DifferentiableBSpline(
        num_control_points=26,
        num_output_points=100,
        degree=3
    ).to(device)
    
    for idx, (start_pose, goal_pose, ax) in enumerate(zip(
        start_poses[:num_samples], goal_poses[:num_samples], axes[:num_samples]
    )):
        # Generate control points
        full_cp, middle_cp = generate_initial_trajectory_control_points(
            start_pose, goal_pose, num_middle_points=24, device=device, map_bounds=map_bounds
        )
        
        # Reconstruct trajectory
        reconstructed_traj = bspline_layer(full_cp.unsqueeze(0))  # (1, 100, 2)
        reconstructed_traj = reconstructed_traj.squeeze(0).detach().numpy()  # (100, 2)
        
        # Get control points (numpy)
        full_cp_np = full_cp.detach().numpy()  # (26, 2)
        
        # Extract start and goal information
        start_xy = start_pose[:2].numpy()
        goal_xy = goal_pose[:2].numpy()
        start_yaw = start_pose[2].item()
        goal_yaw = goal_pose[2].item()
        
        # Plot
        ax.set_xlim(map_bounds[0] - 2, map_bounds[1] + 2)
        ax.set_ylim(map_bounds[0] - 2, map_bounds[1] + 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw map boundary
        map_min, map_max = map_bounds
        rect = plt.Rectangle((map_min, map_min), map_max - map_min, map_max - map_min,
                             fill=False, edgecolor='red', linewidth=2, linestyle='--', label='Map Boundary')
        ax.add_patch(rect)
        
        # Draw reconstructed trajectory
        ax.plot(reconstructed_traj[:, 0], reconstructed_traj[:, 1], 'b-', linewidth=2, label='Reconstructed Trajectory')
        
        # Draw all control points
        ax.plot(full_cp_np[:, 0], full_cp_np[:, 1], 'ko', markersize=4, alpha=0.5)
        
        # Highlight start, 2nd, penultimate, and goal control points
        ax.plot(*full_cp_np[0], 'gs', markersize=10, label='Start (Fixed)', zorder=5)
        ax.plot(*full_cp_np[1], 'b^', markersize=10, label='2nd Point', zorder=5)
        ax.plot(*full_cp_np[-2], 'r^', markersize=10, label='Penultimate Point', zorder=5)
        ax.plot(*full_cp_np[-1], 'rs', markersize=10, label='Goal (Fixed)', zorder=5)
        
        # Draw heading arrows at start and goal
        arrow_length = 2.0
        arrow_width = 0.3
        
        # Arrow at start position
        arrow_start = FancyArrowPatch(
            start_xy, 
            start_xy + arrow_length * np.array([np.cos(start_yaw), np.sin(start_yaw)]),
            arrowstyle='->', mutation_scale=25, color='green', linewidth=2.5, zorder=6
        )
        ax.add_patch(arrow_start)
        
        # Arrow at goal position
        arrow_goal = FancyArrowPatch(
            goal_xy,
            goal_xy + arrow_length * np.array([np.cos(goal_yaw), np.sin(goal_yaw)]),
            arrowstyle='->', mutation_scale=25, color='red', linewidth=2.5, zorder=6
        )
        ax.add_patch(arrow_goal)
        
        # Set title and labels
        ax.set_title(f'Scenario {idx + 1}: Initial Trajectory\nStart({start_xy[0]:.1f}, {start_xy[1]:.1f}, {start_yaw:.2f}rad) -> Goal({goal_xy[0]:.1f}, {goal_xy[1]:.1f}, {goal_yaw:.2f}rad)', 
                    fontsize=10)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        
        # Print control point coordinates
        print(f"\nScenario {idx + 1}:")
        print(f"  Start: {full_cp_np[0]} (Fixed)")
        print(f"  2nd Point: {full_cp_np[1]} (Along start heading {start_yaw:.2f}rad)")
        print(f"  Penultimate Point: {full_cp_np[-2]} (Along goal heading {goal_yaw:.2f}rad)")
        print(f"  Goal: {full_cp_np[-1]} (Fixed)")
        
        # Check if all points are within bounds
        min_coord = full_cp_np.min()
        max_coord = full_cp_np.max()
        in_bounds = (min_coord >= map_bounds[0]) and (max_coord <= map_bounds[1])
        print(f"  All points within [{map_bounds[0]}, {map_bounds[1]}]: {in_bounds}")
        print(f"  Coordinate range: [{min_coord:.2f}, {max_coord:.2f}]")
    
    plt.tight_layout()
    plt.savefig('/home/sdu/MPT/initial_trajectories_visualization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: /home/sdu/MPT/initial_trajectories_visualization.png")
    plt.show()


def test_batch_generation():
    """
    Test batch generation of initial trajectories
    """
    print("\n" + "="*60)
    print("Batch Generation Test")
    print("="*60)
    
    device = 'cpu'
    B = 5  # batch size
    
    # Generate random start and goal poses
    start_poses = torch.randn(B, 3)
    start_poses[:, :2] = start_poses[:, :2] * 15  # Coordinate range [-15, 15]
    start_poses[:, 2] = torch.rand(B) * 2 * np.pi - np.pi  # yaw range [-pi, pi]
    
    goal_poses = torch.randn(B, 3)
    goal_poses[:, :2] = goal_poses[:, :2] * 15
    goal_poses[:, 2] = torch.rand(B) * 2 * np.pi - np.pi
    
    # Generate control points
    full_cp, middle_cp = generate_initial_trajectory_control_points(
        start_poses, goal_poses, num_middle_points=24, device=device
    )
    
    print(f"Input shapes:")
    print(f"  start_poses: {start_poses.shape}")
    print(f"  goal_poses: {goal_poses.shape}")
    print(f"\nOutput shapes:")
    print(f"  full_cp: {full_cp.shape}")
    print(f"  middle_cp: {middle_cp.shape}")
    
    # Check if all control points are within bounds
    map_min, map_max = -20, 20
    in_bounds = (full_cp >= map_min).all() and (full_cp <= map_max).all()
    print(f"\nAll control points within [{map_min}, {map_max}]: {in_bounds}")
    
    if not in_bounds:
        out_of_bounds_idx = (full_cp < map_min) | (full_cp > map_max)
        print(f"Warning: Found points outside bounds!")
        print(f"  Number of out-of-bounds points: {out_of_bounds_idx.sum().item()}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Global min coordinate: {full_cp.min().item():.2f}")
    print(f"  Global max coordinate: {full_cp.max().item():.2f}")
    print(f"  Average coordinate: {full_cp.mean().item():.2f}")


if __name__ == '__main__':
    print("="*60)
    print("Initial Trajectory Generation Test Script")
    print("="*60)
    
    # Test 1: Visualize different scenarios
    visualize_initial_trajectories(num_samples=4)
    
    # Test 2: Batch generation test
    test_batch_generation()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
