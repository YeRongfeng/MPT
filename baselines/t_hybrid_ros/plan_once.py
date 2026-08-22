#!/usr/bin/env python3
"""One-shot ROS client for official T-Hybrid A*.

Publishes OccupancyGrid + start/goal, waits for /sPath (fallback /path).
Must run with the Noetic Python that has rospy, not the project .venv.
"""
from __future__ import print_function

import argparse
import math
import sys
import time

import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from tf.transformations import quaternion_from_euler


def _yaw_to_quat(yaw):
    return quaternion_from_euler(0.0, 0.0, float(yaw))


def _load_occupancy(path):
    with open(path) as handle:
        parts = handle.read().split()
    width = int(parts[0])
    height = int(parts[1])
    values = [100 if int(v) else 0 for v in parts[2 : 2 + width * height]]
    if len(values) != width * height:
        raise ValueError("occupancy file does not match width*height")
    return width, height, values


def _pose(x, y, yaw):
    qx, qy, qz, qw = _yaw_to_quat(yaw)
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = 0.0
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("occupancy")
    parser.add_argument("sx", type=float)
    parser.add_argument("sy", type=float)
    parser.add_argument("st", type=float)
    parser.add_argument("gx", type=float)
    parser.add_argument("gy", type=float)
    parser.add_argument("gt", type=float)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    width, height, data = _load_occupancy(args.occupancy)
    rospy.init_node("t_hybrid_plan_once", anonymous=True)

    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.header.stamp = rospy.Time.now()
    grid.info.resolution = float(args.resolution)
    grid.info.width = width
    grid.info.height = height
    grid.info.origin.orientation.w = 1.0
    grid.data = data

    pub_map = rospy.Publisher("/map", OccupancyGrid, queue_size=1, latch=True)
    pub_start = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
    pub_goal = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)

    got = {"path": None}

    def _on_path(msg):
        if msg.poses:
            got["path"] = msg

    rospy.Subscriber("/sPath", Path, _on_path, queue_size=1)
    rospy.Subscriber("/path", Path, _on_path, queue_size=1)
    rospy.sleep(0.5)

    t0 = rospy.Time.now()
    pub_map.publish(grid)
    rospy.sleep(0.3)

    start = PoseWithCovarianceStamped()
    start.header = _pose(args.sx, args.sy, args.st).header
    start.pose.pose = _pose(args.sx, args.sy, args.st).pose
    goal = _pose(args.gx, args.gy, args.gt)
    pub_start.publish(start)
    rospy.sleep(0.2)
    pub_goal.publish(goal)

    deadline = time.time() + float(args.timeout)
    while time.time() < deadline and not rospy.is_shutdown():
        path = got["path"]
        if path is not None and path.header.stamp >= t0 and len(path.poses) >= 2:
            print("found %d" % len(path.poses))
            for pose in path.poses:
                yaw = 2.0 * math.atan2(pose.pose.orientation.z, pose.pose.orientation.w)
                print("%.6f %.6f %.6f" % (pose.pose.position.x, pose.pose.position.y, yaw))
            return 0
        rospy.sleep(0.05)

    print("found 0")
    print("timeout waiting for /sPath or /path", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
