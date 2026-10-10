"""Generic comparison: KISS-ICP on the 3D scans, with no wall or IMU knowledge.

KISS-ICP is an existing open source lidar odometry (pip install kiss-icp).
I run it on the same data to see how my wall methods do next to a general
purpose one. It gets the 3D scans as they are and nothing else: no walls,
no gyro, no delay handling. This file is only a thin wrapper around it.

It is called kiss.py and not kiss_icp.py so it does not clash with the
library's own name.
"""
import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP

from lidar_odom.preprocess import SELF_RADIUS

MAX_RANGE = 20.0   # m, the room is about 8 x 15 so this drops the far floor returns
VOXEL = 0.2        # m, KISS-ICP's own rule of thumb is max range / 100


def run(data):
    """Return one pose (x, y, yaw) of base_link per scan, starting at zero."""
    T, start = data["T_base_lidar"], data["scan_start"]
    config = KISSConfig()
    # deskewing needs a time for every point and this bag has none
    config.data.deskew = False
    config.data.max_range = MAX_RANGE
    config.mapping.voxel_size = VOXEL
    odometry = KissICP(config)

    poses = np.zeros((len(start) - 1, 3))
    for i in range(len(poses)):
        xyz = data["scan_xyz"][start[i]:start[i + 1]]
        # take the robot's own body out, same cut as in preprocessing.
        # Otherwise it would be matched as if it were part of the room.
        xyz = xyz[np.hypot(xyz[:, 0], xyz[:, 1]) > SELF_RADIUS]
        # second argument is the per point timestamps, empty because there
        # are none
        odometry.register_frame(xyz.astype(np.float64), np.empty(0))
        # last_pose is where the LIDAR is relative to where the lidar
        # started. I want base_link relative to where base_link started,
        # so: robot to lidar, the lidar's motion, lidar back to robot.
        # Without this every turn would show up as fake sideways movement,
        # the lidar is half a metre off centre.
        base = T @ odometry.last_pose @ np.linalg.inv(T)
        # flat floor, so I just read x, y and yaw out of the 3D pose
        poses[i] = base[0, 3], base[1, 3], np.arctan2(base[1, 0], base[0, 0])
    # arctan2 jumps at +-180 deg, unwrap makes the heading continuous
    poses[:, 2] = np.unwrap(poses[:, 2])
    return poses - poses[0]
