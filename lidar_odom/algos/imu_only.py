"""Baseline: IMU only. Yaw from the gyro, position from the accelerometer.

This is the "what if I had no lidar" method. I did not expect it to work
for position and it does not (it ends up tens of metres off), but that is
the point: it shows what the lidar adds. The heading it gives is good, and
the lidar methods reuse that part.
"""
import numpy as np

from lidar_odom.imu import integrate, standstill, yaw_from_gyro


def rotate(xy, yaw):
    """Rotate each row of `xy` by the matching angle in `yaw`."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack([c * xy[:, 0] - s * xy[:, 1],
                            s * xy[:, 0] + c * xy[:, 1]])


def run(data):
    """Return one pose (x, y, yaw) of base_link per scan, starting at zero."""
    t, gyro, acc = data["imu_t"], data["imu_gyro"], data["imu_acc"]
    still = standstill(t, gyro, acc)
    yaw, _ = yaw_from_gyro(t, gyro, still)

    # /tf_static gives the IMU exactly the same mount as the lidar (turned
    # 180 deg, about half a metre away from base_link), so I reuse that matrix
    R_base_imu = data["T_base_lidar"][:2, :2]
    lever = data["T_base_lidar"][:2, 3]

    # take off the resting reading (gravity leaking in plus bias), turn the
    # acceleration into the robot's axes, then into the fixed start frame.
    # It has to be a fixed frame before integrating, the robot's own axes
    # keep turning.
    acc_robot = (acc[:, :2] - acc[still, :2].mean(axis=0)) @ R_base_imu.T
    velocity = integrate(rotate(acc_robot, yaw), t)
    imu_xy = integrate(velocity, t)

    # that was the path of the IMU, not of base_link. Step back along the
    # lever arm. The + lever at the end makes the path start at zero.
    base_xy = imu_xy - rotate(np.tile(lever, (len(t), 1)), yaw) + lever

    # the IMU runs at 60 Hz, I want one pose per lidar scan (20 Hz)
    scan_t = data["scan_t"]
    poses = np.column_stack([np.interp(scan_t, t, base_xy[:, 0]),
                             np.interp(scan_t, t, base_xy[:, 1]),
                             np.interp(scan_t, t, yaw)])
    poses[:, 2] -= poses[0, 2]
    return poses
