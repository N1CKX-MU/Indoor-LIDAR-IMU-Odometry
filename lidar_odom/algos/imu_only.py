"""Baseline: IMU only. Yaw from the gyro, position from the accelerometer."""
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

    R_base_imu = data["T_base_lidar"][:2, :2]
    lever = data["T_base_lidar"][:2, 3]

    acc_robot = (acc[:, :2] - acc[still, :2].mean(axis=0)) @ R_base_imu.T
    velocity = integrate(rotate(acc_robot, yaw), t)
    imu_xy = integrate(velocity, t)

    base_xy = imu_xy - rotate(np.tile(lever, (len(t), 1)), yaw) + lever

    scan_t = data["scan_t"]
    poses = np.column_stack([np.interp(scan_t, t, base_xy[:, 0]),
                             np.interp(scan_t, t, base_xy[:, 1]),
                             np.interp(scan_t, t, yaw)])
    poses[:, 2] -= poses[0, 2]
    return poses
