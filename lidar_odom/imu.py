"""Gyro bias estimation and yaw integration."""
import numpy as np

GYRO_STILL = 0.005
ACC_STILL = 0.15
MARGIN = 0.5


def integrate(values, t):
    """Running integral of `values` over time `t` (trapezoid rule), starting at 0."""
    steps = 0.5 * (values[1:] + values[:-1]) * np.diff(t)[:, None]
    return np.vstack([np.zeros(values.shape[1]), np.cumsum(steps, axis=0)])


def standstill(imu_t, gyro, acc):
    """Boolean mask of the IMU samples before the robot first moves."""
    horizontal = acc[:, :2] - np.median(acc[:, :2], axis=0)
    turning = np.abs(gyro[:, 2]) > GYRO_STILL
    accelerating = (np.abs(horizontal) > ACC_STILL).any(axis=1)
    moving = turning | accelerating
    first_motion = imu_t[np.argmax(moving)]
    return imu_t < first_motion - MARGIN


def yaw_from_gyro(imu_t, gyro, still):
    """Yaw angle at every IMU sample [rad], zero at the first sample."""
    bias = gyro[still, 2].mean()
    return integrate(gyro[:, 2:3] - bias, imu_t)[:, 0], bias
