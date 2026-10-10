"""Gyro bias estimation and yaw integration.

Small helpers shared by the IMU baseline and the lidar methods. The useful
thing the IMU gives me in this project is heading: add up the gyro's turn
rate over time. Before that I take out the gyro's bias, which I measure
while the robot is still standing at the start.
"""
import numpy as np

# anything below these counts as "not moving". Resting noise is around
# 0.0002 rad/s and 0.012 m/s^2 and real motion is 0.38 rad/s and ~2 m/s^2,
# so there is a lot of room on both sides and the exact values do not matter.
GYRO_STILL = 0.005
ACC_STILL = 0.15
# stop the standstill window half a second before the first movement,
# to be safe
MARGIN = 0.5


def integrate(values, t):
    """Running integral of `values` over time `t` (trapezoid rule), starting at 0.

    values is 2D (samples x columns), each column is integrated on its own.
    """
    # average of each pair of neighbours times the time between them
    steps = 0.5 * (values[1:] + values[:-1]) * np.diff(t)[:, None]
    # the row of zeros in front keeps the output as long as the input
    return np.vstack([np.zeros(values.shape[1]), np.cumsum(steps, axis=0)])


def standstill(imu_t, gyro, acc):
    """Boolean mask of the IMU samples before the robot first moves.

    The robot counts as still until the first sample where it either turns
    or its horizontal acceleration jumps away from the usual value. In this
    bag that gives me the first 16 seconds.
    """
    # subtract the median so I am looking at changes, not the resting offset
    horizontal = acc[:, :2] - np.median(acc[:, :2], axis=0)
    turning = np.abs(gyro[:, 2]) > GYRO_STILL
    accelerating = (np.abs(horizontal) > ACC_STILL).any(axis=1)
    moving = turning | accelerating
    # argmax on a True/False array gives the position of the first True
    first_motion = imu_t[np.argmax(moving)]
    return imu_t < first_motion - MARGIN


def yaw_from_gyro(imu_t, gyro, still):
    """Yaw angle at every IMU sample [rad], zero at the first sample.

    Also returns the bias it removed. The bias in this bag is tiny (worth
    about 0.1 deg over the whole run) but on a real IMU it would not be,
    so I take it out anyway.
    """
    bias = gyro[still, 2].mean()
    # gyro[:, 2:3] keeps it 2D with one column, which is what integrate wants
    return integrate(gyro[:, 2:3] - bias, imu_t)[:, 0], bias
