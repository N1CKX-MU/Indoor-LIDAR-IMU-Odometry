"""Measure how far the LiDAR lags the IMU, from the two sensors alone.

Why this exists: my heading was fine when the robot drove straight but off
during turns, and the error grew with how fast it turned. That looked like
a timing problem, so I wanted to measure it without touching ground truth.

The heading can be worked out twice, once from the gyro and once from the
lidar (by running the wall odometry). Plotted over time the two curves have
the same shape. If one sensor is late its curve is shifted sideways. So I
slide the gyro curve in small steps and look for the shift where the two
agree best.
"""
import sys

import numpy as np

from lidar_odom.algos import wall_icp
from lidar_odom.imu import standstill, yaw_from_gyro


def main(cache_path="cache/sensors.npz"):
    data = dict(np.load(cache_path))
    t, gyro, acc = data["imu_t"], data["imu_gyro"], data["imu_acc"]
    gyro_yaw, _ = yaw_from_gyro(t, gyro, standstill(t, gyro, acc))

    # run the odometry with the correction switched off, so the lidar
    # heading has no timing assumption baked into it
    wall_icp.LIDAR_DELAY = 0.0
    lidar_yaw = wall_icp.run(data)[:, 2]

    # try delays from 0 to 200 ms in 2 ms steps. For each one, read the
    # gyro heading that much earlier and see how much the difference to the
    # lidar heading wobbles. std ignores a constant offset between the two,
    # which is what I want, only the shape should count.
    delays = np.arange(0.0, 0.2001, 0.002)
    mismatch = [np.std(lidar_yaw - np.interp(data["scan_t"] - d, t, gyro_yaw))
                for d in delays]
    best = int(np.argmin(mismatch))
    print(f"mismatch with no delay: {np.degrees(mismatch[0]):.3f} deg")
    print(f"best delay: {delays[best] * 1000:.0f} ms, "
          f"mismatch {np.degrees(mismatch[best]):.3f} deg")


if __name__ == "__main__":
    main(*sys.argv[1:])
