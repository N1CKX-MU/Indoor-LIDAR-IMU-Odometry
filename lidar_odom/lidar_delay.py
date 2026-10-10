"""Measure how far the LiDAR lags the IMU, from the two sensors alone."""
import sys

import numpy as np

from lidar_odom.algos import wall_icp
from lidar_odom.imu import standstill, yaw_from_gyro


def main(cache_path="cache/sensors.npz"):
    data = dict(np.load(cache_path))
    t, gyro, acc = data["imu_t"], data["imu_gyro"], data["imu_acc"]
    gyro_yaw, _ = yaw_from_gyro(t, gyro, standstill(t, gyro, acc))

    wall_icp.LIDAR_DELAY = 0.0
    lidar_yaw = wall_icp.run(data)[:, 2]

    delays = np.arange(0.0, 0.2001, 0.002)
    mismatch = [np.std(lidar_yaw - np.interp(data["scan_t"] - d, t, gyro_yaw))
                for d in delays]
    best = int(np.argmin(mismatch))
    print(f"mismatch with no delay: {np.degrees(mismatch[0]):.3f} deg")
    print(f"best delay: {delays[best] * 1000:.0f} ms, "
          f"mismatch {np.degrees(mismatch[best]):.3f} deg")


if __name__ == "__main__":
    main(*sys.argv[1:])
