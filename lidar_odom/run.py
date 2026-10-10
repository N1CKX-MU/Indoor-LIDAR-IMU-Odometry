"""Run one odometry algorithm over the cached data and save its trajectory.

Usage: python -m lidar_odom.run <name>, where <name> is a file in algos/
(imu_only, wall_icp, wall_lines, kiss). Every algorithm has the same
run(data) function, so this one script works for all of them.
"""
import importlib
import sys
import time
from pathlib import Path

import numpy as np


def main(name, cache_path="cache/sensors.npz", out_dir="results"):
    # data is only the sensor cache. There is no ground truth in it, so an
    # algorithm could not use it even by accident.
    data = dict(np.load(cache_path))
    # load algos/<name>.py from its name, so adding a new method needs no
    # change here
    algo = importlib.import_module(f"lidar_odom.algos.{name}")

    # I time only the algorithm, not loading the cache or saving the csv.
    # This is where the Hz number in the report comes from.
    t0 = time.perf_counter()
    poses = algo.run(data)
    seconds = time.perf_counter() - t0

    out_path = Path(out_dir) / f"{name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # one row per scan: time, x, y, yaw (radians). Same layout as gt.csv.
    np.savetxt(out_path, np.column_stack([data["scan_t"], poses]),
               delimiter=",", header="t,x,y,yaw", comments="", fmt="%.9f")
    x, y, yaw = poses[-1]
    print(f"{name}: {len(poses)} poses in {seconds:.2f} s "
          f"({len(poses) / seconds:.0f} Hz)")
    print(f"final pose: x {x:.3f} m, y {y:.3f} m, yaw {np.degrees(yaw):.2f} deg")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
