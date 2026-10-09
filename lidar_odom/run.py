"""Run one odometry algorithm over the cached data and save its trajectory."""
import importlib
import sys
import time
from pathlib import Path

import numpy as np


def main(name, cache_path="cache/sensors.npz", out_dir="results"):
    data = dict(np.load(cache_path))
    algo = importlib.import_module(f"lidar_odom.algos.{name}")

    t0 = time.perf_counter()
    poses = algo.run(data)
    seconds = time.perf_counter() - t0

    out_path = Path(out_dir) / f"{name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, np.column_stack([data["scan_t"], poses]),
               delimiter=",", header="t,x,y,yaw", comments="", fmt="%.9f")
    x, y, yaw = poses[-1]
    print(f"{name}: {len(poses)} poses in {seconds:.2f} s "
          f"({len(poses) / seconds:.0f} Hz)")
    print(f"final pose: x {x:.3f} m, y {y:.3f} m, yaw {np.degrees(yaw):.2f} deg")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
