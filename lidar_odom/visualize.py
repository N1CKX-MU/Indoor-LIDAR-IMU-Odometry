"""Plot the raw sensor data in cache/sensors.npz as a sanity check.

I wrote this before any algorithm, just to look at what the lidar and the
imu actually give. Most of the later decisions (cut the floor, cut the robot
body, use a map and not scan to scan) came from staring at these two plots.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write image files, do not open a window
import matplotlib.pyplot as plt
import numpy as np


def get_scan(data, i):
    """Points of scan i, in the LiDAR frame."""
    start = data["scan_start"]
    return data["scan_xyz"][start[i]:start[i + 1]]


def plot_scans(data, out_dir, n_accumulate=20, near=1.5):
    """Four views of the point cloud: one scan, 20 scans, side view, close up.

    Stacking the first 20 scans is only ok because the robot is not moving
    yet at the start of the bag. One Livox scan on its own is quite sparse,
    the pattern lands on different spots every time, so 20 together show
    the room much better.
    """
    one = get_scan(data, 0)
    many = np.concatenate([get_scan(data, i) for i in range(n_accumulate)])
    # the close up is there to see the robot's own body in the scan
    close = many[np.linalg.norm(many[:, :2], axis=1) < near]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # (axis, points, column for x axis, column for y axis, aspect, title)
    panels = [
        (axes[0, 0], one, 0, 1, "equal", "One scan (50 ms), top view"),
        (axes[0, 1], many, 0, 1, "equal", f"First {n_accumulate} scans, top view"),
        (axes[1, 0], many, 0, 2, "auto", f"First {n_accumulate} scans, side view"),
        (axes[1, 1], close, 0, 1, "equal", f"Points within {near} m, top view"),
    ]
    for ax, pts, a, b, aspect, title in panels:
        # colour by height so floor and walls are easy to tell apart
        sc = ax.scatter(pts[:, a], pts[:, b], c=pts[:, 2], s=1,
                        cmap="viridis", vmin=-2, vmax=2)
        ax.plot(0, 0, "r+", markersize=12)  # the sensor
        ax.set_xlabel("xyz"[a] + " [m]")
        ax.set_ylabel("xyz"[b] + " [m]")
        ax.set_title(title)
        ax.set_aspect(aspect)
        ax.grid(True, linewidth=0.3)
    fig.colorbar(sc, ax=axes, label="height z in LiDAR frame [m]", shrink=0.6)
    fig.savefig(out_dir / "scans_raw.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_imu(data, out_dir):
    """Gyro on top, accelerometer below, all three axes each.

    What I was looking for: when does the robot start moving (about 16 s in),
    which axis it turns about (only z), and how noisy the accelerometer is
    (very, it spikes every time the robot starts or stops).
    """
    t = data["imu_t"] - data["imu_t"][0]
    fig, (ax_g, ax_a) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for k, name in enumerate("xyz"):
        ax_g.plot(t, data["imu_gyro"][:, k], linewidth=0.8, label=name)
        ax_a.plot(t, data["imu_acc"][:, k], linewidth=0.8, label=name)
    ax_g.set_ylabel("angular velocity [rad/s]")
    ax_g.set_title("Gyroscope")
    ax_a.set_ylabel("acceleration [m/s^2]")
    ax_a.set_title("Accelerometer")
    ax_a.set_xlabel("time since start [s]")
    for ax in (ax_g, ax_a):
        ax.legend(title="axis", loc="upper right")
        ax.grid(True, linewidth=0.3)
    fig.savefig(out_dir / "imu_raw.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(cache_path, out_dir):
    # dict() reads every array once; without it each data["..."] access
    # would decompress that array from the file again.
    data = dict(np.load(cache_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_scans(data, out_dir)
    plot_imu(data, out_dir)

    # a few numbers to go with the pictures. Points closer than 1 m turned
    # out to be the robot itself.
    sizes = np.diff(data["scan_start"])
    ranges = np.linalg.norm(get_scan(data, 0), axis=1)
    print(f"points per scan: min {sizes.min()}, max {sizes.max()}")
    print(f"scan 0 range: min {ranges.min():.2f} m, max {ranges.max():.2f} m")
    print(f"scan 0 points closer than 1 m: {(ranges < 1).sum()}")
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
