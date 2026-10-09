"""Plot the raw sensor data in cache/sensors.npz as a sanity check."""
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
    one = get_scan(data, 0)
    many = np.concatenate([get_scan(data, i) for i in range(n_accumulate)])
    close = many[np.linalg.norm(many[:, :2], axis=1) < near]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        (axes[0, 0], one, 0, 1, "equal", "One scan (50 ms), top view"),
        (axes[0, 1], many, 0, 1, "equal", f"First {n_accumulate} scans, top view"),
        (axes[1, 0], many, 0, 2, "auto", f"First {n_accumulate} scans, side view"),
        (axes[1, 1], close, 0, 1, "equal", f"Points within {near} m, top view"),
    ]
    for ax, pts, a, b, aspect, title in panels:
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

    sizes = np.diff(data["scan_start"])
    ranges = np.linalg.norm(get_scan(data, 0), axis=1)
    print(f"points per scan: min {sizes.min()}, max {sizes.max()}")
    print(f"scan 0 range: min {ranges.min():.2f} m, max {ranges.max():.2f} m")
    print(f"scan 0 points closer than 1 m: {(ranges < 1).sum()}")
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
