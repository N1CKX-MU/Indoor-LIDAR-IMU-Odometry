"""Turn one raw 3D scan into 2D wall points in the robot (base_link) frame."""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SELF_RADIUS = 1.0   # m, horizontal distance from the sensor; robot body is inside
FLOOR_Z = -0.1      # m in base_link; the floor is at about -0.4
CELL = 0.10         # m, grid cell size for the wall test and downsampling
MIN_POINTS = 3      # a wall cell needs at least this many points ...
MIN_HEIGHT = 0.30   # ... spread over at least this much height [m]


def to_base(xyz, T_base_lidar):
    """Move points from the LiDAR frame to the base_link frame."""
    return xyz @ T_base_lidar[:3, :3].T + T_base_lidar[:3, 3]


def crop(xyz, T_base_lidar):
    """Drop the robot's own body and the floor. Returns points in base_link."""
    not_self = np.hypot(xyz[:, 0], xyz[:, 1]) > SELF_RADIUS
    pts = to_base(xyz[not_self], T_base_lidar)
    return pts[pts[:, 2] > FLOOR_Z]


def wall_points_2d(pts):
    """Keep grid cells that hold a vertical structure; one 2D point per cell."""
    ix = np.floor(pts[:, 0] / CELL).astype(np.int64)
    iy = np.floor(pts[:, 1] / CELL).astype(np.int64)
    _, cell, count = np.unique(ix * 1_000_000 + iy,
                               return_inverse=True, return_counts=True)
    z_min = np.full(len(count), np.inf)
    z_max = np.full(len(count), -np.inf)
    np.minimum.at(z_min, cell, pts[:, 2])
    np.maximum.at(z_max, cell, pts[:, 2])
    is_wall = (count >= MIN_POINTS) & (z_max - z_min >= MIN_HEIGHT)

    x = np.bincount(cell, pts[:, 0]) / count  # mean x of each cell
    y = np.bincount(cell, pts[:, 1]) / count
    return np.column_stack([x, y])[is_wall]


def preprocess(xyz, T_base_lidar):
    return wall_points_2d(crop(xyz, T_base_lidar))


def main(cache_path, out_dir):
    data = dict(np.load(cache_path))
    T = data["T_base_lidar"]
    start = data["scan_start"]
    n_scans = len(start) - 1

    picks = [0, n_scans // 3, 2 * n_scans // 3]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, i in enumerate(picks):
        xyz = data["scan_xyz"][start[i]:start[i + 1]]
        raw = to_base(xyz, T)
        walls = preprocess(xyz, T)
        for ax, pts, name in ((axes[0, col], raw, "raw"),
                              (axes[1, col], walls, "wall points")):
            ax.scatter(pts[:, 0], pts[:, 1], s=1, color="tab:blue")
            ax.plot(0, 0, "r+", markersize=12)  # base_link
            ax.set_title(f"scan {i}: {name} ({len(pts)})")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_aspect("equal")
            ax.grid(True, linewidth=0.3)
        axes[1, col].set_xlim(-15, 15)
        axes[1, col].set_ylim(-15, 15)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "preprocess.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    t0 = time.perf_counter()
    kept = [len(preprocess(data["scan_xyz"][start[i]:start[i + 1]], T))
            for i in range(n_scans)]
    ms = (time.perf_counter() - t0) / n_scans * 1000
    print(f"wall points per scan: min {min(kept)}, "
          f"median {int(np.median(kept))}, max {max(kept)}")
    print(f"preprocessing time: {ms:.2f} ms per scan")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
