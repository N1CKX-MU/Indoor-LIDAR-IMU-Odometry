"""Turn one raw 3D scan into 2D wall points in the robot (base_link) frame.

A raw scan is about 8000 points and a lot of it is not useful: the robot's
own body, the floor, a few stray returns. The task says to use walls, so
this file boils a scan down to roughly 430 points that sit on walls, seen
from above. Every odometry method in algos/ starts from this.
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# I got these numbers by looking at the data, not by tuning for a score.
# In every scan there are ~630 points within 0.8 m of the sensor (the robot
# itself) and then nothing at all until 1.16 m, so 1.0 sits in the gap.
SELF_RADIUS = 1.0   # m, horizontal distance from the sensor; robot body is inside
FLOOR_Z = -0.1      # m in base_link; the floor is at about -0.4
CELL = 0.10         # m, grid cell size for the wall test and downsampling
MIN_POINTS = 3      # a wall cell needs at least this many points ...
MIN_HEIGHT = 0.30   # ... spread over at least this much height [m]


def to_base(xyz, T_base_lidar):
    """Move points from the LiDAR frame to the base_link frame.

    The lidar sits about half a metre off the robot's centre and is mounted
    facing backwards. If I skipped this, every turn on the spot would look
    like the robot moving sideways.
    """
    # points are rows, so it is xyz @ R.T and not R @ xyz
    return xyz @ T_base_lidar[:3, :3].T + T_base_lidar[:3, 3]


def crop(xyz, T_base_lidar):
    """Drop the robot's own body and the floor. Returns points in base_link."""
    # body cut first, in the lidar frame, because "distance from the sensor"
    # is simplest there
    not_self = np.hypot(xyz[:, 0], xyz[:, 1]) > SELF_RADIUS
    pts = to_base(xyz[not_self], T_base_lidar)
    return pts[pts[:, 2] > FLOOR_Z]


def wall_points_2d(pts):
    """Keep grid cells that hold a vertical structure; one 2D point per cell.

    The idea: look at the points from above and lay a 10 cm grid over them.
    A wall is vertical, so its points at different heights all fall into the
    same cell. A cell with a few points spread over some height is a wall.
    A cell with one point, or with points all at the same height, is not.

    Each wall cell then becomes a single 2D point (the average of what is in
    it). That also evens things out, a wall close to the robot gets hit far
    more often than one far away and would otherwise dominate.
    """
    ix = np.floor(pts[:, 0] / CELL).astype(np.int64)
    iy = np.floor(pts[:, 1] / CELL).astype(np.int64)
    # fold (ix, iy) into one number so np.unique can group points by cell.
    # cell[k] is which cell point k is in, count is how many points per cell.
    _, cell, count = np.unique(ix * 1_000_000 + iy,
                               return_inverse=True, return_counts=True)
    # lowest and highest point in each cell
    z_min = np.full(len(count), np.inf)
    z_max = np.full(len(count), -np.inf)
    np.minimum.at(z_min, cell, pts[:, 2])
    np.maximum.at(z_max, cell, pts[:, 2])
    is_wall = (count >= MIN_POINTS) & (z_max - z_min >= MIN_HEIGHT)

    x = np.bincount(cell, pts[:, 0]) / count  # mean x of each cell
    y = np.bincount(cell, pts[:, 1]) / count
    return np.column_stack([x, y])[is_wall]


def preprocess(xyz, T_base_lidar):
    """The whole thing in one call: raw scan in, 2D wall points out."""
    return wall_points_2d(crop(xyz, T_base_lidar))


def main(cache_path, out_dir):
    """Before/after picture for three scans, then counts and timing for all."""
    data = dict(np.load(cache_path))
    T = data["T_base_lidar"]
    start = data["scan_start"]
    n_scans = len(start) - 1

    # first scan, one a third of the way in, one two thirds in
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
        # same limits on the bottom row so you can see the room turn
        # between the three scans
        axes[1, col].set_xlim(-15, 15)
        axes[1, col].set_ylim(-15, 15)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "preprocess.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # run it on every scan. This doubles as a test that no scan comes back
    # empty, and it tells me how long preprocessing takes.
    t0 = time.perf_counter()
    kept = [len(preprocess(data["scan_xyz"][start[i]:start[i + 1]], T))
            for i in range(n_scans)]
    ms = (time.perf_counter() - t0) / n_scans * 1000
    print(f"wall points per scan: min {min(kept)}, "
          f"median {int(np.median(kept))}, max {max(kept)}")
    print(f"preprocessing time: {ms:.2f} ms per scan")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
