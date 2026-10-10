"""Side by side animation: the run as my odometry sees it vs the ground truth.

Left panel: every scan's wall points placed in the room using MY pose for
that scan. Right panel: the same points placed using the ground-truth pose.
If my poses are right the two pictures look the same, the walls stay thin
and the two paths match. If my poses were off, the walls on the left would
smear or drift.

This opens the ground truth, so like evaluate.py it is only for checking the
result. Nothing in algos/ uses it.

Usage: python -m lidar_odom.make_comparison wall_icp figures/comparison.gif
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lidar_odom.evaluate import load, match, to_gt_frame, wrap
from lidar_odom.preprocess import CELL, preprocess


def place(pts, pose):
    """Robot-frame points into the room frame for a robot at pose (x, y, yaw)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + pose[:2]


def build_map(scans, poses):
    """Stack all scans into one point set, keeping one point per 10 cm cell.

    Returns the points and, for every scan, how many points the map had once
    that scan was added. Points are only ever appended, so frame i of the
    animation is just the first count[i] points.
    """
    xy = np.empty((0, 2))
    cells = np.empty(0, np.int64)
    count = []
    for pts, pose in zip(scans, poses):
        world = place(pts, pose)
        ij = np.floor(world / CELL).astype(np.int64)
        ids, first = np.unique(ij[:, 0] * 1_000_000 + ij[:, 1], return_index=True)
        new = ~np.isin(ids, cells)
        xy = np.vstack([xy, world[first[new]]])
        cells = np.concatenate([cells, ids[new]])
        count.append(len(xy))
    return xy, np.array(count)


def main(name, out_path, every=8, cache_path="cache/sensors.npz",
         gt_path="cache/gt.csv", results_dir="results"):
    every = int(every)
    data = dict(np.load(cache_path))
    T, start = data["T_base_lidar"], data["scan_start"]
    est = load(Path(results_dir) / f"{name}.csv")
    gt = match(est, load(gt_path))
    # my odometry starts at (0, 0, 0). Put its first pose on the first
    # ground-truth pose so both panels are drawn in the same frame. Only the
    # first pose is used for this, everything after is my own estimate.
    mine = to_gt_frame(est, gt)
    truth = gt[:, 1:]

    scans = [preprocess(data["scan_xyz"][start[i]:start[i + 1]], T)
             for i in range(len(est))]
    maps = [build_map(scans, mine), build_map(scans, truth)]
    pos_err = 100 * np.linalg.norm(mine[:, :2] - truth[:, :2], axis=1)
    yaw_err = np.degrees(np.abs(wrap(mine[:, 2] - truth[:, 2])))
    t = est[:, 0] - est[0, 0]

    # same axis limits on both panels, taken from the ground-truth map
    lo, hi = maps[1][0].min(axis=0) - 0.5, maps[1][0].max(axis=0) + 0.5

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4), dpi=90)
    titles = [f"My odometry ({name})", "Ground truth"]
    colors = ["tab:orange", "black"]
    art = []
    for ax, title, color in zip(axes, titles, colors):
        ax.set_title(title)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, linewidth=0.3)
        walls, = ax.plot([], [], ".", color="0.45", markersize=1.5)
        scan, = ax.plot([], [], ".", color="tab:blue", markersize=3)
        path, = ax.plot([], [], "-", color=color, linewidth=1.8)
        robot, = ax.plot([], [], "o", color=color, markersize=7)
        nose, = ax.plot([], [], "-", color=color, linewidth=2)
        art.append((walls, scan, path, robot, nose))
    label = fig.suptitle("", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    frames = []
    for i in range(0, len(est), every):
        for (walls, scan, path, robot, nose), (xy, count), poses in zip(art, maps, (mine, truth)):
            walls.set_data(xy[:count[i], 0], xy[:count[i], 1])
            now = place(scans[i], poses[i])
            scan.set_data(now[:, 0], now[:, 1])
            path.set_data(poses[:i + 1, 0], poses[:i + 1, 1])
            x, y, yaw = poses[i]
            robot.set_data([x], [y])
            # short line showing which way the robot faces
            nose.set_data([x, x + 0.6 * np.cos(yaw)], [y, y + 0.6 * np.sin(yaw)])
        label.set_text(f"t = {t[i]:5.1f} s     position error {pos_err[i]:4.1f} cm"
                       f"     yaw error {yaw_err[i]:4.2f} deg")
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(image).quantize(colors=48, dither=Image.Dither.NONE))
    plt.close(fig)

    frame_ms = 1000 * np.median(np.diff(est[:, 0]))
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(frame_ms), loop=0, optimize=True)
    print(f"{len(frames)} frames -> {out_path}")
    print(f"map points: mine {len(maps[0][0])}, ground truth {len(maps[1][0])}")


if __name__ == "__main__":
    main(*sys.argv[1:])
