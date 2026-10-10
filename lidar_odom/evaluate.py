"""Score a trajectory against ground truth: ATE, yaw error and the report plots.

This is the only file in the project that opens the ground truth.

Usage: python -m lidar_odom.evaluate <name> scores results/<name>.csv.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    """Read a t,x,y,yaw_deg csv (works for both results and ground truth).

    The files store yaw in degrees. Everything in here works in radians,
    so I convert it once on the way in.
    """
    rows = np.loadtxt(path, delimiter=",", skiprows=1)
    rows[:, 3] = np.radians(rows[:, 3])
    return rows


def match(est, gt, tolerance=1e-3):
    """Ground-truth rows at the estimate's timestamps.

    I pair rows by time and not by row number, because the ground truth has
    one more row than there are scans. The assert is there so that a timing
    mix up stops the script, it should never just pass quietly.
    """
    # searchsorted gives the first gt row that is not earlier than each
    # estimate, then I step back one where the previous row is closer
    idx = np.clip(np.searchsorted(gt[:, 0], est[:, 0]), 1, len(gt) - 1)
    idx -= (est[:, 0] - gt[idx - 1, 0]) < (gt[idx, 0] - est[:, 0])
    assert np.abs(gt[idx, 0] - est[:, 0]).max() < tolerance
    return gt[idx]


def wrap(angle):
    """Bring an angle into -pi..pi, so 179 deg vs -179 deg counts as 2 deg apart."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def transform(xy, yaw, x, y):
    """Rotate points by yaw, then shift them by (x, y)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return xy @ np.array([[c, s], [-s, c]]) + [x, y]


def to_gt_frame(est, gt):
    """Express the estimate in the ground-truth frame using only the first pose.

    My odometry starts at (0, 0) facing 0. The ground truth starts somewhere
    slightly different. Here I put my first pose on top of the first ground
    truth pose and leave everything else alone, so whatever difference shows
    up later is real drift.
    """
    x0, y0, yaw0 = gt[0, 1:]
    xy = transform(est[:, 1:3], yaw0, x0, y0)
    return np.column_stack([xy, est[:, 3] + yaw0])


def best_fit(xy, target):
    """Rigid 2D transform (rotation + shift, no scale) that best maps xy onto target.

    This is the usual way ATE is defined: line the whole path up as well as
    possible first, then measure what is left. In 2D there is a closed form
    for it. Centre both paths, the rotation comes out of one atan2, and the
    shift is whatever puts the centres on top of each other.
    """
    a, b = xy - xy.mean(axis=0), target - target.mean(axis=0)
    yaw = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    shift = target.mean(axis=0) - transform(xy.mean(axis=0)[None], yaw, 0, 0)[0]
    return transform(xy, yaw, *shift)


def evaluate(est_path, gt_path):
    """All the numbers for one trajectory, as a dict.

    I compute ATE under both alignments. Best fit is the standard one, first
    pose is stricter, and giving both means the choice of alignment cannot
    be used to make the result look better. Yaw error always uses the first
    pose alignment.
    """
    est = load(est_path)
    gt = match(est, load(gt_path))
    placed = to_gt_frame(est, gt)

    # distance between my position and the true one, per scan
    drift = np.linalg.norm(placed[:, :2] - gt[:, 1:3], axis=1)
    fitted = np.linalg.norm(best_fit(est[:, 1:3], gt[:, 1:3]) - gt[:, 1:3], axis=1)
    yaw_error = np.degrees(np.abs(wrap(placed[:, 2] - gt[:, 3])))
    return {
        "t": est[:, 0] - est[0, 0],
        "est": placed,
        "gt": gt[:, 1:],
        "drift": drift,
        "yaw_error": yaw_error,
        # ATE is a root mean square, so big errors count more than small ones
        "ate": np.sqrt((fitted ** 2).mean()),
        "ate_first_pose": np.sqrt((drift ** 2).mean()),
        "final_drift": drift[-1],
        "yaw_mean": yaw_error.mean(),
        "yaw_max": yaw_error.max(),
    }


def plot(result, name, out_dir):
    """The two plots the report asks for: path vs ground truth, error over time."""
    fig, ax = plt.subplots(figsize=(7, 7))
    # ground truth solid, estimate dashed, so you can tell them apart even
    # when they sit on top of each other
    ax.plot(result["gt"][:, 0], result["gt"][:, 1], color="black",
            linewidth=2, label="ground truth")
    ax.plot(result["est"][:, 0], result["est"][:, 1], color="tab:blue",
            linewidth=1.5, linestyle="--", label=name)
    ax.plot(*result["gt"][0, :2], "o", color="black", markersize=7, label="start")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Trajectory: {name} vs ground truth")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.savefig(out_dir / f"{name}_trajectory.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # position and yaw error get a panel each, they have different units
    fig, (ax_p, ax_y) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_p.plot(result["t"], result["drift"], color="tab:blue", linewidth=1.2)
    ax_p.set_ylabel("position error [m]")
    ax_p.set_title(f"Error over time: {name}")
    ax_y.plot(result["t"], result["yaw_error"], color="tab:blue", linewidth=1.2)
    ax_y.set_ylabel("yaw error [deg]")
    ax_y.set_xlabel("time since start [s]")
    for ax in (ax_p, ax_y):
        # start the axis at zero so a small error does not look big
        ax.set_ylim(bottom=0)
        ax.grid(True, linewidth=0.3)
    fig.savefig(out_dir / f"{name}_error.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(name, gt_path="cache/gt.csv", results_dir="results", out_dir="figures"):
    result = evaluate(Path(results_dir) / f"{name}.csv", gt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(result, name, out_dir)
    print(f"{name}")
    print(f"  ATE (best-fit aligned)   {result['ate']:.3f} m")
    print(f"  ATE (first pose aligned) {result['ate_first_pose']:.3f} m")
    print(f"  final position error     {result['final_drift']:.3f} m")
    print(f"  yaw error  mean {result['yaw_mean']:.3f} deg, max {result['yaw_max']:.3f} deg")


if __name__ == "__main__":
    main(*sys.argv[1:])
