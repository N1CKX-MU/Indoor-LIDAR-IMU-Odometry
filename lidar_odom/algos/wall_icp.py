"""Wall odometry: point-to-line ICP of each scan against a map, gyro as rotation prior."""
import numpy as np
from scipy.spatial import cKDTree

from lidar_odom.imu import standstill, yaw_from_gyro
from lidar_odom.preprocess import preprocess

MAP_CELL = 0.10
NEIGHBOURS = 5
MAX_SPREAD = 0.5
MAX_THICKNESS = 0.03
MAX_RESIDUAL = 0.3
HUBER = 0.03
ITERATIONS = 10
CONVERGED = 1e-5
LIDAR_DELAY = 0.082
VELOCITY_SCANS = 4


def to_world(pts, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + pose[:2]


def cell_id(xy):
    ij = np.floor(xy / MAP_CELL).astype(np.int64)
    return ij[:, 0] * 1_000_000 + ij[:, 1]


class Map:
    """Wall points in the world frame, at most one per grid cell."""

    def __init__(self):
        self.xy = np.empty((0, 2))
        self.cells = np.empty(0, np.int64)
        self.tree = None

    def add(self, xy):
        ids, first = np.unique(cell_id(xy), return_index=True)
        new = ~np.isin(ids, self.cells)
        if new.any():
            self.xy = np.vstack([self.xy, xy[first[new]]])
            self.cells = np.concatenate([self.cells, ids[new]])
            self.tree = cKDTree(self.xy)

    def lines(self, xy):
        """Local wall line near each query point: a point on it, its normal, validity."""
        dist, idx = self.tree.query(xy, k=NEIGHBOURS)
        near = self.xy[idx]
        centre = near.mean(axis=1)
        d = near - centre[:, None]
        cov = np.einsum("nki,nkj->nij", d, d) / NEIGHBOURS
        var, vec = np.linalg.eigh(cov)
        ok = (dist[:, -1] < MAX_SPREAD) & (var[:, 0] < MAX_THICKNESS ** 2)
        return centre, vec[:, :, 0], ok


def align(pts, pose, wall_map):
    """Refine `pose` so the scan's wall points fall on the map's wall lines."""
    pose = pose.copy()
    for _ in range(ITERATIONS):
        world = to_world(pts, pose)
        centre, normal, ok = wall_map.lines(world)
        r = ((world - centre) * normal).sum(axis=1)
        ok &= np.abs(r) < MAX_RESIDUAL
        arm = world - pose[:2]
        J = np.column_stack([normal, normal[:, 1] * arm[:, 0] - normal[:, 0] * arm[:, 1]])
        w = ok / np.maximum(1.0, np.abs(r) / HUBER)
        step = np.linalg.solve((J * w[:, None]).T @ J, -(J * w[:, None]).T @ r)
        pose += step
        if np.abs(step).max() < CONVERGED:
            break
    return pose


def run(data):
    """Return one pose (x, y, yaw) of base_link per scan, starting at zero."""
    T, start, scan_t = data["T_base_lidar"], data["scan_start"], data["scan_t"]
    t, gyro, acc = data["imu_t"], data["imu_gyro"], data["imu_acc"]
    gyro_yaw, _ = yaw_from_gyro(t, gyro, standstill(t, gyro, acc))
    seen_yaw = np.interp(scan_t - LIDAR_DELAY, t, gyro_yaw)
    turn = np.diff(seen_yaw, prepend=seen_yaw[0])
    since_seen = np.interp(scan_t, t, gyro_yaw) - seen_yaw

    wall_map = Map()
    seen = np.zeros((len(scan_t), 3))
    for i in range(len(scan_t)):
        pts = preprocess(data["scan_xyz"][start[i]:start[i + 1]], T)
        if i > 0:
            guess = seen[i - 1] + [0.0, 0.0, turn[i]]
            if i > 1:
                guess[:2] += seen[i - 1, :2] - seen[i - 2, :2]
            seen[i] = align(pts, guess, wall_map)
        wall_map.add(to_world(pts, seen[i]))
    n = VELOCITY_SCANS
    velocity = np.zeros((len(scan_t), 2))
    velocity[n:] = (seen[n:, :2] - seen[:-n, :2]) / (scan_t[n:] - scan_t[:-n])[:, None]
    return np.column_stack([seen[:, :2] + velocity * LIDAR_DELAY, seen[:, 2] + since_seen])
