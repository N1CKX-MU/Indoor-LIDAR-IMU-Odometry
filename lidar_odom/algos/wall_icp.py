"""Wall odometry: point-to-line ICP of each scan against a map, gyro as rotation prior.

This is my main method. In short:

  1. every scan is cut down to 2D wall points (preprocess.py)
  2. the wall points of earlier scans are kept as a map of the room
  3. for a new scan I guess the pose (last pose + what the gyro turned),
     then shift and turn the scan until its points sit on the walls in the
     map. Where it ends up is the robot's pose.

The robot stands still for the first 16 s, so the map gets built from a
known spot before anything moves.
"""
import numpy as np
from scipy.spatial import cKDTree

from lidar_odom.imu import standstill, yaw_from_gyro
from lidar_odom.preprocess import preprocess

# I picked these by thinking about the sizes involved and did not tune them
# afterwards.
MAP_CELL = 0.10        # m, the map keeps one point per cell (same as preprocessing)
NEIGHBOURS = 5         # map points used to fit the local piece of wall
MAX_SPREAD = 0.5       # m, the 5 neighbours have to be this close, else no wall here yet
MAX_THICKNESS = 0.03   # m, and this straight, else it is a corner or clutter
MAX_RESIDUAL = 0.3     # m, a point this far from its line is ignored
HUBER = 0.03           # m, points further off than this count for less
ITERATIONS = 10        # upper limit, it normally stops earlier
CONVERGED = 1e-5       # stop when the pose barely changes any more
# how late a scan is compared to the IMU. lidar_delay.py measures about
# 0.082 s. I tried correcting for it, but against the odom -> base_link
# ground truth it made no real difference (yaw error 0.09 deg without,
# 0.10 deg with), so it is switched off. Put 0.082 here to turn it back on.
LIDAR_DELAY = 0.0
VELOCITY_SCANS = 4     # speed is taken over the last 4 scans (0.2 s)


def to_world(pts, pose):
    """Put robot-frame points into the map frame for a robot at pose (x, y, yaw)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return pts @ np.array([[c, s], [-s, c]]) + pose[:2]


def cell_id(xy):
    """One integer per 10 cm grid cell, so cells can be compared quickly."""
    ij = np.floor(xy / MAP_CELL).astype(np.int64)
    return ij[:, 0] * 1_000_000 + ij[:, 1]


class Map:
    """Wall points in the world frame, at most one per grid cell.

    A cell keeps the first point that lands in it and is never updated after
    that. Most of the room gets filled in while the robot is standing still,
    so the map is basically drawn from a known pose. Because nothing is ever
    overwritten, drift from later scans cannot creep into it.
    """

    def __init__(self):
        self.xy = np.empty((0, 2))
        self.cells = np.empty(0, np.int64)
        self.tree = None

    def add(self, xy):
        """Add the points that fall in cells the map does not have yet."""
        ids, first = np.unique(cell_id(xy), return_index=True)
        new = ~np.isin(ids, self.cells)
        if new.any():
            self.xy = np.vstack([self.xy, xy[first[new]]])
            self.cells = np.concatenate([self.cells, ids[new]])
            # the tree is for fast nearest neighbour lookups. Only rebuilt
            # when something was actually added.
            self.tree = cKDTree(self.xy)

    def lines(self, xy):
        """Local wall line near each query point: a point on it, its normal, validity.

        For every query point I take its 5 nearest map points and fit a
        straight line through them. That line is the bit of wall the point
        should be sitting on.
        """
        dist, idx = self.tree.query(xy, k=NEIGHBOURS)
        near = self.xy[idx]
        centre = near.mean(axis=1)
        d = near - centre[:, None]
        # 2x2 covariance of each group of 5, all at once
        cov = np.einsum("nki,nkj->nij", d, d) / NEIGHBOURS
        # eigh sorts smallest first. For points along a wall the small
        # direction is across the wall (the normal) and its variance says
        # how thick the group is.
        var, vec = np.linalg.eigh(cov)
        ok = (dist[:, -1] < MAX_SPREAD) & (var[:, 0] < MAX_THICKNESS ** 2)
        return centre, vec[:, :, 0], ok


def align(pts, pose, wall_map):
    """Refine `pose` so the scan's wall points fall on the map's wall lines.

    Point-to-line ICP. Each round: place the scan with the current pose,
    find the wall line next to every point, measure how far each point is
    from its line, then solve for the small (dx, dy, dyaw) that shrinks
    those distances the most. Repeat until it stops moving.

    I measure to the line and not to the nearest map point because scan and
    map never hit the same spots on a wall. Point to point would drag the
    scan along the wall towards wherever the map happens to have points.
    """
    pose = pose.copy()
    for _ in range(ITERATIONS):
        world = to_world(pts, pose)
        centre, normal, ok = wall_map.lines(world)
        # signed distance of each point from its line
        r = ((world - centre) * normal).sum(axis=1)
        ok &= np.abs(r) < MAX_RESIDUAL
        # how each distance changes if the robot moves or turns a little.
        # Moving by (dx, dy) changes it by normal . (dx, dy). Turning swings
        # the point sideways around the robot, so that term depends on the
        # arm from the robot to the point.
        arm = world - pose[:2]
        J = np.column_stack([normal, normal[:, 1] * arm[:, 0] - normal[:, 0] * arm[:, 1]])
        # weights: 0 for rejected points, 1 for points within 3 cm, less
        # and less beyond that so a few bad matches cannot pull the answer
        w = ok / np.maximum(1.0, np.abs(r) / HUBER)
        # weighted least squares for the three unknowns
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
    # the next three lines are the delay handling. A scan stamped t shows
    # the room as it was LIDAR_DELAY earlier, so the gyro is read at that
    # earlier moment. With LIDAR_DELAY = 0 this is just the gyro heading
    # at each scan and since_seen is all zeros.
    seen_yaw = np.interp(scan_t - LIDAR_DELAY, t, gyro_yaw)
    # how much the robot turned from one scan to the next
    turn = np.diff(seen_yaw, prepend=seen_yaw[0])
    # and how much it turned between that earlier moment and the stamp
    since_seen = np.interp(scan_t, t, gyro_yaw) - seen_yaw

    wall_map = Map()
    # seen[i] is the pose at the moment scan i really shows
    seen = np.zeros((len(scan_t), 3))
    for i in range(len(scan_t)):
        pts = preprocess(data["scan_xyz"][start[i]:start[i + 1]], T)
        if i > 0:
            # first guess: last pose, plus the gyro's turn, plus the last
            # step repeated (the robot probably kept going the same way)
            guess = seen[i - 1] + [0.0, 0.0, turn[i]]
            if i > 1:
                guess[:2] += seen[i - 1, :2] - seen[i - 2, :2]
            seen[i] = align(pts, guess, wall_map)
        # locate first, then add to the map. Scan 0 just starts the map.
        wall_map.add(to_world(pts, seen[i]))
    # move every pose forward from "when the scan was really taken" to its
    # stamp: speed times delay for position, the gyro's turn for heading.
    # Only past scans are used, so this would also work running live.
    # (With the delay at 0 this adds nothing and seen is returned as it is.)
    n = VELOCITY_SCANS
    velocity = np.zeros((len(scan_t), 2))
    velocity[n:] = (seen[n:, :2] - seen[:-n, :2]) / (scan_t[n:] - scan_t[:-n])[:, None]
    return np.column_stack([seen[:, :2] + velocity * LIDAR_DELAY, seen[:, 2] + since_seen])
