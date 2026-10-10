"""Wall odometry: RANSAC wall lines in each scan, matched to a map of wall lines.

A second way of using the walls, to compare with wall_icp. There the walls
stay as lots of points. Here every wall is boiled down to a straight line,
stored as its normal n and its distance d from the origin (n . p = d for
every point p on the wall). The whole room ends up as about 5 lines.

Once I know which line in the scan is which wall in the map, the pose can
be read off directly: how much the walls appear turned is how much the
robot turned, and how much closer or further each wall is tells me how far
the robot moved towards it. No iterating.
"""
import numpy as np

from lidar_odom.algos.wall_icp import LIDAR_DELAY, VELOCITY_SCANS
from lidar_odom.imu import standstill, yaw_from_gyro
from lidar_odom.preprocess import preprocess

TRIALS = 200                     # random point pairs tried per line
MIN_PAIR_DIST = 0.5              # m, skip pairs too close to give a direction
INLIER_DIST = 0.03               # m, a point this close belongs to the line
MIN_INLIERS = 30                 # about 3 m of wall, shorter bits are not trusted
MAX_LINES = 6                    # per scan
MATCH_ANGLE = np.radians(10.0)   # scan line and map line this close in direction ...
MATCH_DIST = 0.5                 # ... and in distance are the same wall
PRIOR_WEIGHT = 1e-3              # small pull towards the guess, see locate()


def find_lines(pts, rng):
    """Walls in one scan as rows (nx, ny, d, inliers): unit normal n, with n . p = d.

    RANSAC: pick two points at random, draw the line through them, count how
    many points lie on it. Do that 200 times and keep the best line. Two
    points on the same wall give a line that lots of points agree with, two
    points on different walls give a line nearly nothing agrees with. Then
    take that wall's points out and look for the next one.
    """
    lines = []
    while len(lines) < MAX_LINES and len(pts) >= MIN_INLIERS:
        # all 200 trials at once: row k of a and b is the k-th random pair
        a = pts[rng.integers(len(pts), size=TRIALS)]
        b = pts[rng.integers(len(pts), size=TRIALS)]
        along = b - a
        length = np.linalg.norm(along, axis=1)
        far = length > MIN_PAIR_DIST
        # the direction a->b turned by 90 deg is the line's normal
        normal = np.column_stack([-along[far, 1], along[far, 0]]) / length[far, None]
        # distance of every point (rows) from every candidate line (columns)
        dist = np.abs(pts @ normal.T - (normal * a[far]).sum(axis=1))
        best = (dist < INLIER_DIST).sum(axis=0).argmax()
        on_line = dist[:, best] < INLIER_DIST
        if on_line.sum() < MIN_INLIERS:
            break
        # the two point line was only a rough guess. Refit through all the
        # points that agreed with it, that is a lot more accurate.
        centre = pts[on_line].mean(axis=0)
        _, vec = np.linalg.eigh(np.cov((pts[on_line] - centre).T))
        lines.append([vec[0, 0], vec[1, 0], vec[:, 0] @ centre, on_line.sum()])
        pts = pts[~on_line]
    return np.array(lines).reshape(-1, 4)


def to_world(lines, pose):
    """The same lines expressed in the world frame, for a robot at `pose`.

    The normal turns with the robot. The distance changes by however far
    the robot has moved along that normal.
    """
    c, s = np.cos(pose[2]), np.sin(pose[2])
    normal = lines[:, :2] @ np.array([[c, s], [-s, c]])
    return np.column_stack([normal, lines[:, 2] + normal @ pose[:2], lines[:, 3]])


def match(world, wall_map):
    """Index of the map line each world line is (-1 if none), and the sign that aligns it.

    One thing to watch: (n, d) and (-n, -d) are the same line. So if a scan
    line points the opposite way to a map line I flip it before comparing,
    and hand the sign back so the caller can flip it too.
    """
    # rows are scan lines, columns are map lines
    cos = world[:, :2] @ wall_map[:, :2].T
    sign = np.where(cos < 0, -1.0, 1.0)
    gap = np.abs(wall_map[:, 2] - sign * world[:, 2:3])
    # pairs that are too different in direction or distance are ruled out
    gap[(np.abs(cos) < np.cos(MATCH_ANGLE)) | (gap > MATCH_DIST)] = np.inf
    idx = gap.argmin(axis=1)
    rows = np.arange(len(world))
    return np.where(np.isfinite(gap[rows, idx]), idx, -1), sign[rows, idx]


def locate(lines, guess, wall_map):
    """Pose from matched lines: heading from their directions, position from their offsets.

    Also returns the lines that matched nothing, already placed in the
    world, so they can be added to the map as new walls.
    """
    idx, sign = match(to_world(lines, guess), wall_map)
    # flip the lines that need it (normal and d, not the inlier count)
    lines = lines * np.column_stack([sign, sign, sign, np.ones(len(lines))])
    hit = idx >= 0
    pose = guess.copy()
    if hit.any():
        # walls with more points behind them count for more
        seen, ref, weight = to_world(lines[hit], guess), wall_map[idx[hit]], lines[hit, 3]
        # heading: the cross product of two unit normals is the sine of the
        # angle between them, i.e. how wrong my guessed heading is
        cross = seen[:, 0] * ref[:, 1] - seen[:, 1] * ref[:, 0]
        pose[2] += np.average(np.arcsin(cross), weights=weight)
        # position: each wall gives one equation, n_map . t = d_map - d_scan.
        # One wall only pins the robot down along its own normal, so I need
        # walls in two directions. PRIOR_WEIGHT keeps the solve defined if
        # only parallel walls matched (did not happen in this bag).
        A, b = ref[:, :2], ref[:, 2] - lines[hit, 2]
        prior = PRIOR_WEIGHT * weight.sum()
        pose[:2] = np.linalg.solve((A.T * weight) @ A + prior * np.eye(2),
                                   (A.T * weight) @ b + prior * guess[:2])
    return pose, to_world(lines[~hit], pose)


def run(data):
    """Return one pose (x, y, yaw) of base_link per scan, starting at zero."""
    # the gyro part and the delay handling are the same as in wall_icp
    T, start, scan_t = data["T_base_lidar"], data["scan_start"], data["scan_t"]
    t, gyro, acc = data["imu_t"], data["imu_gyro"], data["imu_acc"]
    gyro_yaw, _ = yaw_from_gyro(t, gyro, standstill(t, gyro, acc))
    seen_yaw = np.interp(scan_t - LIDAR_DELAY, t, gyro_yaw)
    turn = np.diff(seen_yaw, prepend=seen_yaw[0])
    since_seen = np.interp(scan_t, t, gyro_yaw) - seen_yaw

    # fixed seed, so RANSAC picks the same random pairs on every run and
    # the result is repeatable
    rng = np.random.default_rng(0)
    wall_map = np.empty((0, 4))
    seen = np.zeros((len(scan_t), 3))
    for i in range(len(scan_t)):
        lines = find_lines(preprocess(data["scan_xyz"][start[i]:start[i + 1]], T), rng)
        # scan 0: all its lines go into the map as they are
        new = to_world(lines, seen[0])
        if i > 0:
            guess = seen[i - 1] + [0.0, 0.0, turn[i]]
            if i > 1:
                guess[:2] += seen[i - 1, :2] - seen[i - 2, :2]
            # later scans: only lines that matched nothing are new walls
            seen[i], new = locate(lines, guess, wall_map)
        wall_map = np.vstack([wall_map, new])
    n = VELOCITY_SCANS
    velocity = np.zeros((len(scan_t), 2))
    velocity[n:] = (seen[n:, :2] - seen[:-n, :2]) / (scan_t[n:] - scan_t[:-n])[:, None]
    return np.column_stack([seen[:, :2] + velocity * LIDAR_DELAY, seen[:, 2] + since_seen])
