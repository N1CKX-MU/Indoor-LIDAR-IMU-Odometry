"""Play the whole run as a 3D point cloud, coloured by preprocessing stage.

Points are shown in the robot (base_link) frame, so the robot stays at the
origin and the room appears to move around it.

I made this to check the preprocessing by eye: does the floor really get
cut, is the robot's body gone, do the walls survive.

Controls: mouse to rotate / zoom / pan, SPACE to pause, close the window to quit.
"""
import sys
import time

import numpy as np
import open3d as o3d

from lidar_odom.preprocess import CELL, FLOOR_Z, SELF_RADIUS, preprocess, to_base

WALL = (0.12, 0.47, 0.71)    # blue: kept as wall points
OTHER = (1.00, 0.50, 0.05)   # orange: above the floor but failed the wall test
FLOOR = (0.55, 0.55, 0.55)   # grey: removed by the floor cut
SELF = (0.84, 0.15, 0.16)    # red: removed by the self cut (robot body)
SPACE_KEY = 32


def cell_id(xy):
    """Same grid cell numbering as in preprocess.py."""
    ix = np.floor(xy[:, 0] / CELL).astype(np.int64)
    iy = np.floor(xy[:, 1] / CELL).astype(np.int64)
    return ix * 1_000_000 + iy


def colorize(xyz, T_base_lidar):
    """Points in base_link plus one colour per point saying what happened to it.

    The thresholds and the wall test are imported from preprocess.py and not
    copied here, so what this shows is always what the real pipeline does.
    """
    pts = to_base(xyz, T_base_lidar)
    is_self = np.hypot(xyz[:, 0], xyz[:, 1]) <= SELF_RADIUS
    is_floor = pts[:, 2] <= FLOOR_Z
    # a point is a wall point if its cell is one of the cells preprocessing kept
    wall_cells = cell_id(preprocess(xyz, T_base_lidar))
    is_wall = np.isin(cell_id(pts), wall_cells) & ~is_self & ~is_floor

    # start with everything orange, then paint over
    colors = np.tile(OTHER, (len(pts), 1))
    colors[is_wall] = WALL
    colors[is_floor] = FLOOR
    colors[is_self] = SELF
    return pts, colors


def look_at_robot(vis):
    """Aim the camera at the robot from above and behind."""
    # Open3D's default centres on the whole cloud, and the far floor
    # returns drag that way off the room
    view = vis.get_view_control()
    view.set_lookat([0.0, 0.0, 1.0])
    view.set_front([-0.5, -0.5, 0.7])
    view.set_up([0.0, 0.0, 1.0])
    view.set_zoom(0.3)


def play(data, speed):
    """Open the window and step through the scans at the given speed."""
    T = data["T_base_lidar"]
    start = data["scan_start"]
    t = data["scan_t"] - data["scan_t"][0]
    n_scans = len(t)
    frame_time = np.median(np.diff(t)) / speed
    # a dict so the key callback below can change it
    state = {"paused": False}

    def toggle_pause(_):
        state["paused"] = not state["paused"]
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("LiDAR playback (base_link frame)", 1280, 800)
    vis.register_key_callback(SPACE_KEY, toggle_pause)
    vis.get_render_option().point_size = 2.0
    # the three arrows at the origin are the robot's axes, 1 m long
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
    cloud = o3d.geometry.PointCloud()

    i = 0
    while vis.poll_events():
        tick = time.perf_counter()
        if i < n_scans and not state["paused"]:
            pts, colors = colorize(data["scan_xyz"][start[i]:start[i + 1]], T)
            cloud.points = o3d.utility.Vector3dVector(pts)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            if i == 0:
                # the cloud has to be added once, after that it is only updated
                vis.add_geometry(cloud)
                look_at_robot(vis)
            else:
                vis.update_geometry(cloud)
            if i % 20 == 0:
                print(f"\rt = {t[i]:6.1f} s   scan {i:4d}/{n_scans}", end="")
            i += 1
            if i == n_scans:
                print("\nend of run; close the window to quit")
        vis.update_renderer()
        # sleep off whatever is left of this frame's time slot
        time.sleep(max(0.0, frame_time - (time.perf_counter() - tick)))
    vis.destroy_window()


def main(cache_path, speed=1.0):
    print("blue = wall, orange = not a wall, grey = floor, red = robot body")
    play(dict(np.load(cache_path)), float(speed))


if __name__ == "__main__":
    main(*sys.argv[1:])
