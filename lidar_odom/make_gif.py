"""Render the 3D playback of the run to an animated GIF (used in the README).

Same picture as view3d.py, but drawn into a hidden window and saved frame
by frame, so the README can show the dataset without anyone having to run
anything.
"""
import sys

import numpy as np
import open3d as o3d
from PIL import Image

from lidar_odom.view3d import colorize, look_at_robot


def main(cache_path, out_path, every=8, width=640, height=400):
    # arguments arrive as text from the command line
    every, width, height = int(every), int(width), int(height)
    data = dict(np.load(cache_path))
    T, start = data["T_base_lidar"], data["scan_start"]
    frame_ms = 1000 * np.median(np.diff(data["scan_t"]))

    vis = o3d.visualization.Visualizer()
    vis.create_window("gif", width, height, visible=False)
    vis.get_render_option().point_size = 2.0
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
    cloud = o3d.geometry.PointCloud()

    frames = []
    # only every 8th scan, shown for the length of one. That keeps the file
    # small and makes the gif play at 8x speed.
    for i in range(0, len(start) - 1, every):
        pts, colors = colorize(data["scan_xyz"][start[i]:start[i + 1]], T)
        cloud.points = o3d.utility.Vector3dVector(pts)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        if not frames:
            vis.add_geometry(cloud)
            look_at_robot(vis)
        else:
            vis.update_geometry(cloud)
        vis.poll_events()
        vis.update_renderer()
        image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        # there are only a handful of colours in the picture, so 32 is
        # plenty and it shrinks the gif a lot
        frames.append(Image.fromarray((image * 255).astype(np.uint8))
                      .quantize(colors=32, dither=Image.Dither.NONE))
    vis.destroy_window()

    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(frame_ms), loop=0, optimize=True)
    print(f"{len(frames)} frames -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
