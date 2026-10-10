# Indoor LiDAR Odometry

2D odometry (x, y, yaw) for an indoor robot from a Livox 3D LiDAR and an IMU,
using walls as the main feature.

![3D playback of the dataset](figures/playback.gif)

The whole run at 8x speed, in the robot's frame: the robot stays at the centre
and the room moves around it. Blue points are kept as walls, orange are
rejected by the wall test, grey are floor, red are the robot's own body.
Regenerate with `python -m lidar_odom.make_gif cache/sensors.npz figures/playback.gif`,
or watch it interactively with `python -m lidar_odom.view3d cache/sensors.npz`.

## Setup

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt

Unpack the bag so that `data/rosbag/metadata.yaml` exists.

## Ground truth

The bag contains ground truth (`/odom_sim` and the `odom -> base_link`
transform). It is used only by the evaluation script, never by the odometry.
