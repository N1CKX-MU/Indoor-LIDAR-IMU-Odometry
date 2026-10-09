# Indoor LiDAR Odometry

2D odometry (x, y, yaw) for an indoor robot from a Livox 3D LiDAR and an IMU,
using walls as the main feature.

## Setup

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt

Unpack the bag so that `data/rosbag/metadata.yaml` exists.

## Ground truth

The bag contains ground truth (`/odom_sim` and the `odom -> base_link`
transform). It is used only by the evaluation script, never by the odometry.
