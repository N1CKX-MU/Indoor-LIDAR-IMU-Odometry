"""
Read the sensor topics from the bag into cache/sensors.npz

Ground Truth (/odom_sim and the odom -> base_link transform on /tf) is never opened here.
"""
import sys 
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)

LIDAR_TOPIC = "/livox/amr/lidar"
IMU_TOPIC = "/livox/amr/imu"
TF_STATIC_TOPIC = "/tf_static"
BASE_FRAME = "base_link"
G = 9.80665  # the Livox IMU reports acceleration in g

def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9

def to_matrix(transform):
    """
    geometry_msgs/Transform -> 4x4 homogeneous matrix.
    """
    t, q = transform.translation, transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = [t.x, t.y, t.z]
    return T

def chain_to_base(static_tf, frame):
    """
    Compose static transforms from BASE_FRAME down to `frame`.
    """
    T = np.eye(4)
    while frame != BASE_FRAME:
        parent, T_parent_child = static_tf[frame]
        T = T_parent_child @ T
        frame = parent
    return T

def main(bag_dir, out_path):
    scan_t, scans, imu_t, gyro, acc = [], [], [], [], []
    static_tf = {}  # child frame -> (parent frame, T_parent_child)
    lidar_frame = None

    with AnyReader([Path(bag_dir)], default_typestore=TYPESTORE) as reader:
        wanted = (LIDAR_TOPIC, IMU_TOPIC, TF_STATIC_TOPIC)
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == LIDAR_TOPIC:
                lidar_frame = msg.header.frame_id
                xyz = np.frombuffer(msg.data, np.float32).reshape(-1, 3)
                scan_t.append(stamp(msg.header))
                scans.append(xyz[np.isfinite(xyz).all(axis=1)])
            elif conn.topic == IMU_TOPIC:
                w, a = msg.angular_velocity, msg.linear_acceleration
                imu_t.append(stamp(msg.header))
                gyro.append([w.x, w.y, w.z])
                acc.append([a.x * G, a.y * G, a.z * G])
            else:
                for tf in msg.transforms:
                    static_tf[tf.child_frame_id] = (
                        tf.header.frame_id, to_matrix(tf.transform))

    # Scans have different sizes, so store one long point array plus offsets:
    # scan i is scan_xyz[scan_start[i]:scan_start[i + 1]].
    scan_start = np.cumsum([0] + [len(s) for s in scans])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        scan_t=np.array(scan_t),
        scan_start=scan_start,
        scan_xyz=np.concatenate(scans),
        imu_t=np.array(imu_t),
        imu_gyro=np.array(gyro),
        imu_acc=np.array(acc),
        T_base_lidar=chain_to_base(static_tf, lidar_frame),
    )
    print(f"{len(scans)} scans, {scan_start[-1]} points, {len(imu_t)} imu samples")
    print(f"lidar {1 / np.median(np.diff(scan_t)):.1f} Hz, "
          f"imu {1 / np.median(np.diff(imu_t)):.1f} Hz")
    print("T_base_lidar =\n", chain_to_base(static_tf, lidar_frame).round(4))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])