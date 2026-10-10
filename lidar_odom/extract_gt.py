"""Write the ground-truth trajectory to cache/gt.csv. Evaluation only.

I kept this apart from extract.py on purpose. The odometry code only ever
loads sensors.npz, and the ground truth sits in its own file that only
evaluate.py opens. That way it is easy to check that no algorithm touches it.
"""
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)

GT_TOPIC = "/odom_sim"


def main(bag_dir, out_path):
    rows = []
    with AnyReader([Path(bag_dir)], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections if c.topic == GT_TOPIC]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            # yaw out of the quaternion. The robot stays flat on the floor,
            # so roll and pitch are zero and yaw is all I need.
            yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            rows.append([t, p.x, p.y, yaw])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # same columns as the result files (t, x, y, yaw) so evaluate.py can
    # read both with one loader. 9 decimals keeps the timestamps exact
    # enough to pair rows by time later.
    np.savetxt(out_path, rows, delimiter=",", header="t,x,y,yaw",
               comments="", fmt="%.9f")
    print(f"{len(rows)} ground-truth poses -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
