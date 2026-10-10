"""Write the ground-truth trajectory to cache/gt.csv. Evaluation only.

I kept this apart from extract.py on purpose. The odometry code only ever
loads sensors.npz, and the ground truth sits in its own file that only
evaluate.py opens. That way it is easy to check that no algorithm touches it.

The bag has the ground truth twice: as /odom_sim messages and as the
odom -> base_link transform on /tf. I first used /odom_sim because it is
simpler to read. Then I compared the two and found they are not the same,
the transform is exactly one sample (50 ms) behind /odom_sim. The assignment
names the transform as the ground truth, so that is the one I read here.
"""
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)

GT_TOPIC = "/tf"
GT_PARENT = "odom"
GT_CHILD = "base_link"


def main(bag_dir, out_path):
    rows = []
    with AnyReader([Path(bag_dir)], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections if c.topic == GT_TOPIC]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            # /tf carries lots of transforms (wheels, lift, ...). I only
            # want the one from odom to base_link.
            for tf in msg.transforms:
                if tf.header.frame_id != GT_PARENT or tf.child_frame_id != GT_CHILD:
                    continue
                p, q = tf.transform.translation, tf.transform.rotation
                # yaw out of the quaternion. The robot stays flat on the
                # floor, so roll and pitch are zero and yaw is all I need.
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y * q.y + q.z * q.z))
                t = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
                rows.append([t, p.x, p.y, np.degrees(yaw)])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # same columns as the result files (t, x, y, yaw in degrees) so
    # evaluate.py can read both with one loader. 9 decimals keeps the
    # timestamps exact enough to pair rows by time later.
    np.savetxt(out_path, rows, delimiter=",", header="t,x,y,yaw_deg",
               comments="", fmt="%.9f")
    print(f"{len(rows)} ground-truth poses -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
