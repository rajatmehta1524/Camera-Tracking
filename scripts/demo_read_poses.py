#%%
import os

import numpy as np
from matplotlib import pyplot as plt
from parsing import read_camera_data

fname_cam_poses_a = os.path.join(
    os.path.split(os.path.dirname(__file__))[0], "data/poses_a.jsonl"
)
fname_cam_poses_b = os.path.join(
    os.path.split(os.path.dirname(__file__))[0], "data/poses_b.jsonl"
)

# Load the pose files
poses_a, position_deltas_a = read_camera_data(fname_cam_poses_a)
poses_b, position_deltas_b = read_camera_data(fname_cam_poses_b)

# NOTE: poses_b are all identity. This is what you need to solve for.
# However, position_deltas_b contains the correct position deltas.

print(f"Read {len(poses_a)} camera poses from {fname_cam_poses_a}")
print(f"Read {len(poses_b)} camera poses from {fname_cam_poses_b}")

vecs_a = np.vstack([p[:3, 3] for p in poses_a])
vecs_b = np.vstack([p[:3, 3] for p in poses_b])

# NOTE: Once you solve for poses_b the track you get should look
# very similar to the poses_a track plotted below.
plt.figure()
plt.plot(vecs_a[:, 0], vecs_a[:, 1], label="poses_a")
plt.plot(vecs_b[:, 0], vecs_b[:, 1], label="poses_b")
plt.axis("equal")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()

plt.figure()
plt.plot(vecs_a[:, 0], vecs_a[:, 2], label="poses_a")
plt.plot(vecs_b[:, 0], vecs_b[:, 2], label="poses_b")
plt.axis("equal")
plt.xlabel("X Position")
plt.ylabel("Z Position")
plt.legend()

plt.figure()
plt.plot(position_deltas_a, label="poses_a")
plt.plot(position_deltas_b, label="poses_b")
plt.xlabel("Index")
plt.ylabel("Position Delta")
plt.legend()

plt.show()

# %%
