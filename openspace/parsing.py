from typing import List
import numpy as np
import json
import os


def read_camera_data(fname: str, return_json: bool = False) -> List[np.ndarray]:
    assert fname.endswith(".jsonl"), f"Unsupported file format: {fname}"
    assert os.path.exists(fname), f"Input file does not exist: {fname}"

    poses = []
    position_deltas = []
    with open(fname, "r") as file:
        for line in file:
            data = json.loads(line)
            pose = np.array(data["transform"]).reshape(4, 4).T
            # Note that pose is now a 4x4 transform where
            # rotation_matrix = pose[:3, :3]
            # translation_vec = pose[:3, 3]
            poses.append(pose)
            position_deltas.append(data["position_delta"])

    return poses, position_deltas