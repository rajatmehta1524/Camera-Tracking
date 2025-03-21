#%%
import os
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
from parsing import read_camera_data
from Feature_Detector_Matching import extract_features, match_features, visualize_matches
from EssentialMatrices import normalize_points, calculate_essential_matrix, decompose_essential_matrix, compute_translation_magnitude
import jsonlines

def read_poses_from_jsonl(file_path):
    poses = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            # Extract the transformation matrix (flattened)
            transform = np.array(obj['transform'], dtype=np.float32).reshape(4, 4)
            # Extract the position delta (translation magnitude)
            position_delta = obj['position_delta']
            poses.append({'transform': transform, 'position_delta': position_delta})
    return poses

def accumulate_poses(rots, trans, initial_pose=np.eye(4)):
    """
    Compute the full camera trajectory using cumulative transformations.
    
    Args:
    - rots (list): List of 3x3 rotation matrices
    - trans (list): List of 3x1 translation vectors
    - initial_pose (numpy array): Initial pose (4x4 identity matrix)
    
    Returns:
    - poses (list): List of 4x4 camera poses
    """
    poses = [initial_pose]
    
    for R, t in zip(rots, trans):
        # Convert R and t into a 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()  # Ensure t is a flat array
        
        # Compute the new pose by multiplying the previous pose
        new_pose = poses[-1] @ T
        poses.append(new_pose)
    
    return poses

def plot_trajectories(poses_a, poses_b):
    """
    Plot the camera trajectories for poses_a and poses_b.
    
    Args:
    - poses_a (list): List of 4x4 matrices for camera A
    - poses_b (list): List of 4x4 matrices for camera B
    """
    traj_a = np.array([pose[:3, 3] for pose in poses_a])
    traj_b = np.array([pose[:3, 3] for pose in poses_b])

    plt.figure(figsize=(8, 6))
    plt.plot(traj_a[:, 0], traj_a[:, 2], label="Camera A", marker="o")
    plt.plot(traj_b[:, 0], traj_b[:, 2], label="Camera B", marker="x")
    plt.xlabel("X Position")
    plt.ylabel("Z Position")
    plt.title("Camera Trajectories")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(traj_a[:, 1], traj_a[:, 2], label="Camera A", marker="o")
    plt.plot(traj_b[:, 1], traj_b[:, 2], label="Camera B", marker="x")
    plt.xlabel("Y Position")
    plt.ylabel("Z Position")
    plt.title("Camera Trajectories")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(traj_a[:, 0], traj_a[:, 1], label="Camera A", marker="o")
    plt.plot(traj_b[:, 0], traj_b[:, 1], label="Camera B", marker="x")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Camera Trajectories")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":

    fname_cam_poses_a = os.path.join(os.path.split(os.path.dirname(__file__))[0], "data/poses_a.jsonl")
    fname_cam_poses_b = os.path.join(os.path.split(os.path.dirname(__file__))[0], "data/poses_b.jsonl")
    fname_cam_poses_b_updated = os.path.join(os.path.split(os.path.dirname(__file__))[0], "data/video_b_updated.jsonl")


    poses_b_data = read_poses_from_jsonl(fname_cam_poses_b)

    # Load the pose files
    poses_a, position_deltas_a = read_camera_data(fname_cam_poses_a)
    poses_b, position_deltas_b = read_camera_data(fname_cam_poses_b)
    poses_b_updated, position_deltas_b_updated = read_camera_data(fname_cam_poses_b_updated)

    # # Initialize ORB detector
    # orb = cv2.ORB_create()

    # # Example usage for video_a and video_b
    # video_a_path = '/home/rj/Downloads/OpenSpace_AI_Home_Assignment/openspace-homework/data/video_a.mp4'
    # video_b_path = '/home/rj/Downloads/OpenSpace_AI_Home_Assignment/openspace-homework/data/video_b.mp4'
    # # Extract features from both videos
    # video_a_keypoints, video_a_descriptors = extract_features(video_a_path, orb)
    # video_b_keypoints, video_b_descriptors = extract_features(video_b_path, orb)

    # # Match features between frames of video_a and video_b
    # matches_list = match_features(video_a_descriptors, video_b_descriptors)
    # # Visualize the matches
    # # visualize_matches(video_a_path, video_b_path, matches_list, video_a_keypoints, video_b_keypoints)


    # # Camera intrinsic matrix
    # K = np.array([[1527.894, 0, 962.880],
    #             [0, 1527.894, 721.309],
    #             [0, 0, 1]])

    # # Normalize points for video_a and video_b
    # video_a_normalized = normalize_points(video_a_keypoints, K)
    # video_b_normalized = normalize_points(video_b_keypoints, K)

    # # Calculate the Essential Matrix
    # essential_matrices = calculate_essential_matrix(video_a_normalized, video_b_normalized)
    # # Decompose the Essential Matrix for each frame pair
    # rotation_matrices = []
    # translation_vectors = []

    # for i, E in enumerate(essential_matrices):
    #     R, t = decompose_essential_matrix(E, video_a_normalized[i], video_b_normalized[i], K)
    #     rotation_matrices.append(R)
    #     translation_vectors.append(t)

    # # Load position_delta values from poses_b.jsonl
    # position_deltas = [entry["position_delta"] for entry in poses_b_data]  # Extract ground-truth values
    # # Compute translation magnitudes
    # scaled_translations, estimated_magnitudes, scaling_factors, translation_errors = compute_translation_magnitude(translation_vectors, position_deltas)
    # # Print mean error
    # print(f"Mean translation error: {np.mean(translation_errors):.6f}")


    # # Initialize pose list
    # poses_b_transforms = []
    # # Start with identity matrix for the first frame
    # initial_pose = np.eye(4)
    # poses_b_transforms.append(initial_pose)
    # # Compute full camera poses

    # poses_b_transforms = accumulate_poses(rotation_matrices, scaled_translations)

    # # Convert transformation matrices to JSON format
    # for i, entry in enumerate(poses_b_data):
    #     entry["transform"] = poses_b_transforms[i].flatten().tolist()

    # # Save the updated poses_b.jsonl file
    # output_file = "poses_b_updated.jsonl"
    # with open(output_file, "w") as f:
    #     for entry in poses_b_data:
    #         f.write(json.dumps(entry) + "\n")

    # print(f"Updated poses_b.jsonl saved as {output_file}")

    poses_a_transforms = poses_a

    # Visualize trajectories
    # plot_trajectories(poses_a_transforms, poses_b_transforms)
    plot_trajectories(poses_a_transforms, poses_b_updated)
