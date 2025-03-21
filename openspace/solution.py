import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_frames(video_path, max_frames=None):
    """Extract frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frames.append(frame)
        count += 1
        
        if max_frames is not None and count >= max_frames:
            break
    
    cap.release()
    return frames

def detect_features(frame):
    """Detect features in a frame using ORB."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    """Match features between two sets of descriptors."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def get_matched_points(kp1, kp2, matches, max_matches=100):
    """Get matching point coordinates from keypoints and matches."""
    # Use only the best matches
    good_matches = matches[:min(max_matches, len(matches))]
    
    # Extract the coordinates of matched points
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    
    return points1, points2

# def extract_camera_positions(poses):
#     """Extract camera positions from a list of pose transforms."""
#     positions = []
    
#     for pose in poses:
#         # Reshape transform to 4x4 matrix
#         T = np.array(pose['transform']).reshape(4, 4)
        
#         # Camera position is the negative of rotation transpose times translation
#         # Or simply the inverse transform applied to the origin
#         R = T[:3, :3]
#         t = T[:3, 3]
        
#         # Camera center C = -R^T * t
#         C = -np.dot(R.T, t)
        
#         positions.append(C)
    
#     return np.array(positions)

def plot_trajectories(positions_a, positions_b, output_dir="plots"):
    """Plot the camera trajectories in different projections."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    pose_a = []
    pose_b = []
    for pose in positions_a:
        pose_a.append(np.array(pose['transform']).reshape(4, 4))

    for pose in positions_b:    
        pose_b.append(np.array(pose['transform']).reshape(4, 4))


    traj_a = np.array([pose[:3, 3] for pose in pose_a])
    traj_b = np.array([pose[:3, 3] for pose in pose_b])

    # print(f"Trajectory A: {traj_a}, Trajectory B: {traj_b}")

    # Extract x, y, z coordinates
    x_a, y_a, z_a = traj_a[:, 0], traj_a[:, 1], traj_a[:, 2]
    x_b, y_b, z_b = traj_b[:, 0], traj_b[:, 1], traj_b[:, 2]
    
    # Plot X-Y projection
    plt.figure(figsize=(10, 8))
    plt.plot(x_a, y_a, 'b-', label='Trajectory A')
    plt.plot(x_b, y_b, 'r-', label='Trajectory B')
    plt.scatter(x_a[0], y_a[0], c='blue', marker='o', s=100, label='Start A')
    plt.scatter(x_b[0], y_b[0], c='red', marker='o', s=100, label='Start B')
    plt.title('Camera Trajectories: X-Y Projection')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'trajectory_xy.png'))
    plt.show()
    
    # Plot X-Z projection
    plt.figure(figsize=(10, 8))
    plt.plot(x_a, z_a, 'b-', label='Trajectory A')
    plt.plot(x_b, z_b, 'r-', label='Trajectory B')
    plt.scatter(x_a[0], z_a[0], c='blue', marker='o', s=100, label='Start A')
    plt.scatter(x_b[0], z_b[0], c='red', marker='o', s=100, label='Start B')
    plt.title('Camera Trajectories: X-Z Projection')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'trajectory_xz.png'))
    plt.show() 

    # Plot Y-Z projection
    plt.figure(figsize=(10, 8))
    plt.plot(y_a, z_a, 'b-', label='Trajectory A')
    plt.plot(y_b, z_b, 'r-', label='Trajectory B')
    plt.scatter(y_a[0], z_a[0], c='blue', marker='o', s=100, label='Start A')
    plt.scatter(y_b[0], z_b[0], c='red', marker='o', s=100, label='Start B')
    plt.title('Camera Trajectories: Y-Z Projection')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'trajectory_yz.png'))
    plt.show()

    # 3D Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_a, y_a, z_a, 'b-', label='Trajectory A')
    ax.plot(x_b, y_b, z_b, 'r-', label='Trajectory B')
    ax.scatter(x_a[0], y_a[0], z_a[0], c='blue', marker='o', s=100, label='Start A')
    ax.scatter(x_b[0], y_b[0], z_b[0], c='red', marker='o', s=100, label='Start B')
    ax.set_title('Camera Trajectories: 3D View')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'trajectory_3d.png'))
    plt.show() 

    # plt.close('all')
    print(f"Trajectory plots saved to {output_dir}")

def main():
    # File paths
    video_a_path = '/home/rj/Downloads/OpenSpace_AI_Home_Assignment/openspace-homework/data/video_a.mp4'
    video_b_path = '/home/rj/Downloads/OpenSpace_AI_Home_Assignment/openspace-homework/data/video_b.mp4'
    jsonl_a_path = '/home/rj/Downloads/OpenSpace_AI_Home_Assignment/openspace-homework/data/poses_a.jsonl'
    jsonl_b_path = '/home/rj/Downloads/OpenSpace_AI_Home_Assignment/openspace-homework/data/poses_b.jsonl'
    output_path = '/home/rj/Downloads/OpenSpace_AI_Home_Assignment/openspace-homework/data/video_b_updated.jsonl'
    
    # Camera intrinsics (you should replace these with actual values)
    # Assuming a standard HD camera if not available
    K = np.array([
        [1000, 0, 960],
        [0, 1000, 540],
        [0, 0, 1]
    ])
    
    # Load the JSONL files
    poses_a = []
    with open(jsonl_a_path, 'r') as f:
        for line in f:
            poses_a.append(json.loads(line))
    
    poses_b = []
    with open(jsonl_b_path, 'r') as f:
        for line in f:
            poses_b.append(json.loads(line))
    
    # Extract the first few frames from each video
    # We only need a few frames to establish the coordinate transformation
    frames_a = extract_frames(video_a_path, max_frames=5)
    frames_b = extract_frames(video_b_path, max_frames=5)
    
    if not frames_a or not frames_b:
        raise ValueError("Could not extract frames from videos")
    
    # Find corresponding features between first frames
    keypoints_a, descriptors_a = detect_features(frames_a[0])
    keypoints_b, descriptors_b = detect_features(frames_b[0])
    
    if descriptors_a is None or descriptors_b is None:
        raise ValueError("No features detected in one or both frames")
    
    matches = match_features(descriptors_a, descriptors_b)
    
    if len(matches) < 8:
        raise ValueError("Not enough matching features found")
    
    # Get matched point coordinates
    points_a, points_b = get_matched_points(keypoints_a, keypoints_b, matches)
    
    # Compute the essential matrix and recover the rotation and translation
    E, mask = cv2.findEssentialMat(points_a, points_b, K, cv2.RANSAC, 0.999, 1.0)
    _, R, t, _ = cv2.recoverPose(E, points_a, points_b, K, mask)
    
    # Create 4x4 transformation matrix from video A's coordinate system to video B's
    T_a_to_b = np.eye(4)
    T_a_to_b[:3, :3] = R
    T_a_to_b[:3, 3] = t.reshape(3)
    
    # Store original poses for trajectory visualization
    original_poses_b = [pose.copy() for pose in poses_b]
    
    # Apply to all poses in video B
    for i in range(len(poses_b)):
        if i < len(poses_a):
            # Get the transform from A
            T_a = np.array(poses_a[i]['transform']).reshape(4, 4)
            
            # Apply the global transformation
            T_b = T_a_to_b @ T_a
            
            # Update the transform in B, keep position_delta as is
            poses_b[i]['transform'] = T_b.flatten().tolist()
    
    # Extract camera positions for trajectory visualization
    # positions_a = extract_camera_positions(poses_a)
    # positions_b = extract_camera_positions(poses_b)
    
    # Plot trajectories
    plot_trajectories(poses_a, poses_b)
    
    # Save the updated poses_b
    with open(output_path, 'w') as f:
        for pose in poses_b:
            f.write(json.dumps(pose) + '\n')
    
    print(f"Updated poses saved to {output_path}")

if __name__ == "__main__":
    main()