#### Calculate Essential Matrix and Decompose it into Rotation and Translation #### 


import numpy as np
import cv2

def normalize_points(keypoints, K):
    """
    Normalize the keypoints using the camera intrinsics matrix K.
    
    Args:
    - keypoints (list): List of keypoints (in pixel coordinates) for each frame
    - K (ndarray): Camera intrinsics matrix
    
    Returns:
    - normalized_points (ndarray): Normalized keypoints in camera coordinate system
    """
    normalized_points = []
    for kp in keypoints:
        if len(kp) == 0:
            normalized_points.append(np.array([]))  # Handle empty keypoints case
            continue

        # Convert keypoints from keypoint objects to numpy array
        points = np.array([k.pt for k in kp], dtype=np.float32)

        # Convert to homogeneous coordinates (add a row of ones)
        ones = np.ones((points.shape[0], 1), dtype=np.float32)  # Shape: (N, 1)
        points_homogeneous = np.hstack((points, ones))  # Shape: (N, 3)

        # Normalize by multiplying with the inverse of K
        normalized_homogeneous = np.dot(np.linalg.inv(K), points_homogeneous.T).T  # Shape: (N, 3)

        # Extract the first two normalized coordinates (divide by last row)
        normalized_coords = normalized_homogeneous[:, :2] / normalized_homogeneous[:, 2:]
        
        # # Normalize points by the camera matrix
        normalized_points.append(normalized_coords)

    # print(f"Normalized points: {normalized_points}")
    print(np.shape(normalized_points))
    return normalized_points


def calculate_essential_matrix(normalized_points_a, normalized_points_b):
    """
    Calculate the Essential Matrix using matched keypoints.
    
    Args:
    - normalized_points_a (list): Normalized keypoints from video_a
    - normalized_points_b (list): Normalized keypoints from video_b
    
    Returns:
    - essential_matrices (list): List of Essential Matrices for each frame pair
    """
    essential_matrices = []
    
    for pts_a, pts_b in zip(normalized_points_a, normalized_points_b):

        # if len(pts_a) != len(pts_b):  
        #     print(f"Mismatch in keypoints: {len(pts_a)} vs {len(pts_b)}")
        #     continue  # Skip this frame pair if points don't match

        # Compute the Essential Matrix using RANSAC to filter out outliers
        E, mask = cv2.findEssentialMat(pts_a, pts_b, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
        essential_matrices.append(E)

    return essential_matrices

def decompose_essential_matrix(E, points1, points2, K):
    """
    Decompose the Essential Matrix into rotation and translation components.
    
    Args:
    - E (ndarray): Essential matrix
    
    Returns:
    - rotation (ndarray): Rotation matrix (3x3)
    - translation (ndarray): Translation vector (3x1)
    """
    # Decompose the Essential Matrix to get rotation (R) and translation (t)
    # using Singular Value Decomposition (SVD)
    _, R, t, _ = cv2.recoverPose(E, points1, points2, K)
    
    return R, t

def compute_translation_magnitude(translations,position_deltas):
    """
    Compute the magnitude of each translation vector.
    
    Args:
    - translations (list): List of 3x1 translation vectors
    - position_deltas (list): Ground-truth position_delta values from poses_b.jsonl
    
    Returns:
    - translation_magnitudes (list): Magnitude of each translation vector
    """
    translation_magnitudes = [np.linalg.norm(t) for t in translations]    

    scaling_factors = [pd / em if em > 0 else 1.0 for pd, em in zip(position_deltas, translation_magnitudes)]

    scaled_translations = [s * t for s, t in zip(scaling_factors, translations)]

    estimated_magnitudes = [np.linalg.norm(t) for t in scaled_translations]    
    translation_errors = [abs(em - pd) for em, pd in zip(estimated_magnitudes, position_deltas)]

    return scaled_translations, estimated_magnitudes, scaling_factors, translation_errors