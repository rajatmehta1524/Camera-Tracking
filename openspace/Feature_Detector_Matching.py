## Feature Detector and Matching

import cv2
import numpy as np

def extract_features(video_path, orb):
    """
    Extract keypoints and descriptors from video.
    
    Args:
    - video_path (str): path to the video file
    
    Returns:
    - keypoints_list (list): list of keypoints for each frame
    - descriptors_list (list): list of descriptors for each frame
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    keypoints_list = []
    descriptors_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale for ORB
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ORB keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    
    cap.release()
    
    return keypoints_list, descriptors_list


def match_features(descriptors_a, descriptors_b):
    """
    Match features between descriptors of two videos.
    
    Args:
    - descriptors_a (list): list of descriptors for video_a
    - descriptors_b (list): list of descriptors for video_b
    
    Returns:
    - matches_list (list): list of matches between frames
    """
    # Initialize Brute Force Matcher (BFMatcher) with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    matches_list = []
    
    # Iterate over frames and match features between the two videos
    for desc_a, desc_b in zip(descriptors_a, descriptors_b):
        if desc_a is not None and desc_b is not None:
            # Match descriptors
            matches = bf.match(desc_a, desc_b)
            
            # Sort matches based on distance (best matches first)
            matches = sorted(matches, key = lambda x: x.distance)
            
            matches_list.append(matches)
    
    return matches_list

def visualize_matches(video_a_path, video_b_path, matches_list, keypoints_a, keypoints_b):
    cap_a = cv2.VideoCapture(video_a_path)
    cap_b = cv2.VideoCapture(video_b_path)
    
    while cap_a.isOpened() and cap_b.isOpened():
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        
        if not ret_a or not ret_b:
            break
        
        for matches, kp_a, kp_b in zip(matches_list, keypoints_a, keypoints_b):
            # Draw matches between keypoints in both frames
            img_matches = cv2.drawMatches(frame_a, kp_a, frame_b, kp_b, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('Matches', img_matches)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    cap_a.release()
    cap_b.release()
    cv2.destroyAllWindows()