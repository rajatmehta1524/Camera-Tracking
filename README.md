# 📸 Camera Trajectory Estimation using Essential Matrix Decomposition

Estimate the 6-DOF pose of a moving camera from monocular video sequences by extracting SIFT features, computing the essential matrix, recovering relative poses, and chaining them into a trajectory. Align and correct the resulting poses to produce smooth camera motion paths.

---

## 🖼️ Demo

| Input Frame | Feature Matching | Trajectory Output |
|-------------|------------------|--------------------|
| ![Frame](outputs/video_screenshot.png) | ![Matches](outputs/matched_features.png) | ![Trajectory](outputs/trajectory_plot.png) |

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Pipeline](#-pipeline)
- [Features](#-features)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Sample Outputs](#-sample-outputs)
- [Results](#-results)
- [Limitations & Future Work](#-limitations--future-work)
- [References](#-references)
- [License](#-license)

---

## 🧠 Overview

This project estimates frame-by-frame camera poses from monocular video using geometric methods. By detecting and matching features between consecutive frames and computing the essential matrix, we recover relative camera poses. The final trajectory is built by chaining these poses and applying corrections.

---

## 🔧 Pipeline

```
Video → Frame Extraction → SIFT Detection → Feature Matching →  
→ Essential Matrix Computation → Pose Recovery (R, t) →  
→ Cheirality Check → Pose Chaining → Trajectory Correction → Visualization
```

---

## ✨ Features

- 📌 SIFT-based feature detection and FLANN-based matching
- 🔍 Essential matrix estimation via 5-point algorithm
- 📈 Pose recovery using `cv2.recoverPose` with cheirality validation
- 🔄 Trajectory construction via chained transformations
- 🧭 Trajectory correction for pose alignment
- 🖼️ Output visualization with trajectory plots and feature match images

---

## 📁 Folder Structure

```
camera-trajectory-estimation/
├── main.py
├── EssentialMatrices.py
├── Feature_Detector_Matching.py
├── parsing.py
├── data/
│   ├── video_a.mp4
│   ├── video_b.mp4
│   ├── poses_a.jsonl
│   └── poses_b.jsonl
└── outputs/
    ├── video_screenshot.png
    ├── matched_features.png
    └── trajectory_plot.png
```

---

## 💻 Installation

Install Python dependencies:

```bash
pip install opencv-python numpy matplotlib tqdm
```

Ensure the videos and pose files are inside the `data/` folder.

---

## ▶️ How to Run

Run the main pipeline:

```bash
python main.py
```

This will:
- Parse the videos and pose files
- Detect & match features
- Compute and recover relative poses
- Chain and correct trajectories
- Save the output poses and visualizations to `outputs/`

---

## 🖼️ Sample Outputs

- **Matched Features**  
  ![Matches](outputs/matched_features.png)

- **Camera Trajectory Plot**  
  ![Trajectory](outputs/trajectory_plot.png)

- **Sample Frame**  
  ![Frame](outputs/video_screenshot.png)

---

## 📊 Results

- Estimated poses are saved into `poses_b.jsonl` with updated `transform` matrices
- The final trajectory visually aligns with the original (ground truth-like) path
- Pose chaining and correction reduce drift and inconsistencies

---

## ⚠️ Limitations & Future Work

- No absolute scale recovery (translation is up to scale)
- Assumes mostly static environments and similar camera motion
- May be improved with loop closure detection or global optimization (SLAM)
- Extendable to stereo or depth-based camera inputs

---

## 📚 References

- [Multiple View Geometry - Hartley & Zisserman](https://www.robots.ox.ac.uk/~vgg/hzbook/)
- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [Cheirality Condition](https://en.wikipedia.org/wiki/Cheirality)

---

