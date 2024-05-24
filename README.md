# HO-Perception-Project

## Overview
This project focuses on [brief description of the project's main goal, e.g., object detection and pose estimation using computer vision techniques].

## Contents

### 1. `images/`
- **Purpose:** Directory containing images used for testing and calibration.

### 2. `All_Marker_ids.npy`
- **Purpose:** Stores all marker IDs used in the project.

### 3. `All_rvecs.npy`
- **Purpose:** Stores rotation vectors for all markers.

### 4. `All_tvecs.npy`
- **Purpose:** Stores translation vectors for all markers.

### 5. `Camera_calibration.py`
- **Purpose:** Script for camera calibration.
- **Functions:**
  - Calibrates the camera using images of a known calibration pattern.

### 6. `Camera_pose.py`
- **Purpose:** Determines the camera's pose relative to the observed scene.
- **Functions:**
  - Estimates the camera's position and orientation.

### 7. `Object_Pose_Estimation.py`
- **Purpose:** Estimates the pose of objects in the scene.
- **Functions:**
  - Captures and computes the 3D pose of objects using detected markers.

### 8. `Object_detection.py`
- **Purpose:** Detects objects within images.
- **Functions:**
  - Implements algorithms to identify and locate objects.

### 9. `Robot_calibration.py`
- **Purpose:** Calibrates the robot's coordinate system with the camera's.
- **Functions:**
  - Aligns the robot's positional data with the camera's data.

### 10. `calibration.yaml`
- **Purpose:** Configuration file storing camera calibration parameters.

## Getting Started

### Prerequisites
Required libraries 
OpenCV, version: 4.9.0
NumPy,  version : 1.24.4
Matplotlib version: 3.7.5

### How to run the Project
The project can be run in two ways:
## Through VS Code
Open the project folder into VS code and run the file.

## Through Ubuntu Terminal
Navigate to the project directory

Open Terminal

# Command to run camera calibration file
python3 Camera_calibration.py

# Command to run object estimation file
python3 Object_Pose_Estimation.py

# Command to run camera pose file
python3 Camera_pose.py

# Command to run the robot calibration file
python3 Robot_calibration.py

