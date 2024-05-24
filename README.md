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
This code is responsible for camera calibration. It uses OpenCV and Numpy libraries to perform calibration using ArUco markers. The code captures frames from the webcam for ArUco marker detection. Detected marker data is used for calibration. Lastly, the camera-calibrated parameters are saved to a YAML file.
### 6. `Camera_pose.py`
camera
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
This fil contains the camera calibration parameters obtained from the camera calibration file.

### Getting Started

### Prerequisites
Required libraries  
OpenCV, version: 4.9.0  
NumPy,  version: 1.24.4  
Matplotlib, version: 3.7.5  

### How to run the Project
The project can be run in two ways:
## Through VS Code
Open the project folder into VS code and run the file.

## Through Ubuntu Terminal
Navigate to the project directory

Open Terminal

### Command to run camera calibration file
```bash
python3 Camera_calibration.py
```
## Command to run the robot calibration file
```bash
python3 Robot_calibration.py
```


## Command to run camera pose file
```bash
python3 Camera_pose.py
```

### Command to run object pose estimation file
```bash
python3 Object_Pose_Estimation.py
```

