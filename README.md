# HO-Perception-Project

## Overview
This project focuses on developing a real-time 2d pose estimation system for robots and its applications in automated parts-picking operations. The project involves camera calibration, robot calibration, object detection and pose estimation algorithms.
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

## Contents

### 1. `Camera_calibration.py`
This code is responsible for camera calibration. It uses OpenCV and Numpy libraries to perform calibration using ArUco markers. The code captures frames from the webcam for ArUco marker detection. Detected marker data is used for calibration. Lastly, the camera-calibrated parameters are saved to a YAML file.

### 2. `Robot_calibration.py`
This code is used to calculate the  rigid transformation matrix between two sets of 3D coordinates, one  observed by a camera and another by a robot. Here the robot points are defined as well as **All_tvecs.npy** obtained from the camera pose is used for loading camera data. The result of this code is a transformation matrix. This matrix allows for the conversion of points from the camera's coordinate system to the robot's coordinate system.

### 3. `Camera_pose.py`

This code uses the ArUco module to implement augmented reality for obtaining camera pose. It opens the camera feed, detects ArUco markers, and estimates their poses using the **solvePnP()** function. Lastly, the code sorts and saves the obtained tvecs (translation vector) and rvecs (rotation vector) corresponding to the ArUco marker IDs.

### 4. `Object_Pose_Estimation.py`
This code is responsible for estimating the pose of the detected object in the robot’s coordinate system based on its observation from the camera. Here The rigid transformation matrix (RTC) obtained from robot calibration and the camera matrix obtained from camera calibration are used. The program captures the video frame  and processes them to detect contours, and for each contour of a significant area, calculates its orientation and centroid using Principal component analysis (PCA). The pose of the object is transformed into the camera coordinate system and then into the robot coordinate system. The object's pose and major axis are updated and visualized in a 3D plot using matplotlib library. The script also draws axes on the original frame for visualization.

### 5. `images`
This Directory contains the images used for the camera calibration.

### 6. `calibration.yaml`
This file contains the camera calibration parameters obtained from the camera calibration file.

### 7. `All_Marker_ids.npy`
This file contains the ArUco marker IDs used associated with the tvecs and rvecs from the camera pose program.

### 8. `All_rvecs.npy`
This file contains the rotation vectors obtained from the camera pose program.

### 9. `All_tvecs.npy`
This file contains the translation vectors obtained from the camera pose program.

