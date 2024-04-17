import numpy as np
import cv2

def compute_rigid_transformation(rvecs, tvecs, T_robot):
    # Convert rotation vector to rotation matrix using Rodrigues formula
    R_cam = np.zeros((3, 3))
    cv2.Rodrigues(rvecs, R_cam)
    
    # Create a 4x4 homogeneous transformation matrix for the object pose w.r.t camera
    T_cam = np.eye(4)
    T_cam[:3, :3] = R_cam
    T_cam[:3, 3] = tvecs.flatten()  # Flatten tvecs to ensure correct shape
    
    # Invert T_robot to get T_robot_inv
    T_robot_inv = np.linalg.inv(T_robot)

    # Compute RTC = T_robot_inv * T_cam
    RTC = np.dot(T_robot_inv, T_cam)
    
    return RTC

# Example rotation vector and translation vector representing object pose w.r.t camera
rvecs = np.load("all_rvecs.npy")
tvecs = np.load("all_tvecs.npy")

print(tvecs.shape)
print(rvecs.shape)

# Example 3D position represented by a 3x3 matrix
position_matrix = np.array([[[4,9,0], [6,9,0], [8,9,0]],
                            [[4,11,0], [6,11,0], [8,11,0]],
                            [[4,13,0], [6,13,0], [8,13,0]]])

# Example axis-angle representation for robot positions
axis_angle = np.array([-2.47825913, 0., 1.23912957])

# Convert axis-angle representation to rotation matrix
R_matrix = cv2.Rodrigues(axis_angle)[0]

# Extract translation vector from the 3x3 position matrix
translation_vector = position_matrix[:3, 2]

# Create a 4x4 homogeneous transformation matrix for the robot
T_robot = np.eye(4)
T_robot[:3, :3] = R_matrix
T_robot[:3, 3] = translation_vector

RTC = compute_rigid_transformation(rvecs, tvecs, T_robot)
print("Rigid transformation matrix RTC:")
print(RTC)
