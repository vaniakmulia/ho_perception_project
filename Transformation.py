############################################
############This file is Not in use now
############################################
import numpy as np

def compute_rigid_transformation(position_matrix_cam, position_matrix_robot):
    # Assume the first point corresponds to the origin in both coordinate systems
    # Compute the translation vector between the first points
    tvecs_cam = position_matrix_cam[0]
    tvecs_robot = position_matrix_robot[0]

    # Create a 4x4 homogeneous transformation matrix for the object pose w.r.t camera
    T_cam = np.eye(4)
    T_cam[:3, 3] = tvecs_cam.flatten()  # Flatten tvecs_cam to ensure correct shape
    
    # Create a 4x4 homogeneous transformation matrix for the object pose w.r.t robot
    T_robot = np.eye(4)
    T_robot[:3, 3] = tvecs_robot.flatten()  # Flatten tvecs_robot to ensure correct shape
    
    # Invert T_robot to get T_robot_inv
    T_robot_inv = np.linalg.inv(T_robot)

    # Compute RTC = T_robot_inv * T_cam
    RTC = np.dot(T_robot_inv, T_cam)
    
    return RTC

# Example position matrices representing object pose w.r.t camera and robot
position_matrix_cam = np.load("all_tvecs.npy")  # Assuming you have this file with camera positions
position_matrix_robot = np.array([np.array([4,9,0]).reshape((3,1)),
                                  np.array([6,9,0]).reshape((3,1)),
                                  np.array([8,9,0]).reshape((3,1)),
                                  np.array([4,11,0]).reshape((3,1)),
                                  np.array([6,11,0]).reshape((3,1)),
                                  np.array([8,11,0]).reshape((3,1)),
                                  np.array([4,13,0]).reshape((3,1)),
                                  np.array([6,13,0]).reshape((3,1)),
                                  np.array([8,13,0]).reshape((3,1))])

print(position_matrix_cam.shape)
print(position_matrix_robot.shape)

RTC = compute_rigid_transformation(position_matrix_cam, position_matrix_robot)
print("Rigid transformation matrix RTC:")
print(RTC)
