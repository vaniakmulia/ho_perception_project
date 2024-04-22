import numpy as np

def compute_rigid_transformation(position_matrix_cam, position_matrix_robot):
    num_points = position_matrix_cam.shape[0]
    assert num_points == position_matrix_robot.shape[0], "Number of points must match"

    # Initialize lists to store T_cam and T_robot matrices
    T_cam_list = []
    T_robot_list = []

    for i in range(num_points):
        # Extract corresponding points
        point_cam = position_matrix_cam[i]
        point_robot = position_matrix_robot[i]

        # Compute the translation vectors between the points
        tvecs_cam = point_cam.flatten()
        tvecs_robot = point_robot.flatten()

        # Create a 4x4 homogeneous transformation matrix for the object pose w.r.t camera
        T_cam = np.eye(4)
        T_cam[:3, 3] = tvecs_cam
        T_cam_list.append(T_cam)

        # Create a 4x4 homogeneous transformation matrix for the object pose w.r.t robot
        T_robot = np.eye(4)
        T_robot[:3, 3] = tvecs_robot
        T_robot_list.append(T_robot)

    # Stack the T_cam and T_robot matrices into matrices A and B respectively
    A = np.vstack(T_robot_list)  # A is T_robot stacked
    B = np.vstack(T_cam_list)     # B is T_cam stacked


    # Use SVD method to compute the least squares solution for RTC
    U, Sigma, VT = np.linalg.svd(A, full_matrices=False)
    Sigma_plus = np.diag(1 / Sigma)
    LS = VT.T @ Sigma_plus @ U.T @ B

    # Reshape the least squares solution to obtain the transformation matrix
    RTC = LS.reshape((4, 4))

    # Normalize the matrix to ensure the last element of the last row is 1
    RTC /= RTC[3, 3]

    # Explicitly set the last row to [0 0 0 1]
    RTC[3, :] = [0, 0, 0, 1]

    return RTC

# Example usage
position_matrix_cam = np.load("all_tvecs.npy")  # Assuming you have this file with camera positions
print("cam matrix:",position_matrix_cam)
position_matrix_robot = np.array([np.array([4, 9, 0]).reshape((3, 1)),
                                  np.array([6, 9, 0]).reshape((3, 1)),
                                  np.array([8, 9, 0]).reshape((3, 1)),
                                  np.array([4, 11, 0]).reshape((3, 1)),
                                  np.array([6, 11, 0]).reshape((3, 1)),
                                  np.array([8, 11, 0]).reshape((3, 1)),
                                  np.array([4, 13, 0]).reshape((3, 1)),
                                  np.array([6, 13, 0]).reshape((3, 1)),
                                  np.array([8, 13, 0]).reshape((3, 1))])

print("robot position:", position_matrix_robot)

RTC = compute_rigid_transformation(position_matrix_cam, position_matrix_robot)
print("Rigid transformation matrix RTC:")
print(RTC)
