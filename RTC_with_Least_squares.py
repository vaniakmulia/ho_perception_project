import numpy as np

def compute_rigid_transformation(position_matrix_cam, position_matrix_robot):
    num_points = position_matrix_cam.shape[0]
    assert num_points == position_matrix_robot.shape[0], "Number of points must match"

    # Initialize a list to store transformation matrices for each pair of points
    transformations = []

    for i in range(num_points):
        # Extract corresponding points
        point_cam = position_matrix_cam[i]
        point_robot = position_matrix_robot[i]

        # Compute the translation vector between the points
        tvecs_cam = point_cam.flatten()
        tvecs_robot = point_robot.flatten()

        # Create a 4x4 homogeneous transformation matrix for the object pose w.r.t camera
        T_cam = np.eye(4)
        T_cam[:3, 3] = tvecs_cam

        # Create a 4x4 homogeneous transformation matrix for the object pose w.r.t robot
        T_robot = np.eye(4)
        T_robot[:3, 3] = tvecs_robot

        # Invert T_robot to get T_robot_inv
        T_robot_inv = np.linalg.inv(T_robot)

        # Compute RTC = T_robot_inv * T_cam and append to transformations list
        RTC = np.dot(T_robot_inv, T_cam)
        transformations.append(RTC.flatten())

    # Stack the transformation matrices into a single matrix
    A = np.vstack(transformations)

    # Use least squares to compute the rigid transformation matrix
    _, _, V = np.linalg.svd(A)
    RTC_flat = V[-1, :]

    # Reshape the flattened matrix to obtain the transformation matrix
    RTC = RTC_flat.reshape((4, 4))

    # Ensure the matrix is rigid (i.e., rotation part is orthogonal)
    R = RTC[:3, :3]
    U, S, Vt = np.linalg.svd(R)
    R = np.dot(U, Vt)
    RTC[:3, :3] = R

    # Normalize the matrix to ensure the last element of the last row is 1
    RTC /= RTC[3, 3]

    return RTC

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
