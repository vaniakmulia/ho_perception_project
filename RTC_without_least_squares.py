import numpy as np

def compute_rigid_transformation(position_matrix_cam, position_matrix_robot):
    num_points = position_matrix_cam.shape[0]
    assert num_points == position_matrix_robot.shape[0], "Number of points must match"

    # Initialize variables to store cumulative rotation and translation
    cumulative_rotation = np.eye(3)
    cumulative_translation = np.zeros((3, 1))

    for i in range(num_points):
        # Extract corresponding points
        point_cam = position_matrix_cam[i]
        point_robot = position_matrix_robot[i]

        # Compute the translation vector between the points
        tvecs_cam = point_cam.flatten()
        tvecs_robot = point_robot.flatten()

        # Compute the rotation and translation between points
        R = np.eye(3)
        t = tvecs_cam - tvecs_robot

        # Update cumulative rotation and translation
        cumulative_rotation += R
        cumulative_translation += t.reshape((3, 1))

    # Compute the average rotation and translation
    average_rotation = cumulative_rotation / num_points
    average_translation = cumulative_translation / num_points

    # Construct the rigid transformation matrix
    RTC = np.eye(4)
    RTC[:3, :3] = average_rotation
    RTC[:3, 3] = average_translation.flatten()

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
