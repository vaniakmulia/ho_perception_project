import numpy as np

def compute_rigid_transformation(position_matrix_cam, position_matrix_robot):
    # Rearrange coefficients and constants in A
    position_matrix_cam_rearranged = np.zeros((len(position_matrix_cam) * 3, 12))

    for i in range(len(position_matrix_cam)):
        x, y, z = position_matrix_cam[i][0][0], position_matrix_cam[i][1][0], position_matrix_cam[i][2][0]
        position_matrix_cam_rearranged[3*i] = [x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        position_matrix_cam_rearranged[3*i+1] = [0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0]
        position_matrix_cam_rearranged[3*i+2] = [0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1]

    # Reshape position matrix robot to a column matrix
    position_matrix_robot_reshaped = position_matrix_robot.reshape(-1, 1)

    # Compute (A^T A) and (A^T B)
    ATA = np.dot(position_matrix_cam_rearranged.T, position_matrix_cam_rearranged)
    ATB = np.dot(position_matrix_cam_rearranged.T, position_matrix_robot_reshaped)

    # Calculate the inverse of (A^T A)
    ATA_inv = np.linalg.inv(ATA)

    # Compute Transformation matrix
    RTC = np.dot(ATA_inv, ATB)

    # Reshape RTC to 3x4 matrix
    RTC_reshaped = RTC.reshape(3, 4)

    # Append the final row to RTC_reshaped
    Rigid_Transformation_Matrix = np.vstack((RTC_reshaped, [0, 0, 0, 1]))

    return Rigid_Transformation_Matrix

def main():
    # Object position matrix with respect to robot
    position_matrix_robot = np.array([np.array([4, 9, 0]).reshape((3, 1)),
                  np.array([6, 9, 0]).reshape((3, 1)),
                  np.array([8, 9, 0]).reshape((3, 1)),
                  np.array([4, 11, 0]).reshape((3, 1)),
                  np.array([6, 11, 0]).reshape((3, 1)),
                  np.array([8, 11, 0]).reshape((3, 1)),
                  np.array([4, 13, 0]).reshape((3, 1)),
                  np.array([6, 13, 0]).reshape((3, 1)),
                  np.array([8, 13, 0]).reshape((3, 1))])

    # Load the data from the .npy file
    all_tvecs = np.load("all_tvecs.npy")

    # Reshape the data to have 9 rows and 3 columns
    all_tvecs = all_tvecs.reshape(9, 3)

    # Create the object position matrix with respect to camera
    position_matrix_cam = np.array([tvec.reshape((3, 1)) for tvec in all_tvecs])

    # Compute the rigid transformation matrix
    Rigid_Transformation_Matrix = compute_rigid_transformation(position_matrix_cam, position_matrix_robot)

    # Print the transformation matrix
    print("Rigid transformation matrix RTC:")
    print(Rigid_Transformation_Matrix)

if __name__ == "__main__":
    main()
