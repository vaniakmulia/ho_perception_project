import numpy as np

def compute_rigid_transformation(position_matrix_cam, position_matrix_robot):
    # Rearrange coefficients and constants in A
    position_matrix_cam_rearranged = np.zeros((len(position_matrix_cam) * 3, 12))

    for i in range(len(position_matrix_cam)):
        x, y, z = position_matrix_cam[i][0][0], position_matrix_cam[i][1][0], position_matrix_cam[i][2][0]
        position_matrix_cam_rearranged[3*i] = [x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        position_matrix_cam_rearranged[3*i+1] = [0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0]
        position_matrix_cam_rearranged[3*i+2] = [0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1]

    print("Psotion matrix cam rearranged:", position_matrix_cam_rearranged)
    # Reshape position matrix robot to a column matrix
    position_matrix_robot_reshaped = position_matrix_robot.reshape(-1, 1)


    # Compute (A^T A) and (A^T B)
    ATA = position_matrix_cam_rearranged.T @ position_matrix_cam_rearranged
    ATB = position_matrix_cam_rearranged.T @ position_matrix_robot_reshaped

    # Calculate the inverse of (A^T A)
    ATA_inv = np.linalg.inv(ATA)

    # Compute Transformation matrix
    RTC = ATA_inv @ ATB

    print("RTC:", RTC)

    # Reshape RTC to 3x4 matrix
    RTC_reshaped = RTC.reshape(3, 4)


    print("RTC_RESHAPED:", RTC_reshaped)

    # Append the final row to RTC_reshaped
    Rigid_Transformation_Matrix = np.vstack((RTC_reshaped, [0, 0, 0, 1]))

    return Rigid_Transformation_Matrix

def main():
    # Object position matrix with respect to robot
    position_matrix_robot = np.array([np.array([0.019, 0.019, 0]).reshape((3, 1)),
    np.array([0.072, 0.019, 0]).reshape((3, 1)),  
    np.array([0.125, 0.019, 0]).reshape((3, 1)),
    np.array([0.019, 0.072, 0]).reshape((3, 1)),
    np.array([0.072, 0.072, 0]).reshape((3, 1)),
    np.array([0.125, 0.072, 0]).reshape((3, 1)),
    np.array([0.019, 0.125, 0]).reshape((3, 1)),  
    np.array([0.072, 0.125, 0]).reshape((3, 1)),  
    np.array([0.125, 0.125, 0]).reshape((3, 1)), 
    np.array([0.019, 0.019, -0.12]).reshape((3, 1)),
    np.array([0.072, 0.019, -0.12]).reshape((3, 1)),
    np.array([0.125, 0.019, -0.12]).reshape((3, 1)),
    np.array([0.019, 0.072, -0.12]).reshape((3, 1)),
    np.array([0.072, 0.072, -0.12]).reshape((3, 1)),
    np.array([0.125, 0.072, -0.12]).reshape((3, 1)),
    np.array([0.019, 0.125, -0.12]).reshape((3, 1)),
    np.array([0.072, 0.125, -0.12]).reshape((3, 1)),
    np.array([0.125, 0.125, -0.12]).reshape((3, 1))])

    
    print("Robot position:", position_matrix_robot)
    print("Robot Position shape:",position_matrix_robot.shape)

    # Load the data from the .npy file
    all_tvecs = np.load("All_tvecs.npy")
    print("ALL_Tvecs:", all_tvecs)


    # Reshape the data to have 18 rows and 3 columns
    all_tvecs_reshaped = all_tvecs.reshape(18, 3)
    print(all_tvecs.shape)


    # Create the object position matrix with respect to camera
    position_matrix_cam = np.array([tvec.reshape((3, 1)) for tvec in all_tvecs_reshaped])

    # Compute the rigid transformation matrix
    Rigid_Transformation_Matrix = compute_rigid_transformation(position_matrix_cam, position_matrix_robot)

    # Print the transformation matrix
    print("Rigid transformation matrix RTC:")
    print(Rigid_Transformation_Matrix)

if __name__ == "__main__":
    main()
