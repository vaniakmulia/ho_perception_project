import cv2
import cv2.aruco
import numpy as np

# Parse Inputs
dict_name = "DICT_ARUCO_ORIGINAL"
# marker_id = 25
marker_length = 0.038

# Map dictionary names to their corresponding enum values
dict_map = {

    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

# Use Aruco marker dictionary

dictionary = cv2.aruco.getPredefinedDictionary(dict_map[dict_name])


# Open Camera

web_cam = cv2.VideoCapture(2)
if not web_cam.isOpened():
    print("Error: Unable to open camera.")
    exit()

# Open the YAML file
fs = cv2.FileStorage("/home/alan/hands on perception mini proyect/HO-Perception-Project/calibration.yaml", cv2.FILE_STORAGE_READ)
# fs = cv2.FileStorage("/home/sawera/IFRoS-Master/2nd-Semester/HO-Perception/HO-Perception-Project/calibration.yaml", cv2.FILE_STORAGE_READ)

# Check if the file is opened successfully
if not fs.isOpened():
    print("Error: Couldn't open calibration file")
    exit()

# Read camera matrix and distortion coefficients from the YAML file
camera_matrix = fs.getNode("cameraMatrix").mat()
dist_coeffs = fs.getNode("distCoeffs").mat()
fs.release()


# Lists to store rvecs and tvecs
all_rvecs = []
all_tvecs = []

# 
while web_cam.isOpened():
    ret, frame = web_cam.read()
    if not ret:
        print("Error: Unable to read the frame")
        break

    frame_without_overlay = frame.copy()

    # Marker Detection
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, arucoParams)
    marker_corners, marker_ids , rejected_candidates = detector.detectMarkers(frame)
    cv2.aruco.drawDetectedMarkers(frame,marker_corners,marker_ids)


    # If atleast one marker is detected
    if marker_ids is not None:
        # Initialize a list of 3D corners of the marker (for pose estimation)
        object_points = np.zeros((4, 1, 3))
        object_points[0] = np.array((-marker_length/2, marker_length/2, 0))
        object_points[1] = np.array((marker_length/2, marker_length/2, 0))
        object_points[2] = np.array((marker_length/2, -marker_length/2, 0))
        object_points[3] = np.array((-marker_length/2, -marker_length/2, 0))


        # Loop for all detected markers
        if key_pressed == 99:
            for i in range(len(marker_ids)):
                # Estimate the pose of the marker
                _, rvec, tvec = cv2.solvePnP(object_points, marker_corners[i][0], camera_matrix, dist_coeffs, None, None)

                # Store rvecs and tvecs
                all_rvecs.append(rvec)
                all_tvecs.append(tvec)

                print ("Rvecs stored")
                print ("tvecs stored")

                # Draw axes on the marker
                # cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length)

                # # Display pose information
                # pose_info = "Marker {}: X: {:.2f}m, Y: {:.2f}m, Z: {:.2f}m".format(marker_ids[i], tvec[0][0], tvec[1][0], tvec[2][0])
                # cv2.putText(frame, pose_info, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output Window", frame)
    key_pressed = cv2.waitKey(1)


    if key_pressed == 27:
        break


web_cam.release()
cv2.destroyAllWindows()

# Convert lists of rvecs and tvecs to numpy arrays
all_rvecs = np.array(all_rvecs)
all_tvecs = np.array(all_tvecs)

# Save the numpy arrays to files if needed
np.save("all_rvecs.npy", all_rvecs)
np.save("all_tvecs.npy", all_tvecs)

