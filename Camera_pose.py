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
fs = cv2.FileStorage("/home/sawera/IFRoS-Master/2nd-Semester/HO-Perception/HO-Perception-Project/calibration.yaml", cv2.FILE_STORAGE_READ)

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
marker_ids_list = []  


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
    print("Marker_ids:", marker_ids)


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
                marker_ids_list.append(marker_ids[i][0])  # Append the marker ID


                print ("Rvecs stored")
                print ("tvecs stored")


    cv2.imshow("Output Window", frame)
    key_pressed = cv2.waitKey(1)


    if key_pressed == 27:
        break


web_cam.release()
cv2.destroyAllWindows()

# Convert lists of rvecs and tvecs to numpy arrays
all_rvecs = np.array(all_rvecs)
all_tvecs = np.array(all_tvecs)

# Convert marker_ids_list to a numpy array
marker_ids_array = np.array(marker_ids_list).flatten()

# Sort marker IDs and corresponding tvecs and rvecs
sorted_ids = np.sort(marker_ids_array)
sorted_indices = np.argsort(marker_ids_array)
sorted_tvecs = all_tvecs[sorted_indices]
sorted_rvecs = all_rvecs[sorted_indices]

# Save the numpy arrays
np.save("All_rvecs.npy", sorted_rvecs)
np.save("All_tvecs.npy", sorted_tvecs)
np.save("Marker_ids.npy", sorted_ids)
