import cv2
import cv2.aruco
import numpy as np

# Parse inputs
dict_name = "DICT_ARUCO_ORIGINAL"  # dictionary
rows = 3  # number of rows
columns = 3  # number of columns
marker_length = 0.038  # side length of a single marker in meters
distance = 0.015  # distance between markers in meters
camera_filename = "calibration.yaml"  # name of the calibration parameters file (yaml)

# Map dictionary names to their corresponding enum values
dict_map = {

    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

# Use Aruco marker dictionary
dictionary = cv2.aruco.getPredefinedDictionary(dict_map[dict_name])

# Open camera
web_cam = cv2.VideoCapture(2)
if not web_cam.isOpened():
    print("Error: Unable to open camera.")
    exit()

img_id = 0

all_marker_corners = []
all_marker_ids = []

while web_cam.isOpened():
    ret, frame = web_cam.read()
    if not ret:
        print("Error: Unable to read frame from camera.")
        break

    frame_without_overlay = frame.copy()

    # Marker Detection
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, arucoParams)
    marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(frame)
    cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

    cv2.imshow("Output Window", frame)
    key_pressed = cv2.waitKey(1)

    if key_pressed == 99:  # 'c' key pressed
        img_id += 1
        img_filename = f"image{img_id}.png"
        cv2.imwrite(img_filename, frame_without_overlay)

        all_marker_corners.append(marker_corners)
        all_marker_ids.append(marker_ids)
        print(f"Image {img_id} saved.")

    if key_pressed == 27:  # ESC key pressed
        break

web_cam.release()
cv2.destroyAllWindows()

#Prepare data for calibration
nFrames = len(all_marker_corners)
objectPoints = []  # List to store object points
imagePoints = []   # List to store image points

# Create gridboard object
gridboard = cv2.aruco.GridBoard((columns, rows), marker_length, distance, dictionary)

# Pre-process image points and object points for every frame
for frame in range(nFrames):
    current_img_points = np.array([])
    current_obj_points = np.array([])

    # Match object points with image points (using the gridboard)
    current_obj_points, current_img_points = gridboard.matchImagePoints(all_marker_corners[frame], all_marker_ids[frame], current_obj_points, current_img_points)

    # Store the pre-processed image points and object points
    if current_img_points.any() and current_obj_points.any():
        imagePoints.append(np.concatenate(current_img_points))
        objectPoints.append(np.concatenate(current_obj_points))


# Perform camera calibration
imgSize = frame_without_overlay.shape[1::-1]  # Extract width and height only
cameraMatrix = np.eye(3)
distCoeffs = np.zeros((5, 1))
rvecs = []
tvecs = []

if len(objectPoints) > 0:
    repError, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints, imagePoints, imgSize, cameraMatrix, distCoeffs, rvecs, tvecs
    )

   # Save calibration parameters
    with open(camera_filename, 'w') as fs:
        fs.write("%YAML:1.0\n---\n")
        fs.write("cameraMatrix: !!opencv-matrix\n")
        fs.write("   rows: 3\n")
        fs.write("   cols: 3\n")
        fs.write("   dt: d\n")
        fs.write("   data: [ ")
        fs.write(", ".join([str(value) for value in cameraMatrix.flatten()]))
        fs.write(" ]\n")
        fs.write("distCoeffs: !!opencv-matrix\n")
        fs.write("   rows: 1\n")
        fs.write("   cols: 5\n")
        fs.write("   dt: d\n")
        fs.write("   data: [ ")
        fs.write(", ".join([str(value) for value in distCoeffs.flatten()]))
        fs.write(" ]\n")
        fs.write("repError: " + str(repError) + "\n")

    print("Calibration done. Reprojection error:", repError)


else:
    print("Error: No markers detected for calibration.")


