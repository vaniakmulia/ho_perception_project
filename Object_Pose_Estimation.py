import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Camera to robot coordinate system transformation matrix
RTC = np.array([[ 0.96234677,  0.02133771, -0.05224719,  0.03211006],
                [-0.0576901 ,  0.89363069, -0.35595705,  0.23226098],
                [ 0.08573501,  0.22593953,  0.87087617, -0.54960689],
                [ 0.        ,  0.        ,  0.        ,  1.        ]])

# Camera intrinsic parameters
camera_matrix = np.array([[624.3784374801866, 0.0, 317.37965262889304],
                           [0.0, 623.7788449192459, 219.41351936116175],
                           [0.0, 0.0, 1.0]])

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    angle = np.arctan2(p[1] - q[1], p[0] - q[0])
    hypotenuse = np.sqrt((p[1] - q[1]) ** 2 + (p[0] - q[0]) ** 2)

    q[0] = p[0] - scale * hypotenuse * np.cos(angle)
    q[1] = p[1] - scale * hypotenuse * np.sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    p[0] = q[0] + 9 * np.cos(angle + np.pi / 4)
    p[1] = q[1] + 9 * np.sin(angle + np.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

    p[0] = q[0] + 9 * np.cos(angle - np.pi / 4)
    p[1] = q[1] + 9 * np.sin(angle - np.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
        
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean=np.empty((0)))

    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 1)

    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

    return cntr, eigenvectors[0], angle

cap = cv2.VideoCapture(0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

object_pose_x = []
object_pose_y = []
object_pose_z = []

major_axis_x = []
major_axis_y = []
major_axis_z = []

# Set fixed axes limits for stability
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, 10, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if 1e2 < area < 1e5:
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 2)
            centroid, major_axis_vector, angle = getOrientation(c, frame)
            
            centroid_ICS = np.array([[centroid[0]], [centroid[1]], [1]])
            centroid_CCS = np.dot(np.linalg.inv(camera_matrix), centroid_ICS)
            centroid_CCS /= centroid_CCS[2]

            z = 0.34
            centroid_CCS_3d = np.array([centroid_CCS[0][0], centroid_CCS[1][0], z, 1])

            object_pose_robot = np.dot(RTC, centroid_CCS_3d)
            object_pose_robot = object_pose_robot[:3]
            print("Object Position RCS:", object_pose_robot)

            major_axis_ICS = np.array([[major_axis_vector[0]], [major_axis_vector[1]], [0]])
            major_axis_CCS = np.dot(np.linalg.inv(camera_matrix), major_axis_ICS)
            major_axis_CCS_3d = np.array([major_axis_CCS[0][0], major_axis_CCS[1][0], 0, 0])

            major_axis_robot = np.dot(RTC[:3, :3], major_axis_CCS_3d[:3])
            print("Object Orientation RCS", major_axis_CCS_3d)

            # Clear and update the object position lists
            object_pose_x = [object_pose_robot[0]]
            object_pose_y = [object_pose_robot[1]]
            object_pose_z = [object_pose_robot[2]]

            # Clear and update the major axis lists
            major_axis_x = [major_axis_robot[0]]
            major_axis_y = [major_axis_robot[1]]
            major_axis_z = [major_axis_robot[2]]

            ax.clear()

            # Set fixed axes limits for stability
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)

            ax.quiver(object_pose_x, object_pose_y, object_pose_z,
                      major_axis_x, major_axis_y, major_axis_z,
                      length=150, color='red')

            ax.scatter(object_pose_x, object_pose_y, object_pose_z, 
                       color='blue', s=100, label='Object Position')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Object Pose in Robot CS')
            ax.set_box_aspect([1, 1, 1])

            # Display the plot
            plt.pause(0.01)

    # Show the frame
    cv2.imshow('Object Pose Camera CS', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
