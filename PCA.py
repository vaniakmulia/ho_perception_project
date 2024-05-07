import cv2
import numpy as np

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    angle = np.arctan2(p[1] - q[1], p[0] - q[0])  # angle in radians
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

        print(data_pts)

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean=np.empty((0)))

    print("Major axis vector:", eigenvectors[0])
    print("Minor axis vector:", eigenvectors[1])


    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 1)

    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians

    # Label with the rotation angle
    # label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    # textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
    # cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    

    return angle

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, 10, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if 1e2 < area < 1e5:
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 2)
            getOrientation(c, frame)

    cv2.imshow('Contours', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
