import cv2
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blurring to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply simple thresholding 
    _, binary = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    # Apply Canny edge detection
    edges = cv2.Canny(binary, 10, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Find the centroid and orientation of each contour
    for contour in contours:
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        # # Calculate orientation
        # orientation = 0
        # if len(contour) >= 5:
        #     ellipse = cv2.fitEllipse(contour)
        #     orientation = ellipse[2]

        # Draw centroid
        cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)
        
        # Draw orientation line
        # Indicates the angle at which the object is rotated with respect to centroid
        # Not useful here
        # angle_rad = np.deg2rad(orientation)
        # length = 100
        # x2 = int(cX + length * np.cos(angle_rad))
        # y2 = int(cY + length * np.sin(angle_rad))
        # cv2.line(frame, (cX, cY), (x2, y2), (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Contours-Pose', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the capture object
cap.release()
cv2.destroyAllWindows()
