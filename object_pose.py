import cv2
import numpy as np

def estimate_pose(contour):
    # Calculate centroid
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    # Estimate bounding box
    x, y, w, h = cv2.boundingRect(contour)
    bbox = ((x, y), (x + w, y + h))
    
    # Estimate orientation
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    
    return (cX, cY), bbox, angle

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blurring with larger kernel size to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Apply simple thresholding to create binary image
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # Apply Canny edge detection
    edges = cv2.Canny(binary, 30, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through detected contours
    for contour in contours:
        # Estimate pose
        centroid, bbox, angle = estimate_pose(contour)
        
        # Draw bounding box
        cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)
        
        # Draw centroid
        cv2.circle(frame, centroid, 5, (255, 255, 255), -1)
        
        # Draw orientation line
        box = cv2.boxPoints(((centroid[0], centroid[1]), (bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]), angle))
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
    
    # Display the result
    cv2.imshow('Object Pose Estimation', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object
cap.release()
cv2.destroyAllWindows()
