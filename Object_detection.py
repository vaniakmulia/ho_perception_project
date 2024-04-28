import cv2

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blurring to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Apply simple thresholding 
    _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Apply Canny edge detection
    edges = cv2.Canny(binary, 30, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Contours', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the capture object
cap.release()
cv2.destroyAllWindows()

