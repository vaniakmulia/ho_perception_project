import cv2

# Initialize the video capture object
cap = cv2.VideoCapture(0)

image_counter = 0

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

    # Display the result
    cv2.imshow('Contours', frame)
    
    # Save the image if 'c' is pressed
    key = cv2.waitKey(1)
    if key == ord('c'):
        image_counter += 1
        filename = 'contour_image_{}.png'.format(image_counter)
        cv2.imwrite(filename, frame)
    # Break the loop if 'q' is pressed
    elif key == 27:
        break

# Release the capture object
cap.release()
cv2.destroyAllWindows()
