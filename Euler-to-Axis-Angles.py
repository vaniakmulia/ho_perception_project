import numpy as np
import cv2

# Function to convert Euler angles to axis-angle representation
def euler_to_axis_angle(euler_angles):
    # Convert Euler angles to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(euler_angles)

    # Extract the angle from the rotation matrix
    angle = np.arccos((np.trace(rotation_matrix) - 1) / 2)

    # Calculate the axis
    axis = 1 / (2 * np.sin(angle)) * np.array([
        rotation_matrix[2, 1] - rotation_matrix[1, 2],
        rotation_matrix[0, 2] - rotation_matrix[2, 0],
        rotation_matrix[1, 0] - rotation_matrix[0, 1]
    ])

    # Combine angle and axis
    axis_angle = axis * angle

    return axis_angle

# Example Euler angles (in radians)
euler_angles = np.array([np.pi, 0, -np.pi/2])  # Example Euler angles (roll, pitch, yaw)

# Convert Euler angles to axis-angle representation
axis_angle = euler_to_axis_angle(euler_angles)

print("Axis-angle representation:", axis_angle)
