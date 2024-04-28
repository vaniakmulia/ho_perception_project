def determinant(matrix):
    if len(matrix) != 3 or len(matrix[0]) != 3:
        raise ValueError("Matrix must be 3x3")
    
    det = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])) - \
          (matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])) + \
          (matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
    
    return det

# Example usage:
matrix = [[0.96234677, 0.02133771, -0.05224719],
          [-0.0576901, 0.89363069, -0.35595705],
          [0.08573501, 0.22593953, 0.87087617]]

print("Matrix:")
for row in matrix:
    print(row)

print("Determinant:", determinant(matrix))
