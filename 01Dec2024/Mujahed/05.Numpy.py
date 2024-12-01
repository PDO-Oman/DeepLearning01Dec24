import numpy as np

# 1. **Creating NumPy Arrays**

# Create a 1D array (Vector)
vector = np.array([1, 2, 3, 4, 5])
print("1D Array (Vector):\n", vector)

# Create a 2D array (Matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\n2D Array (Matrix):\n", matrix)

# Create a 3D array (Tensor)
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\n3D Array (Tensor):\n", tensor)


# 2. **Array Operations**

# Element-wise addition
add_result = vector + 2
print("\nVector + 2:\n", add_result)

# Element-wise multiplication
multiply_result = vector * 3
print("\nVector * 3:\n", multiply_result)

# Matrix multiplication (Dot product)
dot_product = np.dot(matrix, matrix)
print("\nMatrix * Matrix (Dot product):\n", dot_product)

# Transposing a matrix
transpose_matrix = np.transpose(matrix)
print("\nTranspose of Matrix:\n", transpose_matrix)