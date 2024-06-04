import numpy as np

# Define the matrix
A = np.array([[1, 2], [1, 0]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Normalize the eigenvectors
normalized_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

# Calculate the difference
difference = eigenvectors - normalized_eigenvectors

# Print the difference
print(eigenvalues)
print(eigenvectors)
