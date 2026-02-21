import numpy as np

def gaussian_elimination(A, b):
    # Aug matrix
    A = A.astype(float)
    b = b.astype(float)
    n = len(b) # Two unknowns
    matrix = np.column_stack((A, b))

    # forward elimination
    for i in range(n):
        best_row = i  # best_row: row with largest |value| in column i
        for k in range(i+1, n):
            if abs(matrix[k, i]) > abs(matrix[best_row, i]):
                best_row = k
        matrix[i], matrix[best_row] = matrix[best_row].copy(), matrix[i].copy()

        for j in range(i+1, n):
            factor = matrix[j, i] / matrix[i, i]
            matrix[j, i:] = matrix[j, i:] - factor * matrix[i, i:]

    # back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ax = np.dot(matrix[i, i+1:n], x[i+1:])
        x[i] = (matrix[i, -1] - sum_ax) / matrix[i, i]

    return x

# Test
A_test = np.array([[2, 1], [4, 3]])
b_test = np.array([8, 18])
print(gaussian_elimination(A_test, b_test))
