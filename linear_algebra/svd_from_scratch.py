# Hello guys 

import numpy as np
import math
import random

from numpy.linalg import eig

A = np.array([[3, 2],
              [2, 3],])

m, n = A.shape

# V
AtA = A.T @ A
e_v, V = np.linalg.eigh(AtA) # eigh return s a tuple of pair items (1: list of eigenvalues, and 2: matrix of eigenvectors)
idx_v = np.argsort(e_v)[::-1] # flips to descending order
e_v = e_v[idx_v] #start at the end and go backwards to the beginning
V = V[:, idx_v]

# Sigma
sigma = np.zeros((n, n))
singular_values = np.sqrt(np.clip(e_v, 0, None)) # if no. is less than zero, force it to be zero if any no.higher than 0, ignore
k = min(m, n)
sigma[:k, :k] = np.diag(singular_values[:k])

# U
U = np.zeros((m, n))
U[:, :k] = (A @ V) / singular_values[:k]

A_reconstructed = U @ sigma @ V.T

print(f"Original Matrix A:\n", A)
print(f"Matrix Shape A:", A.shape)
print(f"Reconstructed Matrix A:\n", np.round(A_reconstructed, 4))