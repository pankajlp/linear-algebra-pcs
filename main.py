import numpy as np
import matplotlib.pyplot as plt
from src.pca import compute_pca

# Sample data
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2, 1.6],
    [1, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

# PCA
pc = compute_pca(X)

# Center data
X_meaned = X - np.mean(X, axis=0)

# Plot
plt.scatter(X_meaned[:, 0], X_meaned[:, 1])

# Draw principal component
plt.quiver(0, 0, pc[0], pc[1], scale=3)

plt.title("PCA - Principal Component")
plt.show()
