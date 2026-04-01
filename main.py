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
pc, eigenvalues = compute_pca(X)

# Center data
X_meaned = X - np.mean(X, axis=0)

# Projection (2D → 1D)
X_reduced = X_meaned.dot(pc)

print("Projected Data:\n", X_reduced)

# Explained variance (CORRECT ORDER)
explained_variance = eigenvalues / np.sum(eigenvalues)
print("Explained Variance:", explained_variance)

# Reconstruct points on line
X_projected = np.outer(X_reduced, pc)

# Plot
plt.scatter(X_meaned[:, 0], X_meaned[:, 1], label="Original")
plt.scatter(X_projected[:, 0], X_projected[:, 1], color='red', label="Projected")

# Principal direction
plt.quiver(0, 0, pc[0], pc[1], scale=3)

plt.legend()
plt.title("PCA Projection")
plt.show()