import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from src.pca import compute_pca

# Load dataset
data = load_iris()
X = data.data

# PCA
pc, eigenvalues = compute_pca(X)

# Mean center
X_meaned = X - np.mean(X, axis=0)

# Projection (4D → 1D)
X_reduced = X_meaned.dot(pc)

print("Projected shape:", X_reduced.shape)

# Explained variance
explained_variance = eigenvalues / np.sum(eigenvalues)
print("Explained Variance:", explained_variance)