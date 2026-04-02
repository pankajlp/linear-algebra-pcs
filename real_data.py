from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = load_iris()
X = data.data
y = data.target

# Mean center
X_meaned = X - np.mean(X, axis=0)

# Covariance
cov_matrix = np.cov(X_meaned, rowvar=False)

# Eigen
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]
explained_variance = eigenvalues / np.sum(eigenvalues)

cumulative_variance = np.cumsum(explained_variance)

print("Explained:", explained_variance)
print("Cumulative:", cumulative_variance)
# Top 2 components
W = eigenvectors[:, :2]
threshold = 0.95
num_components = np.argmax(cumulative_variance >= threshold) + 1

print("Optimal components:", num_components)
# Project
X_2D = X_meaned.dot(W)

# Plot
plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y)
plt.title("PCA - Iris Dataset (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()