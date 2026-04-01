import numpy as np

def compute_pca(X):
    # Step 1: Mean center
    X_meaned = X - np.mean(X, axis=0)

    # Step 2: Covariance matrix
    cov_matrix = np.cov(X_meaned, rowvar=False)

    # Step 3: Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Step 4: Sort eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Select top component
    principal_component = eigenvectors[:, 0]

    return principal_component