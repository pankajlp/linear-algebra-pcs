# 📊 PCA from Scratch (Linear Algebra Project)

## 📌 Overview
This project implements Principal Component Analysis (PCA) using basic linear algebra concepts like covariance, eigenvalues, and eigenvectors.

## 🚀 Features
- Compute covariance matrix
- Calculate eigenvalues & eigenvectors
- Perform dimensionality reduction (2D → 1D)
- Visualize principal components

## 🧠 Concepts Used
- Linear Algebra (Matrices, Eigenvalues, Eigenvectors)
- Probability (Variance, Covariance)
- Machine Learning (Dimensionality Reduction)

## 📊 PCA Insights

- Reduced 4D Iris dataset to 2D using PCA
- First principal component captured ~92% variance
- First 2 components captured ~97% variance
- Demonstrated dimensionality reduction with minimal information loss

## 📉 Feature Selection Strategy

Used cumulative explained variance to select optimal number of components (95% threshold).

## 📂 Project Structure
- src/ → core logic
- data/ → datasets
- notebooks/ → experiments

## 🎯 Goal
To understand how PCA works internally without relying on libraries like sklearn.

## 🔧 Tech Stack
- Python
- NumPy
- Matplotlib

---

## 📈 Future Improvements
- Compare with sklearn PCA
- Apply on real datasets
- Extend to multiple components
