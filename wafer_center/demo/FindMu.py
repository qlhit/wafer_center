import numpy as np

def FindMu(Sxx, Syy, Sxy):
    # Spectral Decompose
    Sigma = np.array([[Sxx, Sxy], [Sxy, Syy]])
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)

    # Finding the index of the minimum eigenvalue
    ind = np.argmin(eigenvalues)

    # Finding mu
    mu_temp = np.zeros(2)
    mu_temp[0] = np.mod(np.arctan2(eigenvectors[1, ind], eigenvectors[0, ind]) + 2 * np.pi, 2 * np.pi)
    mu_temp[1] = np.mod(mu_temp[0] + np.pi, 2 * np.pi)

    mu = np.sort(mu_temp)
    lambda_ = eigenvalues  # Use a different name to avoid conflict with built-in function

    return mu, lambda_
