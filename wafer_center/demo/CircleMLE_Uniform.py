import numpy as np
from scipy.optimize import minimize

def CircleMLE_Uniform(x, y):
    a0 = np.mean(x)
    b0 = np.mean(y)
    rho0 = np.mean(np.sqrt((x - a0)**2 + (y - b0)**2))
    sigma0 = np.sqrt(0.5 * np.mean((x - a0)**2 + (y - b0)**2 - rho0**2))
    x0 = np.array([rho0, a0, b0, np.sqrt(sigma0)])

    # Step 2 - Minimizing the log likelihood numerically
    n = x.shape[0]
    VM = lambda xx: VM_negLogLikeUniform(xx, x, y, n)

    # Optimization options
    bounds = [(0, np.inf), (None, None), (None, None), (0, np.inf)]

    # Minimization
    sol = minimize(VM, x0, bounds=bounds, method='trust-constr')

    return sol.x

def VM_negLogLikeUniform(xx, x, y, n):
    # 这里需要实现负对数似然函数的逻辑
    # 示例实现，您需要根据实际逻辑进行调整
    rho, a, b, sigma = xx
    likelihood = -np.sum(np.log(np.exp(-((x - a)**2 + (y - b)**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)))
    return -likelihood  # 返回负对数似然


points = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0],
    [0.0, -1.0],
    [0.707, 0.707],
    [-0.707, 0.707],
    [-0.707, -0.707],
    [0.707, -0.707]
])
print(CircleMLE_Uniform(points[:,0],points[:,1]))