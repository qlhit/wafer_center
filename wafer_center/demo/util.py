import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.special import iv

def GetMoments(x, y):
    # Computing the moments of X and Y
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    Sigma = np.cov(x, y, bias=True)  # bias=True for population covariance
    print(Sigma)
    Sxx = Sigma[0, 0]
    Syy = Sigma[1, 1]
    Sxy = Sigma[0, 1]
    return x_bar, y_bar, Sxx, Syy, Sxy
def UniformityTest(x, y, x_bar, y_bar, Sxx, Syy, Sxy):
    n = len(x)
    Var = np.mean(((x - x_bar)**2 + (y - y_bar)**2)**2) / 2
    Stats = (Sxx - Syy)**2 + 4 * Sxy**2
    stat = n * Stats / Var
    pval = chi2.sf(stat, 2)  # Use survival function for upper tail probability
    return pval, stat
def VM_negLogLike3(par, x, y, n):
    # 参数解析
    kappa = par[0]
    mu = par[1]
    rho = par[2]
    a = par[3]
    b = par[4]
    sigma = par[5]

    # 计算 A 和 B
    A = kappa * np.cos(mu) + (x - a) * rho / sigma**2
    B = kappa * np.sin(mu) + (y - b) * rho / sigma**2
    D = np.sqrt(A**2 + B**2)

    # 负对数似然
    y_val = -np.sum(np.log(iv(0,D))) - np.sum(D) + n * np.log(iv(0,kappa)) + n * kappa + \
            (1 / (2 * sigma**2)) * np.sum((x - a)**2 + (y - b)**2 + rho**2) + \
            n * np.log(sigma**2)

    # 梯度计算
    if len(par) > 1:  # 如果需要梯度
        r = 0.5 * (1 - iv(2,D) / iv(0,D))

        f = np.zeros(6)  # 初始化梯度向量
        # dl/dkappa
        f[0] = -np.sum(r * (kappa + rho / sigma**2 * (np.cos(mu) * (x - a) + np.sin(mu) * (y - b))) -
                       iv(1,kappa) / iv(0,kappa))
        # dl/dmu
        f[1] = -np.sum(r * kappa * rho / sigma**2 * (-np.sin(mu) * (x - a) + np.cos(mu) * (y - b)))
        # dl/drho
        f[2] = -np.sum(r * (kappa / sigma**2 * (np.cos(mu) * (x - a) + np.sin(mu) * (y - b)) +
                            (rho / sigma**4) * ((x - a)**2 + (y - b)**2)) - rho / sigma**2)
        # dl/da
        f[3] = -np.sum(-r * ((kappa * rho * np.cos(mu)) / sigma**2 + (rho**2 * (x - a)) / sigma**4) +
                       (x - a) / sigma**2)
        # dl/db
        f[4] = -np.sum(-r * ((kappa * rho * np.sin(mu)) / sigma**2 + (rho**2 * (y - b)) / sigma**4) +
                       (y - b) / sigma**2)
        # dl/dsigma^2
        f[5] = -np.sum(-r * (kappa * rho / sigma**4 * (np.cos(mu) * (x - a) + np.sin(mu) * (y - b)) +
                             rho**2 / sigma**6 * ((x - a)**2 + (y - b)**2)) +
                       0.5 * sigma**(-4) * ((x - a)**2 + (y - b)**2 + rho**2) - 1 / sigma**2)

    return y_val, f if len(par) > 1 else y_val
def get_mu_bounds1(Sxy):
    if Sxy < 0:
        lb1 = 0
        ub1 = 0.5 * np.pi
        lb2 = np.pi
        ub2 = 1.5 * np.pi
    else:
        lb1 = 0.5 * np.pi
        ub1 = np.pi
        lb2 = 1.5 * np.pi
        ub2 = 2 * np.pi

    return lb1, lb2, ub1, ub2
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
def bisection1(a, b, tol, f, max_iter):
    """
    Bisection method for finding roots of a function.
    Parameters:
    a : float : left side
    b : float : right side
    tol : float : tolerance
    f : function : function handle
    max_iter : int : maximum possible iterations
    Returns:
    res : float : root found
    flag : int : status flag (0 for success, 1 for failure)
    """
    # Initial points check
    if f(a) * f(b) > 0:
        while (f(a) * f(b) > 0) and (a <= b):
            a += 0.01
    if f(a) * f(b) > 0:
        res = 1
        flag = 1
    else:
        # Iterations
        flag = 0
        iter_count = 1
        c = (a + b) / 2  # Calculate the midpoint
        while iter_count <= max_iter and abs(f(c)) >= tol:
            iter_count += 1
            if (f(c) * f(a)) > 0:
                a = c
            else:
                b = c
            c = (a + b) / 2  # Update the midpoint
        if iter_count > max_iter:
            flag = 1
        res = c
    return res, flag
