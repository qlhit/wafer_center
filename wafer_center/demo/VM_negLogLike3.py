import numpy as np
from scipy.special import i0, i1, i2

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
    y_val = -np.sum(np.log(i0(D))) - np.sum(D) + n * np.log(i0(kappa)) + n * kappa + \
            (1 / (2 * sigma**2)) * np.sum((x - a)**2 + (y - b)**2 + rho**2) + \
            n * np.log(sigma**2)

    # 梯度计算
    if len(par) > 1:  # 如果需要梯度
        r = 0.5 * (1 - i2(D) / i0(D))

        f = np.zeros(6)  # 初始化梯度向量
        # dl/dkappa
        f[0] = -np.sum(r * (kappa + rho / sigma**2 * (np.cos(mu) * (x - a) + np.sin(mu) * (y - b))) -
                       i1(kappa) / i0(kappa))
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
