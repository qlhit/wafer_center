import util
import numpy as np
from scipy.special import gamma, gammainc, gammaincinv
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



max_sigma = 5
n = 2  # 残差维度
alpha = 0.99  # 置信度
max_iter = 100
# 预计算伽马函数常数
C = 1 / (2 ** (n / 2) * util.gamma_complete(n / 2))
k = np.sqrt(2 * gammaincinv((n - 1) / 2, alpha))


def sigma_aware_weights(residuals):
    """基于残差的概率密度计算权重"""
    weights = []
    for r in residuals:
        if r > k * max_sigma:
            weights.append(0.0)
            continue
        term1 = util.gamma_upper((n - 1) / 2, r ** 2 / (2 * max_sigma ** 2))
        term2 = util.gamma_upper((n - 1) / 2, k ** 2 / 2)
        numerator = C * (2 ** ((n - 1) / 2)) * (term1 - term2)
        denominator = 1 / max_sigma
        weights.append(numerator * denominator)
    return np.array(weights)


def magsac_solver( A, b):
    """鲁棒加权最小二乘求解"""
    # 初始估计
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(max_iter):
        # 计算残差
        residuals = np.abs(A @ x - b)
        # 计算MAGSAC++权重
        weights = sigma_aware_weights(residuals)
        # 加权最小二乘
        W = np.diag(weights)
        try:
            x = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ b
        except np.linalg.LinAlgError:
            break
    return x


def compute_quality( model, points):
    # Implement quality function from Eq.(3)
    residuals = compute_residuals(points, model)
    loss = sum(rho(r) for r in residuals)
    return 1.0 / loss


def rho(r):
    # Implement rho function from the paper
    if r > k * max_sigma:
        return max_sigma * C * 2 ** ((n - 1) / 2) * util.gamma_lower((n + 1) / 2, k ** 2 / 2)
    term1 = util.gamma_lower((n + 1) / 2, r ** 2 / (2 * max_sigma ** 2))
    term2 = util.gamma_upper((n - 1) / 2, r ** 2 / (2 * max_sigma ** 2)) - util.gamma_upper((n - 1) / 2,
                                                                                                      k ** 2 / 2)
    return C * 2 ** ((n + 1) / 2) * (max_sigma ** 2 / 2 * term1 + r ** 2 / 4 * term2)
