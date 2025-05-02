import util
import numpy as np
from scipy.special import gamma, gammainc, gammaincinv
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class magsacpp:
    def __init__(self, max_sigma=5.0, quantile=0.95, max_iter=100):
        self.max_sigma = max_sigma
        self.n = 2  # 残差维度
        self.alpha = 0.99  # 置信度
        self.max_iter = max_iter
        # 预计算伽马函数常数
        self.C = 1 / (2 ** (self.n / 2) * util.gamma_complete(self.n / 2))
        self.k = np.sqrt(2 * gammaincinv((self.n - 1) / 2, self.alpha))

    def sigma_aware_weights(self, residuals):
        """基于残差的概率密度计算权重"""
        weights = []
        for r in residuals:
            if r > self.k * self.max_sigma:
                weights.append(0.0)
                continue
            term1 = util.gamma_upper((self.n - 1) / 2, r ** 2 / (2 * self.max_sigma ** 2))
            term2 = util.gamma_upper((self.n - 1) / 2, self.k ** 2 / 2)
            numerator = self.C * (2 ** ((self.n - 1) / 2)) * (term1 - term2)
            denominator = 1 / self.max_sigma
            weights.append(numerator * denominator)
        return np.array(weights)

    def magsac_solver(self, A, b):
        """鲁棒加权最小二乘求解"""
        # 初始估计
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        for _ in range(self.max_iter):
            # 计算残差
            residuals = np.abs(A @ x - b)
            # 计算MAGSAC++权重
            weights = self.sigma_aware_weights(residuals)
            # 加权最小二乘
            W = np.diag(weights)
            try:
                x = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ b
            except np.linalg.LinAlgError:
                break
        return x

    def compute_quality(self, model, points):
        # Implement quality function from Eq.(3)
        residuals = self.compute_residuals(points, model)
        loss = sum(self.rho(r) for r in residuals)
        return 1.0 / loss

    def rho(self, r):
        # Implement rho function from the paper
        if r > self.k * self.max_sigma:
            return self.max_sigma * self.C * 2 ** ((self.n - 1) / 2) * util.gamma_lower((self.n + 1) / 2, self.k ** 2 / 2)
        term1 = util.gamma_lower((self.n + 1) / 2, r ** 2 / (2 * self.max_sigma ** 2))
        term2 = util.gamma_upper((self.n - 1) / 2, r ** 2 / (2 * self.max_sigma ** 2)) - util.gamma_upper((self.n - 1) / 2, self.k ** 2 / 2)
        return self.C * 2 ** ((self.n + 1) / 2) * (self.max_sigma ** 2 / 2 * term1 + r ** 2 / 4 * term2)

    def build_design_matrix(self, theta, phi, d, R):
        """构造设计矩阵A（根据图片公式）"""
        # 示例实现，需根据具体公式调整
        A = []
        for i, th in enumerate(theta):
            row = [
                np.sin(th - phi),  # sin(θ_i - φ)
                -1,  # -1项
                -(R + d * (i + 1) ** 2)  # -(R + d*i^2)
            ]
            A.append(row)
        return np.array(A)

# 示例使用
theta = np.deg2rad(np.linspace(0, 180, 100))  # 观测角度
phi = np.pi / 4  # 待估计参数
d = 0.1  # 已知参数
R = 1.0  # 已知参数

solver = magsacpp(max_sigma=2.0)

# 构造观测矩阵
A = solver.build_design_matrix(theta, phi, d, R)
b = np.array([np.sin(th - phi) * 0.5 - (R + d * (i + 1) ** 2) for i, th in enumerate(theta)])

# 加入噪声和异常值
b += np.random.normal(0, 0.5, len(b))
b[::10] += 5.0  # 添加10%的异常值

# 鲁棒估计
x_robust = solver.magsac_solver(A, b)
print(f"鲁棒估计结果: {x_robust}")

# 传统最小二乘对比
x_ls = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"传统最小二乘: {x_ls}")
