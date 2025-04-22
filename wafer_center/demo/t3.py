import numpy as np
import matplotlib.pyplot as plt

# 生成带噪声和离群点的模拟数据
np.random.seed(42)
n_points = 100
theta = np.linspace(0, 2*np.pi, n_points)

# 真实圆参数
a_true, b_true, r_true = 2.0, 3.0, 5.0
x = a_true + r_true * np.cos(theta)
y = b_true + r_true * np.sin(theta)

# 添加高斯噪声
noise = np.random.normal(0, 0.3, n_points)
x += noise
y += noise

# 添加离群点（最后5个点）
x[-5:] += np.random.uniform(-3, 3, 5)
y[-5:] += np.random.uniform(-3, 3, 5)

# 普通最小二乘拟合（无权重）
A = np.vstack([x, y, np.ones(len(x))]).T
b = -(x**2 + y**2)
params = np.linalg.lstsq(A, b, rcond=None)[0]
a_ls = -params[0]/2
b_ls = -params[1]/2
r_ls = np.sqrt(a_ls**2 + b_ls**2 - params[2])

# 计算权重（基于到普通拟合圆的距离）
distances = np.sqrt((x - a_ls)**2 + (y - b_ls)**2) - r_ls
weights = 1 / (np.abs(distances) + 1e-6)  # 防止除以零

# 加权最小二乘拟合
W = np.diag(weights)
ATA = A.T @ W @ A
ATb = A.T @ W @ b
params_wls = np.linalg.solve(ATA, ATb)
a_wls = -params_wls[0]/2
b_wls = -params_wls[1]/2
r_wls = np.sqrt(a_wls**2 + b_wls**2 - params_wls[2])

# 绘图
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(x[:-5], y[:-5], s=20, label='正常点')
ax.scatter(x[-5:], y[-5:], s=30, c='red', marker='x', label='离群点')

# 绘制真实圆
theta_plot = np.linspace(0, 2*np.pi, 100)
ax.plot(a_true + r_true*np.cos(theta_plot),
        b_true + r_true*np.sin(theta_plot),
        'g--', lw=2, label='真实圆')

# 绘制普通最小二乘拟合圆
ax.plot(a_ls + r_ls*np.cos(theta_plot),
        b_ls + r_ls*np.sin(theta_plot),
        'b:', lw=2, label='普通LS')

# 绘制加权最小二乘拟合圆
ax.plot(a_wls + r_wls*np.cos(theta_plot),
        b_wls + r_wls*np.sin(theta_plot),
        'r-', lw=2, label='加权LS')

ax.set_aspect('equal')
ax.set_title('加权最小二乘法拟合圆心对比')
ax.legend()
plt.show()

# 输出结果
print(f"真实圆心: ({a_true:.2f}, {b_true:.2f}), 半径: {r_true:.2f}")
print(f"普通LS拟合结果: ({a_ls:.2f}, {b_ls:.2f}), 半径: {r_ls:.2f}")
print(f"加权LS拟合结果: ({a_wls:.2f}, {b_wls:.2f}), 半径: {r_wls:.2f}")