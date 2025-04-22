import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成带噪声的圆周数据点
def generate_circle_data(center, radius, num_points, noise_level):
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(angles) + np.random.normal(0, noise_level, num_points)
    y = center[1] + radius * np.sin(angles) + np.random.normal(0, noise_level, num_points)
    print(x.reshape(-1, 1))
    print(y.reshape(-1, 1))
    return np.column_stack((x, y))

# 定义残差函数（L-M算法）
def residuals(params, x, y):
    xc, yc, r = params
    return np.sqrt((x - xc)**2 + (y - yc)**2) - r

# 使用最小二乘法拟合圆心
def fit_circle_least_squares(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.c_[x, y, np.ones(len(x))]
    b = x**2 + y**2
    params = np.linalg.lstsq(A, b, rcond=None)[0]
    xc = params[0] / 2
    yc = params[1] / 2
    r = np.sqrt(xc**2 + yc**2 + params[2])
    return xc, yc, r

# 使用L-M算法拟合圆心
def fit_circle_lm(points):
    x = points[:, 0]
    y = points[:, 1]
    x0 = np.mean(x)
    y0 = np.mean(y)
    r0 = np.mean(np.sqrt((x - x0)**2 + (y - y0)**2))
    initial_params = [x0, y0, r0]
    result = least_squares(residuals, initial_params, args=(x, y))
    return result.x

# 生成数据
np.random.seed(0)  # 为了可重复性
center = (2, 3)
radius = 1.5
num_points = 50
noise_level = 0
points = generate_circle_data(center, radius, num_points, noise_level)

# 拟合圆心
center_ls = fit_circle_least_squares(points)
center_lm = fit_circle_lm(points)

# 可视化结果
theta = np.linspace(0, 2 * np.pi, 100)
x_fit_ls = center_ls[0] + center_ls[2] * np.cos(theta)
y_fit_ls = center_ls[1] + center_ls[2] * np.sin(theta)

x_fit_lm = center_lm[0] + center_lm[2] * np.cos(theta)
y_fit_lm = center_lm[1] + center_lm[2] * np.sin(theta)

plt.scatter(points[:, 0], points[:, 1], label='数据点', color='red', alpha=0.5)
plt.plot(x_fit_ls, y_fit_ls, label='最小二乘法拟合', color='blue')
plt.plot(x_fit_lm, y_fit_lm, label='L-M算法拟合', color='green')
plt.scatter(center_ls[0], center_ls[1], label='LS圆心', color='blue', marker='x')
plt.scatter(center_lm[0], center_lm[1], label='L-M圆心', color='red', marker='o')
plt.legend(loc = 'upper right')
plt.axis('equal')
plt.title('圆的拟合比较')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()

# 输出拟合结果
print(f"最小二乘法拟合圆心: {center_ls[:2]}, 半径: {center_ls[2]}")
print(f"L-M算法拟合圆心: {center_lm[:2]}, 半径: {center_lm[2]}")
