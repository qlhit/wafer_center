import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ======================
# 1. 生成模拟数据
# ======================
np.random.seed(42)  # 固定随机种子确保可复现

# 真实圆心和半径
a_true, b_true, r_true = 2.0, 3.0, 5.0
n_points = 50  # 总数据点数

# 生成理论圆周上的坐标
theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
x_theory = a_true + r_true * np.cos(theta)
y_theory = b_true + r_true * np.sin(theta)

# 添加不同噪声（前半部分噪声小，后半部分噪声大）
sigma_small, sigma_large = 0.1, 0.1  # 噪声标准差
x_noise = np.concatenate([
    np.random.normal(0, sigma_small, n_points//2),
    np.random.normal(0, sigma_large, n_points - n_points//2)
])
y_noise = np.concatenate([
    np.random.normal(0, sigma_small, n_points//2),
    np.random.normal(0, sigma_large, n_points - n_points//2)
])
x_data = x_theory + x_noise
y_data = y_theory + y_noise

# 计算权重（与噪声方差成反比，突出小噪声点）
weights = np.concatenate([
    np.ones(n_points//2) / (sigma_small**2),       # 前半部分高权重
    np.ones(n_points - n_points//2) / (sigma_large**2)  # 后半部分低权重
])


# ======================
# 2. 加权最小二乘拟合
# ======================
def circle_residuals(theta, x, y, weights):
    """计算加权残差（用于最小二乘优化）"""
    a, b, c = theta  # c = r²，避免对r直接求导的非线性问题
    residuals = (x - a)**2 + (y - b)**2 - c  # 几何残差
    return np.sqrt(weights) * residuals  # 加权残差，等价于最小化加权平方和

# 初始参数猜测（加权重心法）
a0 = np.sum(weights * x_data) / np.sum(weights)
b0 = np.sum(weights * y_data) / np.sum(weights)
c0 = np.sum(weights * (x_data**2 + y_data**2)) / np.sum(weights) - a0**2 - b0**2
theta0 = [a0, b0, c0]  # 初始参数向量 [a, b, r²]

# 优化求解
result = least_squares(
    circle_residuals, theta0, args=(x_data, y_data, weights),
    method='trf',  # 使用信赖域反射法，适合带边界的问题（此处无边界）
    ftol=1e-20,     # 收敛精度
    max_nfev=100000  # 最大迭代次数
)
a_fit, b_fit, c_fit = result.x  # 提取拟合参数
r_fit = np.sqrt(c_fit)           # 计算拟合半径


# ======================
# 3. 结果可视化
# ======================
plt.figure(figsize=(8, 8))

# 绘制数据点（点大小与权重成正比，归一化显示）
weights_normalized = weights / np.max(weights)  # 归一化权重用于调整点大小
plt.scatter(x_data, y_data, c='tab:blue', s=weights_normalized*50, alpha=0.6,
            label='Data Points (Size ∝ Weight)')

# 绘制真实圆和拟合圆
theta_plot = np.linspace(0, 2*np.pi, 100)
x_true_circle = a_true + r_true * np.cos(theta_plot)
y_true_circle = b_true + r_true * np.sin(theta_plot)
x_fit_circle = a_fit + r_fit * np.cos(theta_plot)
y_fit_circle = b_fit + r_fit * np.sin(theta_plot)
plt.plot(x_true_circle, y_true_circle, 'r-', linewidth=2, label='True Circle')
plt.plot(x_fit_circle, y_fit_circle, 'g--', linewidth=2, label='Fitted Circle')

# 标注圆心
plt.plot(a_true, b_true, 'ro', markersize=8, label='True Center')
plt.plot(a_fit, b_fit, 'gX', markersize=8, label='Fitted Center')

# 图形设置
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Weighted Least Squares Circle Fitting')
plt.legend(loc = 'upper left')
plt.axis('equal')  # 保持纵横比一致，避免圆变形
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()  # 自动调整布局

# 显示图形（若在Jupyter中需添加 %matplotlib inline）
plt.show()


# ======================
# 4. 输出拟合结果
# ======================
print("="*50)
print("                圆拟合结果对比                ")
print("="*50)
print(f"真实圆心坐标：({a_true:.4f}, {b_true:.4f})，真实半径：{r_true:.4f}")
print(f"拟合圆心坐标：({a_fit:.4f}, {b_fit:.4f})，拟合半径：{r_fit:.4f}")
print(f"迭代次数：{result.nfev}，收敛状态：{result.status}（0表示成功）")