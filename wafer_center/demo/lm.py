import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义残差函数
def residuals(params, x, y):
    xc, yc, r = params
    return np.sqrt((x - xc)**2 + (y - yc)**2) - r

# 计算圆心和半径
def fit_circle(points):
    # 提取x和y坐标
    x = points[:, 0]
    y = points[:, 1]
    # 初始猜测 (圆心坐标和半径)
    x0 = np.mean(x)
    y0 = np.mean(y)
    r0 = np.mean(np.sqrt((x - x0)**2 + (y - y0)**2))
    initial_params = [x0, y0, r0]
    # 使用Levenberg-Marquardt算法进行优化
    result = least_squares(residuals, initial_params, args=(x, y))
    # 返回圆心坐标和半径
    xc, yc, r = result.x
    return xc, yc, r
def draw(points):
    # 拟合圆心和半径
    center_x, center_y, radius = fit_circle(points)
    print(f"拟合得到的圆心: ({center_x}, {center_y}), 半径: {radius}")
    # 可视化结果
    theta = np.linspace(0, 2 * np.pi, 100)
    x_fit = center_x + radius * np.cos(theta)
    y_fit = center_y + radius * np.sin(theta)
    plt.scatter(points[:, 0], points[:, 1], label='数据点', color='red')
    plt.plot(x_fit, y_fit, label='拟合圆', color='blue')
    plt.scatter(center_x, center_y, label='圆心', color='green')
    plt.legend()
    plt.axis('equal')
    plt.title('圆的拟合')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # 示例数据：圆周坐标点
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
    draw(points)