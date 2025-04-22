import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def fit_circle_least_squares(points):
    """
    使用最小二乘法拟合圆心
    :param points: 圆周坐标点的数组，形状为 (N, 2)，N为点的数量
    :return: 圆心坐标 (a, b) 和半径 r
    """
    x = points[:, 0]
    y = points[:, 1]
    x_m = np.mean(x)
    y_m = np.mean(y)
    A = np.vstack([x, y, np.ones_like(x)]).T
    b = x**2 + y**2
    p = np.linalg.lstsq(A, b, rcond=None)[0]
    a = p[0] / 2
    b = p[1] / 2
    r = np.sqrt(a**2 + b**2 + p[2])
    return a, b, r


def draw(points):
    # 拟合圆心
    circle_center = fit_circle_least_squares(points)
    print('center_x:',circle_center[0],'center_y:',circle_center[1],'radius:',circle_center[2])
    # 绘制圆和点
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    # 绘制圆周坐标点
    ax.scatter(points[:, 0], points[:, 1], color='red', label='圆周坐标点')
    # 绘制拟合圆
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = circle_center[0] + circle_center[2] * np.cos(theta)
    y_circle = circle_center[1] + circle_center[2] * np.sin(theta)
    ax.plot(x_circle, y_circle, color='blue', label='拟合圆')
    # 绘制圆心
    ax.scatter(circle_center[0], circle_center[1], color='green', marker='x', s=100, label='圆心')
    # 设置图例和标题
    ax.legend(loc = 'upper right')
    ax.set_title('圆周坐标点及拟合圆')
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.grid()
    # 显示图形
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

