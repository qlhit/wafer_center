import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
# 设置字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# ==============================
# 1. 生成带噪声和离群点的模拟数据
# ==============================
def generate_wafer_data(num_inliers=200, num_outliers=50, noise_std=1.0, seed=42):
    """
    生成晶圆圆周模拟数据
    参数:
        num_inliers: 内点数量（正确圆周点）
        num_outliers: 离群点数量
        noise_std: 高斯噪声标准差
        seed: 随机种子
    返回:
        points: 所有点的坐标 (N, 2)
        is_outlier: 点的离群标记 (True/False)
    """
    rng = check_random_state(seed)

    # 真实圆心和半径
    center_true = np.array([0, 0])
    radius = 100.0  # 假设已知晶圆半径

    # 生成内点（均匀分布在圆周上）
    angles = np.linspace(0, 2*np.pi, num_inliers, endpoint=False)
    x = radius * np.cos(angles) + rng.normal(0, noise_std, num_inliers)
    y = radius * np.sin(angles) + rng.normal(0, noise_std, num_inliers)
    inliers = np.column_stack([x, y])

    #离群点（集中在两个区域）
    # 区域1：半径偏离的正态分布
    outliers1 = rng.normal(0, 1.5*radius, (num_outliers//2, 2))
    # 区域2：均匀分布在矩形区域
    outliers2 = rng.uniform(-1.5*radius, 1.5*radius, (num_outliers - num_outliers//2, 2))
    outliers = np.vstack([outliers1, outliers2])

    # 合并数据
    points = np.vstack([inliers, outliers])
    is_outlier = np.concatenate([np.zeros(num_inliers, dtype=bool),
                                 np.ones(num_outliers, dtype=bool)])

    return points, is_outlier, center_true, radius

# 生成数据
points, is_outlier, center_true, radius = generate_wafer_data()

# ==============================
# 2. 传统RANSAC圆心估计
# ==============================
def ransac_circle(points, radius, num_iter=1000, threshold=1.0):
    """
    传统RANSAC圆心估计（已知半径）
    参数:
        points: 输入点集 (N, 2)
        radius: 已知的固定半径
        num_iter: RANSAC迭代次数
        threshold: 内点距离阈值
    返回:
        best_center: 最佳圆心估计
       liers: 内点索引
    """
    best_inliers = []
    best_center = None

    for _ in range(num_iter):
        # 随机采样2个点
        sample_idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[sample_idx]

        # 计算圆心候选（两点的垂直平分线交点）
        mid_point = (p1 + p2) / 2
        dir_vec = p2 - p1
        if np.linalg.norm(dir_vec) < 1e-6:  # 避免重复点
            continue
        normal_vec = np.array([-dir_vec[1], dir_vec[0]])
        normal_vec /= np.linalg.norm(normal_vec)

        # 计算圆心候选
        d_sq = radius**2 - (np.linalg.norm(p1 - mid_point)**2)
        if d_sq < 0:  # 无解情况
            continue
        d = np.sqrt(d_sq)
        center_candidate = mid_point + d * normal_vec

        # 统计内点
        distances = np.linalg.norm(points - center_candidate, axis=1)
        inliers = np.abs(distances - radius) < threshold
        if np.sum(inliers) > len(best_inliers):
            best_inliers = inliers
            best_center = center_candidate

    return best_center, best_inliers

# 运行传统RANSAC
center_ransac, inliers_ransac = ransac_circle(points, radius)

# ==============================
# 3. 神经网络加权RANSAC（模拟）
# ==============================
def weighted_ransac(points, radius, weights, num_iter=200, threshold=1.0):
    """
    神经网络引导的加权RANSAC
    参数:
        weights: 点的采样权重（模拟神经网络输出）
    """
    best_inliers = []
    best_center = None
    weights = weights / np.sum(weights)  # 归一化

    for _ in range(num_iter):
        # 根据权重采样2个点
        sample_idx = np.random.choice(len(points), 2, replace=False, p=weights)
        p1, p2 = points[sample_idx]

        # 后续步骤与传统RANSAC相同
        mid_point = (p1 + p2) / 2
        dir_vec = p2 - p1
        if np.linalg.norm(dir_vec) < 1e-6:
            continue
        normal_vec = np.array([-dir_vec[1], dir_vec[0]])
        normal_vec /= np.linalg.norm(normal_vec)

        d_sq = radius**2 - (np.linalg.norm(p1 - mid_point)**2)
        if d_sq < 0:
            continue
        d = np.sqrt(d_sq)
        center_candidate = mid_point + d * normal_vec

        distances = np.linalg.norm(points - center_candidate, axis=1)
        inliers = np.abs(distances - radius) < threshold
        if np.sum(inliers) > len(best_inliers):
            best_inliers = inliers
            best_center = center_candidate

    return best_center, best_inliers

# 模拟神经网络权重（假设正确预测内点的高权重）
weights = np.zeros(len(points))
weights[~is_outlier] = 0.9  # 内点权重设为0.9
weights[is_outlier] = 0.1   # 离群点权重设为0.1
weights += 0.05 * np.random.rand(len(points))  # 添加微小噪声
weights = np.clip(weights, 0.1, 1.0)  # 确保权重非负

# 运行加权RANSAC
center_weighted, inliers_weighted = weighted_ransac(points,radius, weights)

# ==============================
# 4. 可视化结果
# ==============================
def plot_circle(ax, center, radius, color, label):
    """绘制圆"""
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, color=color, linestyle='--', linewidth=1.5, label=label)

plt.figure(figsize=(10, 6))

# 绘制数据点
plt.scatter(points[~is_outlier, 0], points[~is_outlier, 1],
            c='blue', s=20, label='Inliers (真实圆周点)')
plt.scatter(points[is_outlier, 0], points[is_outlier, 1],
            c='red', s=20, alpha=0.6, label='Outliers (离群点)')

# 绘制真实圆心
plt.scatter(center_true[0], center_true[1],
            c='black', s=150, marker='*', edgecolor='gold', linewidth=1.5,
            label='True Center')

# 绘制RANSAC估计结果
plt.scatter(center_ransac[0], center_ransac[1],
            c='green', s=120, marker='s', edgecolor='white', linewidth=1,
            label='RANSAC Estimate')
plot_circle(plt.gca(), center_ransac, radius, 'green', 'RANSAC Circle')

# 绘制加权RANSAC估计结果
plt.scatter(center_weighted[0], center_weighted[1],
            c='purple', s=120, marker='D', edgecolor='white', linewidth=1,
            label='Weighted RANSAC')
plot_circle(plt.gca(), center_weighted, radius, 'purple', 'Weighted Circle')

# 图例与标注
plt.axis('equal')
plt.title('Wafer Center Estimation Comparison\n(RANSAC vs Neural Weighted RANSAC)')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))

# 误差标注
err_ransac = np.linalg.norm(center_ransac - center_true)
err_weighted = np.linalg.norm(center_weighted - center_true)
plt.text(0.05, 0.15,
         f'True Center: ({center_true[0]:.1f}, {center_true[1]:.1f})\n'
         f'RANSAC Error: {err_ransac:.2f} mm\n'
         f'Weighted Error: {err_weighted:.2f} mm',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()