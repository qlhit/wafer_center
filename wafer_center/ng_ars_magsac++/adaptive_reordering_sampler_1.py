import numpy as np
import matplotlib.pyplot as plt

# 设置字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 生成模拟数据
np.random.seed(42)
n_inliers = 500
n_outliers = 200
noise = 0.3

# 真实圆心和半径
true_cx, true_cy = 5, 5
true_r = 3

# 内点：均匀角度，带噪声
theta = np.random.rand(n_inliers) * 2 * np.pi
r = true_r + np.random.randn(n_inliers) * noise
x_in = true_cx + r * np.cos(theta)
y_in = true_cy + r * np.sin(theta)

# 离群点：随机分布
x_out = np.random.uniform(0, 10, n_outliers)
y_out = np.random.uniform(0, 10, n_outliers)

x = np.concatenate([x_in, x_out])
y = np.concatenate([y_in, y_out])
points = np.vstack([x, y]).T

# 绘制原始数据
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x_in, y_in, c='blue', label='Inliers', alpha=0.6)
plt.scatter(x_out, y_out, c='red', label='Outliers', alpha=0.6)
plt.title("原始数据（处理前）")
plt.legend()


# 实现ARS
class AdaptiveReorderingSampler:
    def __init__(self, points, m=3, a=1, b_init=1, threshold=0.5, max_iterations=1000):
        self.points = points
        self.m = m
        self.a = a
        self.b_init = b_init
        self.threshold = threshold
        self.max_iterations = max_iterations

        n_points = points.shape[0]
        self.mu = np.ones(n_points) * 0.5  # 初始概率
        self.b = np.full(n_points, b_init)
        self.n = np.zeros(n_points, dtype=int)

        self.best_model = (0, 0, 0)
        self.best_inliers = np.zeros(n_points, dtype=bool)
        self.best_num_inliers = 0

    def update_mu(self, indices):
        for idx in indices:
            self.n[idx] += 1
            self.b[idx] = self.b_init + self.n[idx]
            self.mu[idx] = self.a / (self.a + self.b[idx])

    def circle_fit(self, sample):
        if len(sample) < 3:
            return (0, 0, 0)
        p1, p2, p3 = sample[0], sample[1], sample[2]
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        A = 2 * np.array([[x2 - x1, y2 - y1], [x3 - x2, y3 - y2]])
        b = np.array([
            x2 ** 2 + y2 ** 2 - x1 ** 2 - y1 ** 2,
            x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2
        ])
        try:
            cx, cy = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return (0, 0, 0)
        r = np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
        return (cx, cy, r)

    def fit(self):
        for _ in range(self.max_iterations):
            sorted_indices = np.argsort(-self.mu)
            sample_indices = sorted_indices[:self.m]
            sample = self.points[sample_indices]
            cx, cy, r = self.circle_fit(sample)
            if r <= 0:
                continue

            distances = np.abs(np.sqrt((self.points[:, 0] - cx) ** 2 + (self.points[:, 1] - cy) ** 2) - r)
            inliers = distances < self.threshold
            num_inliers = np.sum(inliers)

            if num_inliers > self.best_num_inliers:
                self.best_num_inliers = num_inliers
                self.best_model = (cx, cy, r)
                self.best_inliers = inliers
            else:
                self.update_mu(sample_indices)
        return self.best_model, self.best_inliers


# 实现传统RANSAC
class RANSAC:
    def __init__(self, points, m=3, threshold=0.2, max_iterations=1000):
        self.points = points
        self.m = m
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.best_model = (0, 0, 0)
        self.best_inliers = np.zeros(len(points), dtype=bool)
        self.best_num_inliers = 0

    def circle_fit(self, sample):
        if len(sample) < 3:
            return (0, 0, 0)
        p1, p2, p3 = sample[0], sample[1], sample[2]
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        A = 2 * np.array([[x2 - x1, y2 - y1], [x3 - x2, y3 - y2]])
        b = np.array([
            x2 ** 2 + y2 ** 2 - x1 ** 2 - y1 ** 2,
            x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2
        ])
        try:
            cx, cy = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return (0, 0, 0)
        r = np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
        return (cx, cy, r)

    def fit(self):
        for _ in range(self.max_iterations):
            sample_indices = np.random.choice(len(self.points), self.m, replace=False)
            sample = self.points[sample_indices]

            cx, cy, r = self.circle_fit(sample)
            if r <= 0:
                continue

            distances = np.abs(np.sqrt((self.points[:, 0] - cx) ** 2 + (self.points[:, 1] - cy) ** 2) - r)
            inliers = distances < self.threshold
            num_inliers = np.sum(inliers)

            if num_inliers > self.best_num_inliers:
                self.best_num_inliers = num_inliers
                self.best_model = (cx, cy, r)
                self.best_inliers = inliers
        return self.best_model, self.best_inliers


# 运行ARS和RANSAC
ars = AdaptiveReorderingSampler(points, threshold=0.23, max_iterations=1000)
ars_model, ars_inliers = ars.fit()

ransac = RANSAC(points, threshold=0.2, max_iterations=1000)
ransac_model, ransac_inliers = ransac.fit()

# 绘制ARS结果
plt.subplot(1, 2, 2)
plt.scatter(points[:, 0], points[:, 1], c='gray', alpha=0.3, label='所有点')
plt.scatter(points[ars_inliers, 0], points[ars_inliers, 1], c='blue', alpha=0.6, label='ARS内点')
cx, cy, r = ars_model
theta = np.linspace(0, 2 * np.pi, 100)
x_circle = cx + r * np.cos(theta)
y_circle = cy + r * np.sin(theta)
plt.plot(x_circle, y_circle, c='red', linewidth=2, label='拟合圆')
plt.title('ARS处理后结果')
plt.legend()

plt.tight_layout()
plt.show()

# 绘制RANSAC结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(points[:, 0], points[:, 1], c='gray', alpha=0.3, label='所有点')
plt.scatter(points[ransac_inliers, 0], points[ransac_inliers, 1], c='green', alpha=0.6, label='RANSAC内点')
cx_r, cy_r, r_r = ransac_model
x_circle_r = cx_r + r_r * np.cos(theta)
y_circle_r = cy_r + r_r * np.sin(theta)
plt.plot(x_circle_r, y_circle_r, c='orange', linewidth=2, label='拟合圆')
plt.title('传统RANSAC结果')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(points[:, 0], points[:, 1], c='gray', alpha=0.3, label='所有点')
plt.scatter(points[ars_inliers, 0], points[ars_inliers, 1], c='blue', alpha=0.6, label='ARS内点')
plt.plot(x_circle, y_circle, c='red', linewidth=2, label='ARS拟合圆')
plt.title('ARS结果对比')
plt.legend()

plt.tight_layout()
plt.show()

# 输出模型参数
print("真实模型: 中心(5.00, 5.00), 半径3.00")
print(f"ARS拟合结果: 中心({cx:.2f}, {cy:.2f}), 半径{r:.2f}")
print(f"RANSAC拟合结果: 中心({cx_r:.2f}, {cy_r:.2f}), 半径{r_r:.2f}")
