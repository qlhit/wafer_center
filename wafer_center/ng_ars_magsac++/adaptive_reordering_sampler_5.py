import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def generate_circle_data(num_points, radius=1.0, noise_std=0):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta) + np.random.normal(0, noise_std, num_points)
    y = radius * np.sin(theta) + np.random.normal(0, noise_std, num_points)
    data = np.column_stack((x, y))
    return torch.tensor(data, dtype=torch.float32), torch.tensor([radius], dtype=torch.float32)


class WeightPredictor(nn.Module):
    def __init__(self, in_num, hide_num, out_num):
        super(WeightPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_num, hide_num),
            nn.LayerNorm(hide_num),
            nn.ReLU(),
            nn.Linear(hide_num, hide_num),
            nn.ReLU(),
            nn.Linear(hide_num, out_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def weighted_least_squares_loss(weights, data):
    x = data[:, 0]
    y = data[:, 1]
    n = x.shape[0]
    X = torch.stack([2 * x, 2 * y, torch.ones(n)], dim=1)
    W = torch.diag(weights.squeeze())
    z = (x ** 2 + y ** 2).unsqueeze(1)
    XtWX = X.T @ W @ X
    XtWz = X.T @ W @ z
    # 修改部分：根据实际返回值数量解包
    result = torch.linalg.lstsq(XtWX, XtWz)
    theta = result.solution
    a, b, c = theta[0], theta[1], theta[2]
    r = torch.sqrt(a ** 2 + b ** 2 - c)
    residuals = torch.sqrt((x - a) ** 2 + (y - b) ** 2) - r
    return torch.sum(weights * residuals ** 2)


def get_newWeights(weights, data, true_center, radius):
    # 计算每个点到圆心的距离
    dx = data[:, 0] - true_center[0]
    dy = data[:, 1] - true_center[1]
    distance = torch.sqrt(dx ** 2 + dy ** 2)
    # 计算距离过小的样本点的惩罚因子
    min_distance = radius - 0.02 * radius
    max_distance = radius + 0.02 * radius
    penalty_min = torch.where(distance < min_distance, torch.ones_like(distance), torch.zeros_like(distance))
    penalty_max = torch.where(distance > max_distance, torch.ones_like(distance), torch.zeros_like(distance))
    penalty = penalty_min + penalty_max
    # 应用惩罚因子到权重上
    return weights + weights * penalty.view(-1, 1)


def train_model(data, true_center, radius, batch_size=32, epochs=200, lr=1e-4):
    model = WeightPredictor(2, 32, 1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        # 随机打乱数据
        indices = torch.randperm(data.shape[0])
        shuffled_data = data[indices]
        for i in range(0, data.shape[0], batch_size):
            batch = shuffled_data[i:i + batch_size]
            optimizer.zero_grad()
            # 计算权重偏移量并加上初始权重 1
            weights = model(batch)
            loss = weighted_least_squares_loss(weights, data)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model, losses


if __name__ == "__main__":
    # 超参数
    num_points = 60
    radius = 5.0  # 真实半径（圆心设为原点(0,0)）
    noise_std = 0.1
    batch_size = 30
    epochs = 300

    # 生成数据
    data, _ = generate_circle_data(num_points, radius, noise_std)
    true_center = torch.tensor([0.0, 0.0], dtype=torch.float32)  # 真实圆心

    # 训练模型
    model, losses = train_model(data, true_center, radius, batch_size, epochs)

    # 预测权重
    with torch.no_grad():
        weights = model(data).numpy().flatten()

    # 绘制结果
    plt.figure(figsize=(12, 12))

    # 子图 1：样本点权重分布
    plt.subplot(2, 1, 1)
    plt.scatter(data[:, 0], data[:, 1], c=weights, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Weight')
    plt.title('Sampled Points with Predicted Weights')
    plt.xlabel('X')
    plt.ylabel('Y')
    circle = plt.Circle((0, 0), radius, color='r', fill=False, linestyle='--')
    plt.gca().add_patch(circle)

    # 子图 2：LOSS 函数曲线
    plt.subplot(2, 1, 2)
    plt.plot(losses)
    plt.title('Weighted Least Squares Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()
