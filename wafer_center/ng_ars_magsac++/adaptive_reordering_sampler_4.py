import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# 1. 生成晶圆圆周数据（带噪声）
def generate_circle_data(num_points, radius=1.0, noise_std=0.1):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta) + np.random.normal(0, noise_std, num_points)
    y = radius * np.sin(theta) + np.random.normal(0, noise_std, num_points)
    data = np.column_stack((x, y))
    return torch.tensor(data, dtype=torch.float32), torch.tensor([radius], dtype=torch.float32)

# 2. 定义神经网络模型（预测权重）
class WeightPredictor(nn.Module):
    def __init__(self):
        super(WeightPredictor, self).__init__()
        self.fc1 = nn.Linear(2, 32)  # 输入为(x, y)坐标
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)  # 输入为(x, y)坐标
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)   # 输出为权重（范围建议用Sigmoid限制在[0,1]）
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x))))))

# 3. 加权最小二乘损失函数（拟合圆心）
def weighted_least_squares_loss(weights, data, true_center):
    # 计算每个点到圆心的距离误差并加权, 半径
    data = weights * data
    dx = data[:, 0] - true_center[0]
    dy = data[:, 1] - true_center[1]
    distance_sq = dx**2 + dy**2 - radius
    weighted_loss = torch.sum(torch.abs(distance_sq))
    return weighted_loss

# 4. 训练过程
def train_model(data, true_center, batch_size=32, epochs=200, lr=1e-4):
    model = WeightPredictor()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        # 随机打乱数据
        indices = torch.randperm(data.shape[0])
        shuffled_data = data[indices]
        for i in range(0, data.shape[0], batch_size):
            batch = shuffled_data[i:i+batch_size]
            optimizer.zero_grad()
            weights = model(batch)
            loss = weighted_least_squares_loss(weights, batch, true_center)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model, losses

# 5. 主流程
if __name__ == "__main__":
    # 超参数
    num_points = 500
    radius = 5.0  # 真实半径（圆心设为原点(0,0)）
    noise_std = 0
    batch_size = 64
    epochs = 300

    # 生成数据
    data, _ = generate_circle_data(num_points, radius, noise_std)
    true_center = torch.tensor([0.0, 0.0], dtype=torch.float32)  # 真实圆心

    # 训练模型
    model, losses = train_model(data, true_center, batch_size, epochs)

    # 预测权重
    with torch.no_grad():
        weights = model(data).numpy().flatten()

    # 6. 绘制结果
    plt.figure(figsize=(15, 5))

    # 子图1：样本点权重分布
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c=weights, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Weight')
    plt.title('Sampled Points with Predicted Weights')
    plt.xlabel('X')
    plt.ylabel('Y')
    circle = plt.Circle((0, 0), radius, color='r', fill=False, linestyle='--')
    plt.gca().add_patch(circle)

    # 子图2：LOSS函数曲线
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Weighted Least Squares Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()