import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import leastsq
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 定义神经网络模型
class WeightPredictor(nn.Module):
    def __init__(self):
        super(WeightPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.LayerNorm(32),
            # nn.Sigmoid(),
            nn.Linear(32, 32),
            # nn.Sigmoid(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return get_newWeights(self.layers(x), x[:,0], x[:,1], torch.tensor([2, 3], dtype=torch.float32), 1.5)


# 加权最小二乘拟合函数
def weighted_least_squares_fit(x, y, w):
    n = x.shape[0]
    X = torch.stack([2 * x, 2 * y, torch.ones(n)], dim=1)
    W = torch.diag(w.squeeze())
    z = (x ** 2 + y ** 2).unsqueeze(1)
    XtWX = X.T @ W @ X
    XtWz = X.T @ W @ z
    # 修改部分：根据实际返回值数量解包
    result = torch.linalg.lstsq(XtWX, XtWz)
    theta = result.solution
    a, b, c = theta[0], theta[1], theta[2]
    r = torch.sqrt(a ** 2 + b ** 2 - c)
    residuals = torch.sqrt((x - a) ** 2 + (y - b) ** 2) - r
    return residuals


def get_newWeights(weights, batch_x, batch_y, true_center, radius):
    # 计算每个点到圆心的距离
    dx = batch_x - true_center[0]
    dy = batch_y - true_center[1]
    distance = torch.sqrt(dx ** 2 + dy ** 2)
    mean_distance = torch.mean(distance)
    # 计算距离过小的样本点的惩罚因子
    min_distance = radius - 0.12 * mean_distance
    max_distance = radius + 0.12 * mean_distance
    penalty_min = torch.where(distance < min_distance, torch.abs(distance - distance/mean_distance), torch.zeros_like(distance))
    penalty_max = torch.where(distance > max_distance, torch.abs(distance - distance/mean_distance), torch.zeros_like(distance))
    penalty = penalty_min + penalty_max
    # 应用惩罚因子到权重上
    return weights - weights * penalty.view(-1, 1)



# 训练函数
def train(model, data_loader, optimizer, epochs):
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in data_loader:
            optimizer.zero_grad()
            w_pred = model(torch.stack([batch_x, batch_y], dim=1))
            residuals = weighted_least_squares_fit(batch_x, batch_y, w_pred)
            loss = torch.sum(w_pred * residuals)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        loss_history.append(avg_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
    return loss_history


# 生成模拟数据
def generate_data(num_points, center_x, center_y, radius, noise_std):
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = center_x + radius * np.cos(angles) + np.random.normal(0, noise_std, num_points)
    y = center_y + radius * np.sin(angles) + np.random.normal(0, noise_std, num_points)
    return x, y


# 主函数
def main():
    # 生成模拟数据
    num_points = 500
    center_x = 2.0
    center_y = 3.0
    radius = 1.5
    noise_std = 0.3
    batch_size = 30
    lr = 1e-4
    epochs = 500
    x, y = generate_data(num_points, center_x, center_y, radius, noise_std)

    # 转换为 PyTorch 张量
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    true_center = torch.tensor([center_x, center_y], dtype=torch.float32)  # 真实圆心

    # 创建数据集和数据加载器
    dataset = TensorDataset(x_tensor, y_tensor)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型和优化器
    model = WeightPredictor()
    optimizer = optim.Adam(model.parameters(), lr)

    # 训练模型

    loss_history = train(model, data_loader, optimizer, epochs)

    # 最终预测和拟合
    with torch.no_grad():
        w_pred = model(torch.stack([x_tensor, y_tensor], dim=1))

    # 可视化采样点分布和拟合圆
    plt.figure(figsize=(12, 6), dpi=300)
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, c=w_pred, cmap='viridis', alpha=0.8)
    plt.colorbar(label='Weight')
    plt.title('Sampling Points with Predicted Weights and Fitted Circle')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 可视化损失函数曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
