import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# 1. 生成带噪声的圆周数据
def generate_circle_data(num_points=1000, r_true=2.0, noise=0.1):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = r_true * np.cos(theta) + np.random.normal(0, noise, num_points)
    y = r_true * np.sin(theta) + np.random.normal(0, noise, num_points)
    return torch.FloatTensor(np.stack([x, y], axis=1)), r_true

# 2. 定义神经网络模型
class CircleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # 输入形状：[batch_size, 2]
        r_pred = self.fc_layers(x)  # 各点独立预测
        return r_pred  # 输出形状：[batch_size, 1]

# 3. 初始化组件
def init_components(data, batch_size=64):
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = CircleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model, loader, optimizer

# 4. 最小二乘损失函数
def least_square_loss(inputs, r_pred):
    x_sq = inputs[:, 0].pow(2).unsqueeze(1)
    y_sq = inputs[:, 1].pow(2).unsqueeze(1)
    return torch.mean((x_sq + y_sq - r_pred.pow(2) - 2) ** 2)

# 5. 训练过程
def train_model(model, loader, optimizer, epochs=100):
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            inputs = batch[0]
            optimizer.zero_grad()
            r_pred = model(inputs)
            loss = least_square_loss(inputs, r_pred)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    return model, loss_history

# 6. 结果可视化
def visualize_results(data, r_pred, r_true, loss_history):
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, 'b')
    plt.title("Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")

    # 绘制圆周对比
    theta = np.linspace(0, 2*np.pi, 100)
    plt.subplot(1, 2, 2)
    plt.scatter(data[:,0], data[:,1], alpha=0.3, label="Noisy Data")
    plt.plot(r_true*np.cos(theta), r_true*np.sin(theta),
             'r--',  label="True Circle")
    plt.plot(r_pred*np.cos(theta), r_pred*np.sin(theta),
             'g-',  label="Predicted Circle")
    plt.axis('equal')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 参数设置
    data, r_true = generate_circle_data(noise=0.1)
    model, loader, optimizer = init_components(data, batch_size=32)

    # 训练模型
    trained_model, loss_hist = train_model(model, loader, optimizer, epochs=200)

    # 计算最终预测半径
    with torch.no_grad():
        r_pred = trained_model(data).mean().sqrt().item()

    # 可视化结果
    visualize_results(data.numpy(), r_pred, r_true, loss_hist)
    print(f"True Radius: {r_true:.2f} | Predicted Radius: {r_pred:.2f}")