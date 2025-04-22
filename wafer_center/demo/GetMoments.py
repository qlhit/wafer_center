import numpy as np

def GetMoments(x, y):
    # Computing the moments of X and Y
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    Sigma = np.cov(x, y, bias=True)  # bias=True for population covariance
    print(Sigma)
    Sxx = Sigma[0, 0]
    Syy = Sigma[1, 1]
    Sxy = Sigma[0, 1]
    print(np.cov(x,bias=True))
    print(np.cov(y,bias=True))
    return x_bar, y_bar, Sxx, Syy, Sxy

if __name__ == "__main__":
    # 示例数据：圆周坐标点
    points = np.array([
        [1, 1],
        [2, 1],
        [3, 1]
    ])
    print(points)
    print(GetMoments(points[:,0],points[:,1]))