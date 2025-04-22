import numpy as np
from sklearn.utils import check_random_state
import cv2

def ransac_circle_fit(points, max_iters=100, inlier_thresh=0.1, min_samples=3):
    best_inliers = []
    best_center, best_radius = None, None
    cv2.findCi
    for _ in range(max_iters):
        # 随机采样3个点
        sample = check_random_state(0).choice(points, min_samples, replace=False)
        x, y = sample[:, 0], sample[:, 1]

        # 代数法拟合初始圆
        A = np.vstack([x, y, np.ones(min_samples)]).T
        B = -(x**2 + y**2)
        D, E, F = np.linalg.lstsq(A, B, rcond=None)[0]
        h, k = -D/2, -E/2
        r = np.sqrt(D**2 + E**2 - 4*F) / 2

        # 评估内点
        residuals = np.abs(np.sqrt((points[:,0]-h)**2 + (points[:,1]-k)**2) - r)
        inliers = np.where(residuals < inlier_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_center = (h, k)
            best_radius = r

    # 用所有内点重新拟合（可选）
    if best_inliers:
        inlier_points = points[best_inliers]
        x, y = inlier_points[:, 0], inlier_points[:, 1]
        A = np.vstack([x, y, np.ones(len(x))]).T
        B = -(x**2 + y**2)
        D, E, F = np.linalg.lstsq(A, B, rcond=None)[0]
        best_center = (-D/2, -E/2)
        best_radius = np.sqrt(D**2 + E**2 - 4*F) / 2

    return best_center, best_radius

# 示例使用
points = np.array([[0,0], [1,0], [0,1], [1,1], [5,5], [0.5,0.5]])
center, radius = ransac_circle_fit(points)
print(f"拟合圆心：{center}, 半径：{radius:.2f}")