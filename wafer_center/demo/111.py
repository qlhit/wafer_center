import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm
from scipy.signal import savgol_filter
from matplotlib import rcParams
# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import numpy as np
from scipy.stats import t

def spectral_ci(data, confidence=0.95):
    """
    计算光谱信号置信区间
    :param data: (n_samples, n_points) 二维数组
    :return: (means, lower, upper)
    """
    n = data.shape[0]
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0, ddof=1)
    std_errs = stds / np.sqrt(n)

    # 自动选择分布临界值
    if n >= 30:
        from scipy.stats import norm
        crit_val = norm.ppf(1 - (1-confidence)/2)
    else:
        crit_val = t.ppf(1 - (1-confidence)/2, n-1)

    return means, means - crit_val*std_errs, means + crit_val*std_errs
import matplotlib.pyplot as plt
wavelength = np.linspace(400, 700, 200)  # 可见光波长范围
plt.figure(figsize=(12,6))
plt.fill_between(wavelength, lower, upper, alpha=0.3, color='#1f77b4')
plt.plot(wavelength, means, 'k-', lw=1.5)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.title('Optical Spectrum with 95% Confidence Band')