import numpy as np
from scipy.stats import chi2

def UniformityTest(x, y, x_bar, y_bar, Sxx, Syy, Sxy):
    n = len(x)
    Var = np.mean(((x - x_bar)**2 + (y - y_bar)**2)**2) / 2
    Stats = (Sxx - Syy)**2 + 4 * Sxy**2
    stat = n * Stats / Var
    pval = chi2.sf(stat, 2)  # Use survival function for upper tail probability

    return pval, stat
