import numpy as np
from util import *
from scipy.optimize import minimize, root
from scipy.special import iv  # Bessel function of the first kind
class circle:
    def CircleMLE(self, x, y, r=1):
        # Checking input
        # if (x.ndim != 1) or (y.ndim != 1):
        #     raise ValueError('x and y must be numerical vectors')
        #  Reshaping x and y to be column vectors
        # x = x.reshape(-1, 1)
        # y = y.reshape(-1, 1)
        # Step 1 - Computing moments of X,Y
        x_bar, y_bar, Sxx, Syy, Sxy = GetMoments(x, y)
        # Step 2 - Testing for Uniformity
        P_val, stat = UniformityTest(x, y, x_bar, y_bar, Sxx, Syy, Sxy)
        print('P_val:',P_val)
        if P_val > 0.05:
            # Step 2.1
            sol = np.array([CircleMLE_Uniform(x, y)])
            return sol, P_val, stat
        # Step 3 - Estimating mu using the Spectral decomposition
        Mu, lam = FindMu(Sxx, Syy, Sxy)
        lambda_d = np.max(lam) - np.min(lam)
        # Step 4 - R_1, R_2 functions
        R1 = lambda kap: iv(1, kap) / iv(0, kap)
        R2 = lambda kap: iv(2, kap) / iv(0, kap)
        alpha = lambda kap: 1 - R1(kap)**2
        beta = lambda kap: R2(kap) - R1(kap)**2
        # Step 5 - Estimating equations for rho and sigma^2
        Rho = lambda kap: np.sqrt((np.max(lam) - np.min(lam)) / (R1(kap)**2 - R2(kap)))
        sigma_sqr = lambda kap: 0.5 * (Sxx + Syy) - 0.5 * (1 - R1(kap)**2) * Rho(kap)**2
        # Step 6 - Estimating kappa for each value of mu
        M1 = np.mean(np.exp(np.cos(Mu[0]) * ((x - x_bar) * r) + np.sin(Mu[0]) * ((y - y_bar) * r)))
        M2 = np.mean(np.exp(np.cos(Mu[1]) * ((x - x_bar) * r) + np.sin(Mu[1]) * ((y - y_bar) * r)))
        # Estimating equation for kappa
        f1 = lambda k: np.exp(sigma_sqr(k) * r**2 / 2 - Rho(k) * r * R1(k)) * iv(0, Rho(k) * r + k) / iv(0, k) - M1
        f2 = lambda k: np.exp(sigma_sqr(k) * r**2 / 2 - Rho(k) * r * R1(k)) * iv(0, Rho(k) * r + k) / iv(0, k) - M2
        # Kappa lower bound
        Kappa_lower_bound = lambda kap: -alpha(kap) / beta(kap) - (Sxx + Syy) / lambda_d
        kappa0 = bisection1(0, 500, 1e-10, Kappa_lower_bound, 100)
        ub = 500
        lb = kappa0
        # Estimating kappa using numerical method
        try:
            kappa_hat = [minimize(f1, kappa0, bounds=[(lb, ub)]).x[0], minimize(f2, kappa0, bounds=[(lb, ub)]).x[0]]
        except Exception as e:
            raise ValueError('MME of kappa is incomputable, try using scaling factor r=1e-1, r=1e-2 or larger if needed.')
        # Step 7 - Estimating sigma, rho, a and b for each possible value of mu
        rho_hat = Rho(kappa_hat)
        sigma_hat = sigma_sqr(kappa_hat)
        sigma_hat[sigma_hat < 0] = 0  # Set negative sigma^2 to 0
        sigma_hat = np.sqrt(sigma_hat)
        a_hat = x_bar - rho_hat * np.cos(Mu) * R1(kappa_hat)
        b_hat = y_bar - rho_hat * np.sin(Mu) * R1(kappa_hat)
        # Step 8 - Lower and upper bounds for optimization
        l1, l2, u1, u2 = get_mu_bounds1(Sxy)
        lb1 = [0, l1, 0, -np.inf, -np.inf, 0]
        ub1 = [ub, u1, np.inf, np.inf, np.inf, np.sqrt((Sxx + Syy) / 2)]
        lb2 = [0, l2, 0, -np.inf, -np.inf, 0]
        ub2 = [ub, u2, np.inf, np.inf, np.inf, np.sqrt((Sxx + Syy) / 2)]
        # Starting points (for each possible value of mu)
        x01 = [kappa_hat[0], Mu[0], rho_hat[0], a_hat[0], b_hat[0], sigma_hat[0] + np.random.exponential(1)]
        x02 = [kappa_hat[1], Mu[1], rho_hat[1], a_hat[1], b_hat[1], sigma_hat[0] + np.random.exponential(1)]
        # Step 9 - Minimizing the log likelihood numerically
        n = x.shape[0]
        VM = lambda xx: VM_negLogLike3(xx, x, y, n)
        # Minimization
        Theta1 = minimize(VM, x01, bounds=list(zip(lb1, ub1))).x
        Theta2 = minimize(VM, x02, bounds=list(zip(lb2, ub2))).x
        # Choosing optimal solution
        fval1 = VM(Theta1)
        fval2 = VM(Theta2)
        if fval2 > fval1:
            sol = np.reshape(Theta1, (6, 1))
        else:
            sol = np.reshape(Theta2, (6, 1))
        return sol, P_val, stat

circle = circle()
points = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0],
    [0.0, -1.0],
    [0.707, 0.707],
    [-0.707, 0.707],
    [-0.707, -0.707],
    [0.707, -0.707]
])
print(circle.CircleMLE(points[:,0],points[:,1]))

