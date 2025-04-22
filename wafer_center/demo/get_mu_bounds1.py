def get_mu_bounds1(Sxy):
    if Sxy < 0:
        lb1 = 0
        ub1 = 0.5 * np.pi
        lb2 = np.pi
        ub2 = 1.5 * np.pi
    else:
        lb1 = 0.5 * np.pi
        ub1 = np.pi
        lb2 = 1.5 * np.pi
        ub2 = 2 * np.pi

    return lb1, lb2, ub1, ub2
