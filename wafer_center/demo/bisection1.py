def bisection1(a, b, tol, f, max_iter):
    """
    Bisection method for finding roots of a function.
    Parameters:
    a : float : left side
    b : float : right side
    tol : float : tolerance
    f : function : function handle
    max_iter : int : maximum possible iterations
    Returns:
    res : float : root found
    flag : int : status flag (0 for success, 1 for failure)
    """
    # Initial points check
    if f(a) * f(b) > 0:
        while (f(a) * f(b) > 0) and (a <= b):
            a += 0.01
    if f(a) * f(b) > 0:
        res = 1
        flag = 1
    else:
        # Iterations
        flag = 0
        iter_count = 1
        c = (a + b) / 2  # Calculate the midpoint
        while iter_count <= max_iter and abs(f(c)) >= tol:
            iter_count += 1
            if (f(c) * f(a)) > 0:
                a = c
            else:
                b = c
            c = (a + b) / 2  # Update the midpoint
        if iter_count > max_iter:
            flag = 1
        res = c
    return res, flag
