from scipy.special import gammaincc, gammainc, gamma
import torch

# upper incomplete gamma function
def gamma_upper(a, x):
    return gammaincc(a, x)


# lower incomplete gamma function
def gamma_lower(a, x):
    return gammainc(a, x)


def gamma_complete(a):
    return gamma(a)
