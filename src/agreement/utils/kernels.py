import numpy as np


def identity_kernel(x, y, q):
    return float(x == y)


def linear_kernel(x, y, q):
    return 1 - abs(x - y) / (q - 1)


def quadratic_kernel(x, y, q):
    return 1 - (x - y)**2 / (q - 1)**2


def get_weights(q, kernel):
    return np.array([
        [
            kernel(x, y, q) for x in range(q)
        ] for y in range(q)
    ])
