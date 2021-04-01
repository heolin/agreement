import numpy as np
import math

"""
Weights are used to model situations, where categories are represented as (at least) ordinal data.
Using this approach, the agreement between raters is not binary, but it differs depending on the
weights between chosen categories.

For example, while modeling ratio categories with possible values: `{1, 2, 3, 4, 5}`, values 
`5` and `4` are will cause higher agreement than `5` and `1`.

This implementation assumes all categories are evenly distributed. That means the distance between
each category is exactly the same. You can change that by implementing a custom weight kernel.

There is no formal rule that can be used for deciding which set weights should be used
in a particular inter-rater reliability study. The researcher needs to determine how
much each type of disagreement (or partial agreement) should impact the agreement
coefficient, and select the set of weights accordingly.
"""


def identity_kernel(x: int, y: int, q: int) -> float:
    """
    The default kernel, which don't introduce any weights between categories.
    Two values get any positive agreement only if they have exactly the same category.
    """
    return float(x == y)


def linear_kernel(x: int, y: int, q: int) -> float:
    """
    The most basic weight kernel.
    The weights' values depend on whether the values are of alphabetic or numeric type.
    The linear weights are generally smaller than the quadratic weights.
    """
    return 1 - abs(x - y) / (q - 1)


def quadratic_kernel(x: int, y: int, q: int) -> float:
    return 1 - (x - y)**2 / (q - 1)**2


def radical_kernel(x: int, y: int, q: int) -> float:
    """
    This kernel can be used if you find quadratic and linear weights too large.
    """
    return 1 - np.sqrt(abs(x - y)) / np.sqrt((q - 1))


def radio_kernel(x: int, y: int, q: int) -> float:
    """
    This kernel can be used with data of ratio type.
    Ratio weights evaluate the differences between scores relative to their magnitudes.
    """
    _x, _y = x + 1, y + 1
    return 1 - ((_x - _y)/(_x + _y))**2 / ((q - 1)/(q + 1))**2


def ordinal_kernel(x: int, y: int, q: int) -> float:
    """
    This kernel can be used for non-numeric ordinal values.
    The use of ordinal weights requires the categories to be at least of ordinal type.
    So you should be able to rank all categories from the smallest to the largest.
    """
    def m(k, l):
        return math.comb(max(k, l) - min(k, l) + 1, 2)

    if x == y:
        return 1.0
    else:
        return 1 - m(x, y) / m(1, q)


def circular_kernel(x: int, y: int, q: int) -> float:
    """
    These weights would be recommendedif the rating represents the magnitude
    of an angle expressed in degrees or in radians.
    """
    return 1 - (np.sin(np.pi * (x - y) / q) / np.sin(np.pi * (int(q / 2)) / q))**2


def bipolar_kernel(x: int, y: int, q: int) -> float:
    """
    The bipolar weights, which behave like ratio weights at the center of the scale and
    like quadratic weights towards the ends.
    """
    if x == y:
        return 1.0
    else:
        _x, _y = x + 1, y + 1
        return 1 - (_x - _y)**2 / ((_x + _y - 2) * (2 * q - _x - _y))


def compute_weights(q: int, kernel):
    return np.array([
        [
            kernel(x, y, q) for x in range(q)
        ] for y in range(q)
    ])
