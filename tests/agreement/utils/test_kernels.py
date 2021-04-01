import numpy as np
from agreement.utils.kernels import (
    compute_weights, identity_kernel, linear_kernel, quadratic_kernel, ordinal_kernel,
    radical_kernel, radio_kernel, circular_kernel, bipolar_kernel
)


def test_identity_kernel():
    assert np.array_equal(
        compute_weights(5, identity_kernel).round(2),
        np.array([
            [1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1.],
        ])
    )


def test_linear_kernel():
    assert np.array_equal(
        compute_weights(5, linear_kernel).round(2),
        np.array([
            [1.  , 0.75, 0.5 , 0.25, 0.  ],
            [0.75, 1.  , 0.75, 0.5 , 0.25],
            [0.5 , 0.75, 1.  , 0.75, 0.5 ],
            [0.25, 0.5 , 0.75, 1.  , 0.75],
            [0.  , 0.25, 0.5 , 0.75, 1.  ]
        ])
    )


def test_quadratic_kernel():
    assert np.array_equal(
        compute_weights(5, quadratic_kernel).round(2),
        np.array([
            [1.  , 0.94, 0.75, 0.44, 0.  ],
            [0.94, 1.  , 0.94, 0.75, 0.44],
            [0.75, 0.94, 1.  , 0.94, 0.75],
            [0.44, 0.75, 0.94, 1.  , 0.94],
            [0.  , 0.44, 0.75, 0.94, 1.  ]
        ])
    )


def test_ordinal_kernel():
    assert np.array_equal(
        compute_weights(5, ordinal_kernel).round(2),
        np.array([
            [1. , 0.9, 0.7, 0.4, 0. ],
            [0.9, 1. , 0.9, 0.7, 0.4],
            [0.7, 0.9, 1. , 0.9, 0.7],
            [0.4, 0.7, 0.9, 1. , 0.9],
            [0. , 0.4, 0.7, 0.9, 1. ]
        ])
    )


def test_radical_kernel():
    assert np.array_equal(
        compute_weights(5, radical_kernel).round(2),
        np.array([
            [1.  , 0.5 , 0.29, 0.13, 0.  ],
            [0.5 , 1.  , 0.5 , 0.29, 0.13],
            [0.29, 0.5 , 1.  , 0.5 , 0.29],
            [0.13, 0.29, 0.5 , 1.  , 0.5 ],
            [0.  , 0.13, 0.29, 0.5 , 1.  ]
        ])
    )


def test_radio_kernel():
    assert np.array_equal(
        compute_weights(5, radio_kernel).round(2),
        np.array([
            [1.  , 0.75, 0.44, 0.19, 0.  ],
            [0.75, 1.  , 0.91, 0.75, 0.59],
            [0.44, 0.91, 1.  , 0.95, 0.86],
            [0.19, 0.75, 0.95, 1.  , 0.97],
            [0.  , 0.59, 0.86, 0.97, 1.  ]
        ])
    )


def test_circular_kernel():
    assert np.array_equal(
        compute_weights(5, circular_kernel).round(2),
        np.array([
            [ 1.  ,  0.62,  0.  , -0.  ,  0.62],
            [ 0.62,  1.  ,  0.62,  0.  , -0.  ],
            [ 0.  ,  0.62,  1.  ,  0.62,  0.  ],
            [-0.  ,  0.  ,  0.62,  1.  ,  0.62],
            [ 0.62, -0.  ,  0.  ,  0.62,  1.  ]
        ])
    )


def test_bipolar_kernel():
    assert np.array_equal(
        compute_weights(5, bipolar_kernel).round(2),
        np.array([
            [1.  , 0.86, 0.67, 0.4 , 0.  ],
            [0.86, 1.  , 0.93, 0.75, 0.4 ],
            [0.67, 0.93, 1.  , 0.93, 0.67],
            [0.4 , 0.75, 0.93, 1.  , 0.86],
            [0.  , 0.4 , 0.67, 0.86, 1.  ]
        ])
    )
