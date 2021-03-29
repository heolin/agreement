import numpy as np

from agreement.utils.transform import pivot_table_frequency


def test_pivot_table_frequency(dataset_gwet):

    # questions_answers_table,
    assert np.array_equal(
        pivot_table_frequency(dataset_gwet[:, 0], dataset_gwet[:, 2]),
        np.array([
            [3., 0., 0., 0., 0.],
            [0., 3., 1., 0., 0.],
            [0., 0., 4., 0., 0.],
            [0., 0., 4., 0., 0.],
            [0., 4., 0., 0., 0.],
            [1., 1., 1., 1., 0.],
            [0., 0., 0., 4., 0.],
            [3., 1., 0., 0., 0.],
            [0., 4., 0., 0., 0.],
            [0., 0., 0., 0., 3.],
            [2., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.]
        ])
    )

    #users_answers_table = pivot_table_frequency(dataset_gwet[:, 1], dataset_gwet[:, 2])
    assert np.array_equal(
        pivot_table_frequency(dataset_gwet[:, 1], dataset_gwet[:, 2]),
        np.array([
            [3., 3., 2., 1., 0.],
            [2., 4., 2., 1., 1.],
            [1., 3., 5., 1., 1.],
            [3., 3., 2., 2., 1.]
        ])
    )
