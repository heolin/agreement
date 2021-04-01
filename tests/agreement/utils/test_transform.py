import numpy as np

from agreement.utils.transform import pivot_table_frequency


def test_pivot_table_frequency(dataset_gwet):
    dataset = dataset_gwet

    # Tests creating a questions answers table
    assert np.array_equal(
        pivot_table_frequency(dataset[:, 0], dataset[:, 2]),
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

    # Tests creating an users answers table
    assert np.array_equal(
        pivot_table_frequency(dataset[:, 1], dataset[:, 2]),
        np.array([
            [3., 3., 2., 1., 0.],
            [2., 4., 2., 1., 1.],
            [1., 3., 5., 1., 1.],
            [3., 3., 2., 2., 1.]
        ])
    )


def test_pivot_table_frequency_missing_values(dataset_unused_values):
    dataset = dataset_unused_values

    assert pivot_table_frequency(dataset[:, 0], dataset[:, 2]).shape[1] == 2

    # Tests creating a questions answers table
    questions_answers_table = pivot_table_frequency(
        dataset[:, 0], dataset[:, 2], values=np.array([1, 2, 3]))
    assert questions_answers_table.shape[1] == 3

    assert np.array_equal(
        questions_answers_table,
        np.array([
            [0., 0., 2.],
            [0., 0., 2.],
            [0., 0., 2.],
            [0., 1., 1.],
            [0., 2., 0.],
            [0., 0., 2.]
        ])
    )

    # Tests creating a questions answers table
    assert np.array_equal(
        pivot_table_frequency(dataset[:, 1], dataset[:, 2], values=np.array([1, 2, 3])),
        np.array([
            [0., 2., 4.],
            [0., 1., 5.]
        ])
    )

