import numpy as np
from typing import Tuple


def _encode_one_hot(values: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Used to create a one-hot encoding table."""
    _temp = np.zeros((len(values), len(columns)))
    _temp[range(len(values)), values] = 1
    return _temp


def _get_inverted_index(x: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Used to create a inverted index for non-dense array."""
    cols = values[np.argsort(values)]
    return cols, np.searchsorted(cols, x)


def pivot_table_frequency(rows: np.ndarray, columns: np.ndarray, values: np.ndarray = None) -> np.ndarray:
    """
    Used to create a pivot table containing a frequency of each column value for each unique row value.
    Rows can be treated as keys and columns as values. This function creates an array, where each row
    contains a frequency of all possible values for one key.


    :param rows: specifies rows of the table
    :param columns: specifies columns of the table
    :param values: used to provide a predefined list of values for columns.
    Can be used when some of the options don't appear in the columns data.
    :return: a pivot table containing frequencies
    """
    rows, row_pos = np.unique(rows, return_inverse=True)
    if values is None:
        cols, col_pos = np.unique(columns, return_inverse=True)
    else:
        cols, col_pos = _get_inverted_index(columns, values)
    return _encode_one_hot(row_pos, rows).T.dot(_encode_one_hot(col_pos, cols))

