#!/usr/bin/env python3
"""
Module that contains a function to create a pandas DataFrame from a numpy array
"""

import pandas as pd
import string


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray

    Args:
        array: the np.ndarray from which to create the pd.DataFrame

    Returns:
        pd.DataFrame: the newly created DataFrame with alphabetically ordered,
                     capitalized column labels
    """
    # Get the number of columns from the array shape
    num_cols = array.shape[1] if len(array.shape) > 1 else 1

    # Generate column labels: A, B, C, ... up to the number of columns
    # Using string.ascii_uppercase to get uppercase letters
    columns = list(string.ascii_uppercase[:num_cols])

    # Create and return the DataFrame
    return pd.DataFrame(array, columns=columns)
