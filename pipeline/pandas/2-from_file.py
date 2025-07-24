#!/usr/bin/env python3
"""
Module that contains a function to load data from a file as a pandas DataFrame
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame

    Args:
        filename: the file to load from
        delimiter: the column separator

    Returns:
        pd.DataFrame: the loaded DataFrame
    """
    return pd.read_csv(filename, delimiter=delimiter)
