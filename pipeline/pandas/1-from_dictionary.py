#!/usr/bin/env python3
"""
Module that creates a pandas DataFrame from a dictionary
"""

import pandas as pd

# Create the dictionary with the specified data
data = {"First": [0.0, 0.5, 1.0, 1.5],
        "Second": ["one", "two", "three", "four"]}

# Create the DataFrame with custom row labels
df = pd.DataFrame(data, index=["A", "B", "C", "D"])
