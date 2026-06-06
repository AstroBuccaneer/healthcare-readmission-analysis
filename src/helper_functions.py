# helper_functions.py
# Utility functions for the healthcare readmission project
# To be expanded in future iterations

import pandas as pd
import numpy as np


def load_data(filepath):
    """Load dataset from a given filepath."""
    return pd.read_csv(filepath)


def summarize_dataframe(df):
    """Print a quick summary of a dataframe."""
    print("Shape:", df.shape)
    print("Nulls:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("Dtypes:\n", df.dtypes.value_counts())