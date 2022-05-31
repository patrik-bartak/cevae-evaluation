"""Utils

This file defines some functions useful throughout the entire project that do not fit anywhere else.

This file can also be imported as a module and contains the following
functions and classes:

    * HiddenPrints - blocks printing
    * save_pandas_table - function to save a pandas table in a specific directory
    * compact_dict_print - creates a string defining a dictionary without any illegal characters
    * select_features - selects only features from a pandas dataframe
    * generate_coverage_of_model_graph - generates a plot of coverage over first two features based on model outputs
"""

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import *


class HiddenPrints:
    """
    Class to block printing.
    Taken from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def save_pandas_table(dir: str, df: pd.DataFrame):
    plt.clf()
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    fig.set_figheight(30)
    fig.set_figwidth(50)
    tbl = ax.table(cellText=np.around(df.values, 4), colLabels=df.columns, rowLabels=df.index, loc='center')
    tbl.set_fontsize(40)
    tbl.scale(0.4, 8)
    fig.savefig(dir + '.png')
    df.to_csv(dir + '.csv')


def compact_dict_print(dict: Dict[str, Any]):
    result = ''
    for index, key in enumerate(dict):
        result += f'{key}={dict[key]}{"," if index < len(dict) - 1 else ""}'.replace(' ', '_').replace(':', '-')
    return result


def select_features(df: pd.DataFrame, first_n_dims: int = -1):
    if first_n_dims == -1:
        return df[[name for name in df.columns if 'feature' in name and 'proxy' not in name]]
    else:
        raise Exception("Not implemented")


def select_proxies(df: pd.DataFrame, first_n_dims: int = -1):
    if first_n_dims == -1:
        return df[[name for name in df.columns if 'proxy' in name]]
    else:
        raise Exception("Not implemented")
