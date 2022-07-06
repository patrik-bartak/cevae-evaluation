"""
This file defines some utility functions useful throughout the entire project.
"""

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import *


class HiddenPrints:
    """
    Blocks printing
    Taken from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def save_pandas_table(dir: str, df: pd.DataFrame):
    """
    Save a pandas table in a specific directory
    :param dir: Directory to save in
    :param df: Dataframe to save
    :return:
    """
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
    """
    Create a string defining a dictionary without any illegal characters
    :param dict: The dictionary
    :return:
    """
    result = ''
    for index, key in enumerate(dict):
        result += f'{key}={dict[key]}{"," if index < len(dict) - 1 else ""}'.replace(' ', '_').replace(':', '-')
    return result


def select_features(df: pd.DataFrame, first_n_dims: int = -1):
    """
    Select only features from a pandas dataframe
    :param df: The dataframe
    :param first_n_dims: Number of dimensions
    :return:
    """
    if first_n_dims == -1:
        return df[[name for name in df.columns if 'feature' in name and 'proxy' not in name]]
    else:
        raise Exception("Not implemented")


def select_proxies(df: pd.DataFrame, first_n_dims: int = -1):
    """
    Generate a plot of coverage over first two features based on model outputs
    :param df: The dataframe
    :param first_n_dims: Number of dimensions
    :return:
    """
    if first_n_dims == -1:
        return df[[name for name in df.columns if 'proxy' in name]]
    else:
        raise Exception("Not implemented")
