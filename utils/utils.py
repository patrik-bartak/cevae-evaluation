"""
This file defines some utility functions useful throughout the entire project.
"""

import sys
import os

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
from typing import *

import torch
from pyro import distributions as dist


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


def get_simple_latent_and_proxy(dimensions):
    """
    Return latent distributions and proxy for the simple experiment
    :param dimensions: Number of data dimensions
    :return: the distributions and proxy function
    """
    proxy_noise_weights = [
        0.04, 0.17, 0.19, 0.10, 0.11, 0.13, 0.22, 0.14, 0.08,
    ]
    # proxy_noise_weights = [
    #     10.04, 12.17, 15.19, 10.10, 13.11, 14.13, 0.22, 0.14, 0.08,
    # ]
    if dimensions == 1:
        distributions = [
            lambda: dist.Normal(0.8, 1).sample().cpu().item(),
        ]
        proxy_function = lambda z: [
            [dist.Normal(z[0], proxy_noise_weights[0]).sample().cpu().item()],
            [dist.Normal(z[0] + z[0], proxy_noise_weights[1]).sample().cpu().item()],
            [dist.Normal(z[0], proxy_noise_weights[2]).sample().cpu().item()],
            [dist.Normal(z[0], proxy_noise_weights[3]).sample().cpu().item()],
            [dist.Normal(z[0] + z[0], proxy_noise_weights[4]).sample().cpu().item()],
            [dist.Normal(z[0], proxy_noise_weights[5]).sample().cpu().item()],
            [dist.Normal(z[0], proxy_noise_weights[6]).sample().cpu().item()],
            [dist.Normal(z[0], proxy_noise_weights[7]).sample().cpu().item()],
            [dist.Normal(z[0] + z[0], proxy_noise_weights[8]).sample().cpu().item()],
        ]
    elif dimensions == 3:
        distributions = [
            lambda: dist.Normal(1, 0.5).sample().cpu().item(),
            lambda: dist.Normal(-3, 0.5).sample().cpu().item(),
            lambda: dist.Normal(3, 1).sample().cpu().item(),
        ]
        proxy_function = lambda z: [
            [dist.Normal(z[1], proxy_noise_weights[0]).sample().cpu().item()],
            [dist.Normal(z[0] + z[1], proxy_noise_weights[1]).sample().cpu().item()],
            [dist.Normal(z[2], proxy_noise_weights[2]).sample().cpu().item()],
            [dist.Normal(z[0], proxy_noise_weights[3]).sample().cpu().item()],
            [dist.Normal(z[1] + z[2], proxy_noise_weights[4]).sample().cpu().item()],
            [dist.Normal(z[2], proxy_noise_weights[5]).sample().cpu().item()],
            [dist.Normal(z[0], proxy_noise_weights[6]).sample().cpu().item()],
            [dist.Normal(z[1], proxy_noise_weights[7]).sample().cpu().item()],
            [dist.Normal(z[2] + z[0], proxy_noise_weights[8]).sample().cpu().item()],
        ]
    elif dimensions == 0:
        distributions = [
            lambda: dist.Normal(1, 0.5).sample().cpu().item(),
            lambda: dist.Normal(-3, 0.5).sample().cpu().item(),
            lambda: dist.Normal(3, 1).sample().cpu().item(),
        ]
        proxy_function = lambda z: [
            [z[0]],
            [z[1]],
            [z[2]],
        ]
    else:
        raise ValueError(f"Invalid number of dimensions for the latent space: {dimensions}")
    print(f"Num Dimensions: {dimensions}")
    return distributions, proxy_function


def get_complex_latent_and_proxy(dimensions, sample_size):
    """
    Return latent distributions and proxy for the complex experiment
    :param dimensions: Number of data dimensions
    :return: the distributions and proxy function
    """
    proxy_noise_weights = [
        0.04, 0.17, 0.19, 0.10, 0.11, 0.13, 0.22, 0.14, 0.08,
    ]
    if dimensions == 1:
        distributions = [
            lambda: dist.Normal(0.8, 1).sample().item(),
        ]
        proxy_function = lambda z: [
            [dist.Normal(z[0] * z[0] - z[0], proxy_noise_weights[0]).sample().item()],
            [dist.Normal(torch.sigmoid(z[0] / z[0] + -3), proxy_noise_weights[1]).sample().item()],
            [dist.Normal(z[0] * 2 + z[0] + z[0], proxy_noise_weights[2]).sample().item()],
            [dist.Normal(torch.sigmoid(z[0] * z[0]), proxy_noise_weights[3]).sample().item()],
            [dist.Normal(z[0] * z[0] + 2, proxy_noise_weights[4]).sample().item()],
            [dist.Normal(z[0] * 0.2 + z[0] * 2 + z[0], proxy_noise_weights[5]).sample().item()],
            [dist.Normal(np.sin(z[0] * z[0]), proxy_noise_weights[6]).sample().item()],
            [dist.Normal(z[0] / z[0] + -5, proxy_noise_weights[7]).sample().item()],
            [dist.Normal(z[0] + z[0] + z[0], proxy_noise_weights[8]).sample().item()],
        ]
    elif dimensions == 3:
        distributions = [
            lambda: dist.Normal(1, 0.5).expand([sample_size, 1]).sample(),
            lambda: dist.Normal(-3, 0.5).expand([sample_size, 1]).sample(),
            lambda: dist.Normal(3, 1).expand([sample_size, 1]).sample(),
        ]
        proxy_function = lambda z: [
            [dist.Normal(z[1] * z[2] - z[0], proxy_noise_weights[0]).expand([sample_size, 1]).sample()],
            [dist.Normal(torch.sigmoid(z[0] / z[2] + -3), proxy_noise_weights[1]).expand([sample_size, 1]).sample()],
            [dist.Normal(z[2] * 2 + z[0] + z[0], proxy_noise_weights[2]).expand([sample_size, 1]).sample()],
            [dist.Normal(torch.sigmoid(z[0] * z[1]), proxy_noise_weights[3]).expand([sample_size, 1]).sample()],
            [dist.Normal(z[1] * z[0] + 2, proxy_noise_weights[4]).expand([sample_size, 1]).sample()],
            [dist.Normal(z[2] * 0.2 + z[1] * 2 + z[0], proxy_noise_weights[5]).expand([sample_size, 1]).sample()],
            [dist.Normal(np.sin(z[0] * z[0]), proxy_noise_weights[6]).expand([sample_size, 1]).sample()],
            [dist.Normal(z[1] / z[2] + -5, proxy_noise_weights[7]).expand([sample_size, 1]).sample()],
            [dist.Normal(z[2] + z[0] + z[1], proxy_noise_weights[8]).expand([sample_size, 1]).sample()],
        ]
    elif dimensions == 9:
        distributions = [
            lambda: dist.Normal(1, 0.5).sample().item(),
            lambda: dist.Normal(-3, 0.5).sample().item(),
            lambda: dist.Normal(3, 1).sample().item(),
            lambda: dist.Normal(-2, 0.5).sample().item(),
            lambda: dist.Normal(-5, 1.2).sample().item(),
            lambda: dist.Normal(4, 1).sample().item(),
            lambda: dist.Normal(2, 0.2).sample().item(),
            lambda: dist.Normal(0, 0.3).sample().item(),
            lambda: dist.Normal(0.5, 1).sample().item(),
        ]
        proxy_function = lambda z: [
            [dist.Normal(z[1] * z[2], proxy_noise_weights[0]).sample().item()],
            [dist.Normal(z[0] * np.sum(z[4:8]), proxy_noise_weights[1]).sample().item()],
            [dist.Normal(np.sum(z[0:4]), proxy_noise_weights[2]).sample().item()],
            [dist.Normal(z[8] * z[7] + np.sum(z[5:7]), proxy_noise_weights[3]).sample().item()],
            [dist.Normal(z[1] * z[0] + np.sum(z[6:8]), proxy_noise_weights[4]).sample().item()],
            [dist.Normal(z[2] + z[1] + z[0], proxy_noise_weights[5]).sample().item()],
            [dist.Normal(z[3] * z[6], proxy_noise_weights[6]).sample().item()],
            [dist.Normal(z[1] * np.sum(z[3:5]), proxy_noise_weights[7]).sample().item()],
            [dist.Normal(z[2] + z[0] + z[1], proxy_noise_weights[8]).sample().item()],
        ]
    elif dimensions == 0:
        distributions = [
            lambda: dist.Normal(1, 0.5).sample().item(),
            lambda: dist.Normal(-3, 0.5).sample().item(),
            lambda: dist.Normal(3, 1).sample().item(),
        ]
        proxy_function = lambda z: [
            [z[0]],
            [z[1]],
            [z[2]],
        ]
    else:
        raise ValueError(f"Invalid number of dimensions for the latent space: {dimensions}")
    print(f"Num Dimensions: {dimensions}")
    return distributions, proxy_function


def get_complex_second_latent_and_proxy(dimensions):
    """
    Return latent distributions and proxy for the second complex experiment
    :param dimensions: Number of data dimensions
    :return: the distributions and proxy function
    """
    proxy_noise_weights = [
        3.15, 1.26, 10.37, 2.65, 12.86, 15.24, 20.43, 1.76, 2.84,
    ]
    if dimensions == 1:
        distributions = [
            lambda: dist.Normal(0.8, 1).sample().cpu().item(),
        ]
        proxy_function = lambda z: [
            [dist.Normal(z[0] * z[0], proxy_noise_weights[0]).sample().cpu().item()],
            [dist.Normal(z[0] * z[0] + -3, proxy_noise_weights[1]).sample().cpu().item()],
            [dist.Normal(z[0] + z[0] + z[0], proxy_noise_weights[2]).sample().cpu().item()],
            [dist.Normal(z[0] * z[0], proxy_noise_weights[3]).sample().cpu().item()],
            [dist.Normal(z[0] * z[0] + 2, proxy_noise_weights[4]).sample().cpu().item()],
            [dist.Normal(z[0] + z[0] + z[0], proxy_noise_weights[5]).sample().cpu().item()],
            [dist.Normal(z[0] * z[0], proxy_noise_weights[6]).sample().cpu().item()],
            [dist.Normal(z[0] * z[0] + -5, proxy_noise_weights[7]).sample().cpu().item()],
            [dist.Normal(z[0] + z[0] + z[0], proxy_noise_weights[8]).sample().cpu().item()],
        ]
    elif dimensions == 3:
        distributions = [
            lambda: dist.Normal(1, 0.5).sample().cpu().item(),
            lambda: dist.Normal(-3, 0.5).sample().cpu().item(),
            lambda: dist.Normal(3, 1).sample().cpu().item(),
        ]
        proxy_function = lambda z: [
            [dist.Normal(z[1] * z[2], proxy_noise_weights[0]).sample().cpu().item()],
            [dist.Normal(z[0] * z[2] + -3, proxy_noise_weights[1]).sample().cpu().item()],
            [dist.Normal(z[2] + z[0] + z[0], proxy_noise_weights[2]).sample().cpu().item()],
            [dist.Normal(z[0] * z[1], proxy_noise_weights[3]).sample().cpu().item()],
            [dist.Normal(z[1] * z[0] + 2, proxy_noise_weights[4]).sample().cpu().item()],
            [dist.Normal(z[2] + z[1] + z[0], proxy_noise_weights[5]).sample().cpu().item()],
            [dist.Normal(z[0] * z[0], proxy_noise_weights[6]).sample().cpu().item()],
            [dist.Normal(z[1] * z[2] + -5, proxy_noise_weights[7]).sample().cpu().item()],
            [dist.Normal(z[2] + z[0] + z[1], proxy_noise_weights[8]).sample().cpu().item()],
        ]
    elif dimensions == 0:
        distributions = [
            lambda: dist.Normal(1, 0.5).sample().cpu().item(),
            lambda: dist.Normal(-3, 0.5).sample().cpu().item(),
            lambda: dist.Normal(3, 1).sample().cpu().item(),
        ]
        proxy_function = lambda z: [
            [z[0]],
            [z[1]],
            [z[2]],
        ]
    else:
        raise ValueError(f"Invalid number of dimensions for the latent space: {dimensions}")
    print(f"Num Dimensions: {dimensions}")
    return distributions, proxy_function
