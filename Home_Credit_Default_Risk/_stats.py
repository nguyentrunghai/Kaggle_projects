"""
define some statistic functions
"""

import numpy as np
import scipy


def weighted_variance(x, weights):
    average = np.average(x, weights=weights)
    variance = np.average((x - average)**2, weights=weights)
    return variance


def between_group_variance(group_averages, weights):
    return weighted_variance(group_averages, weights=weights)


def within_group_variance(group_variances, weights):
    return np.average(group_variances, weights=weights)


def f_ratio(group_averages, group_variances, weights):
    return between_group_variance(group_averages, weights) / within_group_variance(group_variances, weights)


def corrwith(df, series):
    """
    :param df: a dataframe
    :param series: a series
    :return: corrs, a series
    """
    corrs = df.corrwith(series)
    corrs = corrs.reset_index()
    corrs.columns = ["feature", "corr"]

    corrs["abs_corr"] = np.abs(corrs["corr"])
    corrs = corrs.sort_values("abs_corr", ascending=False)
    corrs = corrs.set_index("feature")

    corrs = corrs.drop(["abs_corr"], axis=1)

    return corrs["corr"]


def mode(x):
    return scipy.stats.mode(x)[0]

