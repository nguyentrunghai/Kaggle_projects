"""
define some statistic functions
"""

import numpy as np


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
