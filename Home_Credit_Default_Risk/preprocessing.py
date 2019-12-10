"""
Define function to preprocess data
"""

import pandas as pd
import numpy as np


def flatten_multiindex_cols(columns):
    fat_cols = ["_".join([str(c) for c in flat_col]) for flat_col in columns.to_flat_index()]
    return fat_cols


def aggregate(df, by, num_stats=("mean",), cat_stats=("count",), prefix=None):
    """
    :param df: dataframe
    :param by: list of column names on which groupby is done
    :param num_stats: list of aggregation statistic functions for numerical columns
    :param cat_stats: list of aggregation statistic functions for categorical columns
    :param prefix: str, to be appended at the begining of each new column names
    :return agg_df: new dataframe
    """
    if prefix is None:
        prefix = ""

    cols_to_group = [df[col] for col in by]

    num_df = df.drop(by, axis=1).select_dtypes('number')
    if num_df.shape[1] > 0:
        num_df = num_df.groupby(cols_to_group).agg(num_stats)
        num_df.columns = [prefix + col for col in flatten_multiindex_cols(num_df.columns)]

    else:
        print("No numerical columns in df")
        num_df = None

    cat_df = df.drop(by, axis=1).select_dtypes('object')
    if cat_df.shape[1] > 0:
        cat_df = pd.get_dummies(cat_df)
        cat_df = cat_df.groupby(cols_to_group).agg(cat_stats)
        cat_df.columns = [prefix + col for col in flatten_multiindex_cols(cat_df.columns)]

    else:
        print("No categorical columns in df")
        cat_df = None

    if (num_df is None) and (cat_df is None):
        return None

    if num_df is None:
        return cat_df.reset_index()

    if cat_df is None:
        return num_df.reset_index()

    merged_df = num_df.merge(cat_df, how="outer", left_index=True, right_index=True)
    return merged_df.reset_index()


def change_dtypes(df):
    """
    change types of columns to reduce memory size
    :param df: dataframe
    :return df: dataframe
    """
    memory = df.memory_usage().sum() / 10**6
    print("Memory usage before changing types %0.2f MB" % memory)

    for col in df.columns:
        if (df[col].dtype == 'object') and (df[col].nunique() < df.shape[0]):
            df[col] = df[col].astype('category')

        elif list(df[col].unique()) == [1, 0]:
            df[col] = df[col].astype(bool)

        elif df[col].dtype == float:
            df[col] = df[col].astype(np.float32)

        elif df[col].dtype == int:
            df[col] = df[col].astype(np.int32)

    memory = df.memory_usage().sum() / 10 ** 6
    print("Memory usage after changing types %0.2f MB" % memory)
    return df
