"""
Define function to preprocess data
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


def flatten_multiindex_cols(columns):
    fat_cols = ["_".join([str(c) for c in flat_col]) for flat_col in columns.to_flat_index()]
    return fat_cols


def aggregate(df, by,
              dtype="all",
              num_stats=None, cat_stats=None,
              onehot_encode=True,
              drop_collin_cols=True):
    """
    :param df: dataframe
    :param by: list of column names on which groupby is done
    :param dtype: str, either "all", "num" or "cat"
    :param num_stats: list of aggregation statistic functions for numerical columns
    :param cat_stats: list of aggregation statistic functions for categorical columns
    :param onehot_encode: bool, whether categorical columns are onehot-encoded
    :param drop_collin_cols: bool, whether to drop collinear columns
    :return agg_df: new dataframe
    """
    assert dtype in ["all", "num", "cat"]
    assert type(by) in [list, tuple], "by must be a list or tuple"
    assert type(num_stats) in [list, tuple], "num_stats must be a list or tuple"
    assert type(cat_stats) in [list, tuple], "cat_stats must be a list or tuple"

    cols_to_group = [df[col] for col in by]

    num_df = df.drop(by, axis=1).select_dtypes("number")
    if num_df.shape[1] > 0:
        num_df = num_df.groupby(cols_to_group).agg(num_stats)
        num_df.columns = [col for col in flatten_multiindex_cols(num_df.columns)]

    else:
        print("No numerical columns in df")
        num_df = None

    cat_df = df.drop(by, axis=1).select_dtypes(["object", "category", "bool"])
    if cat_df.shape[1] > 0:
        if onehot_encode:
            cat_df = pd.get_dummies(cat_df)

        cat_df = cat_df.groupby(cols_to_group).agg(cat_stats)
        cat_df.columns = [col for col in flatten_multiindex_cols(cat_df.columns)]

    else:
        print("No categorical columns in df")
        cat_df = None

    if (num_df is None) and (cat_df is None):
        return None

    elif num_df is None:
        df = cat_df.reset_index()

    elif cat_df is None:
        df = num_df.reset_index()

    else:
        if dtype == "num":
            df = num_df
        elif dtype == "cat":
            df = cat_df
        else:
            df = num_df.merge(cat_df, how="outer", left_index=True, right_index=True)
        df = df.reset_index()

    if drop_collin_cols:
        df = drop_collinear_columns(df, threshold=0.9999)
    return df


def drop_collinear_columns(df, threshold):
    """
    :param df: dataframe
    :param threshold: float, 0 <= threshold <= 1
    :return: dataframe
    """
    df = df.select_dtypes(["number"])
    corr_matr = df.corr().abs()
    upper_matr = corr_matr.where(np.triu(np.ones(corr_matr.shape), k=1).astype(np.bool))
    dropped_cols = [col for col in upper_matr.columns if (upper_matr[col] > threshold).any()]
    print("Drop %d collinear columns" % len(dropped_cols))
    return df.drop(dropped_cols, axis=1)


def change_dtypes(df):
    """
    change types of columns to reduce memory size
    :param df: dataframe
    :return df: dataframe
    """
    memory = df.memory_usage().sum() / 10**6
    print("Memory usage before changing types %0.2f MB" % memory)

    for col in df.columns:
        if (df[col].dtype == "object") and (df[col].nunique() < df.shape[0]):
            df[col] = df[col].astype("category")

        elif list(df[col].unique()) == [1, 0]:
            df[col] = df[col].astype(bool)

        elif df[col].dtype == float:
            df[col] = df[col].astype(np.float32)

        elif df[col].dtype == int:
            df[col] = df[col].astype(np.int32)

    memory = df.memory_usage().sum() / 10 ** 6
    print("Memory usage after changing types %0.2f MB" % memory)
    return df


def deal_with_abnormal_days_employed(df):
    """
    :param df: dataframe
    :return df: data frame
    """
    key = "DAYS_EMPLOYED"
    abnormal_val = df[key].max()
    df[key + "_ABNORMAL"] = df[key] == abnormal_val
    df[key].replace({abnormal_val: np.nan}, inplace=True)
    return df


def onehot_encoding(X_train, X_test):
    """
    :param X_train: dataframe
    :param X_test: dataframe
    :return: (X_train_ohe, X_test_ohe)
    """
    X_train_ohe = pd.get_dummies(X_train)
    X_test_ohe = pd.get_dummies(X_test)

    X_train_ohe, X_test_ohe = X_train_ohe.align(X_test_ohe, join='inner', axis=1)
    return X_train_ohe, X_test_ohe


class GeneralLabelEncoder:
    """
    sklearn LabelEncoder accepts only 1D array or pd Series.
    This class wraps around sklearn LabelEncoder and can handle a 2d array of categoricals
    or a dataframe of mixed types both numeric and categorical
    """
    def __init__(self, fillna_value="NaN"):
        self._fillna_value = fillna_value
        self._label_encoders = None
        self._shape = None
        self._ndim = None
        self._table_type = None
        self._col_names = None

    def _fillna(self, a):
        b = a.astype("object")
        b[b.isnull()] = self._fillna_value
        return b.astype("category")

    def fit(self, x):
        self._shape = x.shape
        self._ndim = len(self._shape)
        assert self._ndim <= 2, "Number of dimension must be <= 2."

        self._table_type = type(x)
        assert self._table_type in [pd.DataFrame, pd.Series, np.ndarray]

        if self._ndim == 1:
            self._label_encoders = LabelEncoder().fit(self._fillna(x))
            return self


        if self._table_type == np.ndarray:
            # assume that all columns are category
            ncols = self._shape[1]
            self._label_encoders = [LabelEncoder().fit(self._fillna(x[:, j])) for j in range(ncols)]

        else:
            self._col_names = x.columns
            cat_cols = x.select_dtypes(["category", "object", "bool"]).columns

            self._label_encoders = []
            for col in self._col_names:
                if col in cat_cols:
                    self._label_encoders.append(LabelEncoder().fit(self._fillna(x[col])))
                else:
                    self._label_encoders.append(None)
        return self

    def transform(self, x):
        assert len(x.shape) == self._ndim, "wrong number of dimension"
        assert self._table_type == type(x), "wrong table type"

        if self._ndim == 1:
            return self._label_encoders.transform(self._fillna(x))

        assert x.shape[1] == self._shape[1]
        encoded_x = x.copy()
        if self._table_type == np.ndarray:
            for i, encoder in enumerate(self._label_encoders):
                if encoder is not None:
                    encoded_x[:, i] = encoder.transform(self._fillna(encoded_x[:, i]))

        else:
            assert (x.columns == self._col_names).all(), "columns are not the same"
            for col, encoder in zip(self._col_names, self._label_encoders):
                if encoder is not None:
                    encoded_x[col] = encoder.transform(self._fillna(encoded_x[col]))

        return encoded_x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
