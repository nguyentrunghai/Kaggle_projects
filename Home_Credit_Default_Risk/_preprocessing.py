"""
Define function to preprocess data
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

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

    num_df = df.drop(by, axis=1).select_dtypes("number")
    if num_df.shape[1] > 0:
        num_df = num_df.groupby(cols_to_group).agg(num_stats)
        num_df.columns = [prefix + col for col in flatten_multiindex_cols(num_df.columns)]

    else:
        print("No numerical columns in df")
        num_df = None

    cat_df = df.drop(by, axis=1).select_dtypes(["object", "category"])
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
    This class wraps around sklearn LabelEncoder and can handle a array of categorical
    or a dataframe of mixed types both numeric and categorical
    """
    def __init__(self):
        self._label_encoders = None
        self._shape = None
        self._ndim = None
        self._table_type = None
        self._col_names = None

    def fit(self, x):
        self._shape = x.shape
        self._ndim = len(self._shape)
        assert self._ndim <= 2, "Number of dimension must be <= 2."

        self._table_type = type(x)
        assert self._table_type in [pd.DataFrame, pd.Series, np.ndarray]

        if self._ndim == 1:
            self._label_encoders = LabelEncoder().fit(x)
            return self


        if self._table_type == np.ndarray:
            # assume that all columns are category
            ncols = self._shape[1]
            self._label_encoders = [LabelEncoder().fit(x[:, j]) for j in range(ncols)]

        else:
            self._col_names = x.columns
            cat_cols = x.select_dtypes(["category", "object", "bool"]).columns

            self._label_encoders = []
            for col in self._col_names:
                if col in cat_cols:
                    self._label_encoders.append(LabelEncoder().fit(x[col]))
                else:
                    self._label_encoders.append(None)
        return self

    def transform(self, x):
        assert len(x.shape) == self._ndim, "wrong number of dimension"
        assert self._table_type == type(x), "wrong table type"

        if self._ndim == 1:
            return self._label_encoders.transform(x)

        assert x.shape[1] == self._shape[1]
        end_coded_x = x.copy()
        if self._table_type == np.ndarray:
            for i, encoder in enumerate(self._label_encoders):
                if encoder is not None:
                    end_coded_x[:, i] = encoder.transform(end_coded_x[:, i])

        else:
            assert (x.columns == self._col_names).all(), "columns are not the same"
            for col, encoder in zip(self._col_names, self._label_encoders):
                if encoder is not None:
                    end_coded_x[col] = encoder.transform(end_coded_x[col])

        return end_coded_x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)



