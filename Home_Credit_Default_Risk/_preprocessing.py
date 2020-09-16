"""
Define function to preprocess data
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from _stats import mode
from _stats import mean_diff, var_diff, range_diff


def flatten_multiindex_cols(columns):
    fat_cols = ["_".join([str(c) for c in flat_col]) for flat_col in columns.to_flat_index()]
    return fat_cols


# TODO refactor this function, it is a mess!
# The "by" column disappears in the result when df does not contain cat columns
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

    cols_to_group = [df[col] for col in by]

    if dtype in ["all", "num"]:
        num_df = df.drop(by, axis=1).select_dtypes("number")
        if num_df.shape[1] > 0:
            assert type(num_stats) in [list, tuple], "num_stats must be a list or tuple"
            num_df = num_df.groupby(cols_to_group).agg(num_stats)
            num_df.columns = [col for col in flatten_multiindex_cols(num_df.columns)]

        else:
            print("No numerical columns in df")
            num_df = None
            if dtype == "num":
                return None

        if dtype == "num":
            df = num_df.reset_index()
            if drop_collin_cols:
                df = drop_collinear_columns(df, threshold=0.9999)
            return df

    if dtype in ["all", "cat"]:
        cat_df = df.drop(by, axis=1).select_dtypes(["object", "category", "bool"])
        if cat_df.shape[1] > 0:
            if onehot_encode:
                cat_df = pd.get_dummies(cat_df)

            assert type(cat_stats) in [list, tuple], "cat_stats must be a list or tuple"
            cat_df = cat_df.groupby(cols_to_group).agg(cat_stats)
            cat_df.columns = [col for col in flatten_multiindex_cols(cat_df.columns)]

        else:
            print("No categorical columns in df")
            cat_df = None
            if dtype == "cat":
                return None

        if dtype == "cat":
            df = cat_df.reset_index()
            if drop_collin_cols:
                df = drop_collinear_columns(df, threshold=0.9999)
            return df

    if (num_df is None) and (cat_df is None):
        return None

    elif num_df is None:
        df = cat_df.reset_index()

    elif cat_df is None:
        df = num_df.reset_index()

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
    df_num = df.select_dtypes(["number"])
    if df_num.shape[1] == 0:
        return df

    corr_matr = df_num.corr().abs()
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

        elif len(df[col].unique()) == 2:
            df[col] = df[col].astype(bool)

        elif df[col].dtype == float:
            df[col] = df[col].astype(np.float32)

        elif df[col].dtype == int:
            df[col] = df[col].astype(np.int32)

    memory = df.memory_usage().sum() / 10 ** 6
    print("Memory usage after changing types %0.2f MB" % memory)
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


# TODO: fix error when test have class which has not been seen in train.
class GeneralLabelEncoder(BaseEstimator, TransformerMixin):
    """
    sklearn LabelEncoder accepts only 1D array or pd Series.
    This class wraps around sklearn LabelEncoder and can handle a 2d array of categoricals
    or a dataframe of mixed types both numeric and categorical
    TransformerMixin gives us the method fit_transform()
    BaseEstimator gives us methods get_params() and set_params()
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

    # no need because of TransformerMixin
    #def fit_transform(self, x):
    #    self.fit(x)
    #    return self.transform(x)


def feature_extraction_application(csv_file):
    """
    :param csv_file: str, path of application csv file
    :return: dataframe
    """
    print("Extracting features from " + csv_file)
    df = pd.read_csv(csv_file)
    df = change_dtypes(df)

    days_emp_max = df["DAYS_EMPLOYED"].max()
    df["DAYS_EMPLOYED_POSITIVE"] = df["DAYS_EMPLOYED"] > 0
    df["DAYS_EMPLOYED"].replace({days_emp_max: np.nan}, inplace=True)

    df["AMT_INCOME_TOTAL_LOG"] = np.log(df["AMT_INCOME_TOTAL"])

    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TO_GOODS"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]

    return df


def feature_extraction_bureau(csv_file):
    """
    :param csv_file: str, path of bureau csv file
    :return: dataframe
    """
    print("Extracting features from " + csv_file)
    df = pd.read_csv(csv_file)
    df = change_dtypes(df)
    df = df.drop(["SK_ID_BUREAU"], axis=1)

    # some engineered features
    # whether DPD over 1, 3, and 6 months
    df["CREDIT_DAY_OVERDUE_OVER_0M"] = df["CREDIT_DAY_OVERDUE"] == 0
    df["CREDIT_DAY_OVERDUE_OVER_1M"] = df["CREDIT_DAY_OVERDUE"] > 30
    df["CREDIT_DAY_OVERDUE_OVER_3M"] = df["CREDIT_DAY_OVERDUE"] > 90
    df["CREDIT_DAY_OVERDUE_OVER_6M"] = df["CREDIT_DAY_OVERDUE"] > 180

    df["DAYS_CREDIT_ENDDATE_POS"] = df["DAYS_CREDIT_ENDDATE"] > 0

    # agg both numerical and categorical columns
    print("Aggregate both numerical and categorical columns")
    df_agg = aggregate(df, by=["SK_ID_CURR"], dtype="all",
                       num_stats=["count", "sum", "mean", np.var, "min", "max"],
                       cat_stats=["sum", "mean"])

    # some more more engineered features
    debt_to_credit = df_agg["AMT_CREDIT_SUM_DEBT_sum"] / df_agg["AMT_CREDIT_SUM_sum"]
    d2c_min = debt_to_credit[debt_to_credit != -np.inf].min()
    d2c_max = debt_to_credit[debt_to_credit != np.inf].max()
    debt_to_credit = debt_to_credit.replace({np.inf: d2c_max, -np.inf: d2c_min})
    df_agg["DEBT_TO_CREDIT"] = debt_to_credit

    overdue_to_debt = df_agg["AMT_CREDIT_SUM_OVERDUE_sum"] / df_agg["AMT_CREDIT_SUM_DEBT_sum"]
    o2d_min = overdue_to_debt[overdue_to_debt != -np.inf].min()
    o2d_max = overdue_to_debt[overdue_to_debt != np.inf].max()
    overdue_to_debt = overdue_to_debt.replace({np.inf: o2d_max, -np.inf: o2d_min})
    df_agg["OVERDUE_TO_DEBT"] = overdue_to_debt

    # time between loans
    print("Aggregate DAYS_CREDIT with mean_diff, var_diff and range_diff")
    df_agg_1 = aggregate(df[["SK_ID_CURR", "DAYS_CREDIT"]], by=["SK_ID_CURR"], dtype="num",
                         num_stats=[mean_diff, var_diff, range_diff])
    for col in df_agg_1.columns:
        tmp = df_agg_1[col]
        tmp_min = tmp[tmp != -np.inf].min()
        tmp_max = tmp[tmp != np.inf].max()
        tmp = tmp.replace({np.inf: tmp_max, -np.inf: tmp_min})
        df_agg_1[col] = tmp
    df_agg = df_agg.merge(df_agg_1, how="outer", on="SK_ID_CURR")

    # agg categorical columns with nunique and mode
    print("Aggregate categorical columns with nunique and mode")
    df_agg_2 = aggregate(df, by=["SK_ID_CURR"], dtype="cat", cat_stats=["nunique", mode], onehot_encode=False)
    df_agg = df_agg.merge(df_agg_2, how="outer", on="SK_ID_CURR")

    return df_agg


def feature_extraction_bureau_balance(bureau_balance_csv_file, bureau_csv_file):
    """
    :param bureau_balance_csv_file: str, path of bureau_balance csv file
    :param bureau_csv_file: str, path of bureau csv file
    :return: dataframe
    """
    print("Extracting features from " + bureau_balance_csv_file)
    df = pd.read_csv(bureau_balance_csv_file)
    df = change_dtypes(df)

    id_cols = pd.read_csv(bureau_csv_file)[["SK_ID_CURR", "SK_ID_BUREAU"]]
    id_cols = change_dtypes(id_cols)

    # agg both numerical columns by SK_ID_BUREAU
    print("Aggregate both numerical and categorical columns by SK_ID_BUREAU")
    df_agg = aggregate(df, by=["SK_ID_BUREAU"], dtype="all",
                       num_stats=["count", "sum", "mean", np.var, "min", "max"],
                       cat_stats=["sum", "mean"])

    print("Aggregate categorical columns by SK_ID_BUREAU with stats nunique and mode")
    df_agg_1 = aggregate(df, by=["SK_ID_BUREAU"], dtype="cat", cat_stats=["nunique", mode], onehot_encode=False)

    df_agg = df_agg.merge(df_agg_1, how="outer", on="SK_ID_BUREAU")

    df_agg = id_cols.merge(df_agg, how="left", on="SK_ID_BUREAU")
    df_agg = df_agg.drop(["SK_ID_BUREAU"], axis=1)

    count_cols = [col for col in df_agg.columns if col.split("_")[-1] in ["count", "sum", "nunique"]]
    print("Fillna these columns with zero:\n", count_cols)
    for col in count_cols:
        df_agg[col] = df_agg[col].fillna(0)

    print("Aggregate both numerical and categorical columns by SK_ID_CURR")
    df_agg = aggregate(df_agg, by=["SK_ID_CURR"], dtype="all",
                       num_stats=["count", "sum", "mean", np.var, "min", "max"],
                       cat_stats=["sum", "mean"])

    return df_agg


def feature_extraction_previous_application(csv_file):
    """
    :param csv_file: str, path to csv file
    :return: dataframe
    """
    print("Extracting features from " + csv_file)
    df = pd.read_csv(csv_file)
    df = change_dtypes(df)
    df = df.drop(["SK_ID_PREV"], axis=1)

    # agg both numerical and categorical columns
    print("Aggregate both numerical and categorical columns")
    df_agg = aggregate(df, by=["SK_ID_CURR"], dtype="all",
                       num_stats=["count", "sum", "mean", np.var, "min", "max"],
                       cat_stats=["sum", "mean"])

    # agg categorical columns with nunique and mode
    print("Aggregate categorical columns with nunique")
    df_agg_1 = aggregate(df, by=["SK_ID_CURR"], dtype="cat", cat_stats=["nunique"], onehot_encode=False)
    df_agg = df_agg.merge(df_agg_1, how="outer", on="SK_ID_CURR")

    return df_agg


def feature_extraction_POS_CASH_balance(POS_CASH_balance_csv_file, previous_application_csv_file):
    """
    :param POS_CASH_balance_csv_file: str, path of POS_CASH_balance csv file
    :param previous_application_csv_file: str, path of previous_application csv file
    :return: dataframe
    """
    print("Extracting features from " + POS_CASH_balance_csv_file)
    df = pd.read_csv(POS_CASH_balance_csv_file)
    df = change_dtypes(df)
    df = df.drop(["SK_ID_CURR"], axis=1)

    id_cols = pd.read_csv(previous_application_csv_file)[["SK_ID_CURR", "SK_ID_PREV"]]
    id_cols = change_dtypes(id_cols)

    # agg numerical and cat columns by SK_ID_PREV
    print("Aggregate numerical columns by SK_ID_PREV")
    df_agg = aggregate(df, by=["SK_ID_PREV"], dtype="all",
                       num_stats=["count", "sum", "mean", np.var, "min", "max"],
                       cat_stats=["sum", "mean"])

    print("Aggregate categorical columns by SK_ID_PREV with nunique")
    df_agg_1 = aggregate(df, by=["SK_ID_PREV"], dtype="cat", cat_stats=["nunique"], onehot_encode=False)

    df_agg = df_agg.merge(df_agg_1, how="outer", on="SK_ID_PREV")

    df_agg = id_cols.merge(df_agg, how="left", on="SK_ID_PREV")
    df_agg = df_agg.drop(["SK_ID_PREV"], axis=1)

    count_cols = [col for col in df_agg.columns if col.split("_")[-1] in ["count", "sum", "nunique"]]
    print("Fillna these columns with zero:\n", count_cols)
    for col in count_cols:
        df_agg[col] = df_agg[col].fillna(0)

    print("Aggregate both numerical and categorical columns by SK_ID_CURR")
    df_agg = aggregate(df_agg, by=["SK_ID_CURR"], dtype="num",
                       num_stats=["count", "sum", "mean", np.var, "min", "max"])

    return df_agg


def feature_extraction_credit_card_balance(credit_card_balance_csv_file, previous_application_csv_file):
    """
    :param credit_card_balance_csv_file: str, path of credit_card_balance csv file
    :param previous_application_csv_file: str, path of previous_application csv file
    :return: dataframe
    """
    return feature_extraction_POS_CASH_balance(credit_card_balance_csv_file, previous_application_csv_file)


def feature_extraction_installments_payments(installments_payments_csv_file, previous_application_csv_file):
    """
    :param installments_payments_csv_file: str, path of credit_card_balance csv file
    :param previous_application_csv_file: str, path of previous_application csv file
    :return: dataframe
    """
    print("Extracting features from " + installments_payments_csv_file)
    df = pd.read_csv(installments_payments_csv_file)
    df = change_dtypes(df)
    df = df.drop(["SK_ID_CURR"], axis=1)

    id_cols = pd.read_csv(previous_application_csv_file)[["SK_ID_CURR", "SK_ID_PREV"]]
    id_cols = change_dtypes(id_cols)

    # agg numerical columns by SK_ID_PREV
    print("Aggregate numerical columns by SK_ID_PREV")
    df_agg = aggregate(df, by=["SK_ID_PREV"], dtype="num",
                       num_stats=["count", "sum", "mean", np.var, "min", "max"])

    df_agg = id_cols.merge(df_agg, how="left", on="SK_ID_PREV")
    df_agg = df_agg.drop(["SK_ID_PREV"], axis=1)

    count_cols = [col for col in df_agg.columns if col.split("_")[-1] in ["count", "sum", "nunique"]]
    print("Fillna these columns with zero:\n", count_cols)
    for col in count_cols:
        df_agg[col] = df_agg[col].fillna(0)

    print("Aggregate both numerical and categorical columns by SK_ID_CURR")
    df_agg = aggregate(df_agg, by=["SK_ID_CURR"], dtype="num",
                       num_stats=["count", "sum", "mean", np.var, "min", "max"])

    return df_agg


def merge_tables(main_csv_file, on="SK_ID_CURR", other_csv_files=None, prefixes=None):
    print("Loading " + main_csv_file)
    df = pd.read_csv(main_csv_file)
    df = change_dtypes(df)

    if other_csv_files is None:
        return df

    assert type(other_csv_files) == list, "other_csv_files must be a list"
    assert type(prefixes) == list, "prefixes must be a list"
    assert len(other_csv_files) == len(prefixes), "other_csv_files and prefixes must have the same len"

    for other_csv, prefix in zip(other_csv_files, prefixes):
        print("Loading ", other_csv)
        other_df = pd.read_csv(other_csv)
        other_df = change_dtypes(other_df)
        new_cols = [col if col == on else prefix + col for col in other_df.columns]
        #print(new_cols)
        other_df.columns = new_cols

        df = df.merge(other_df, how="left", on=on)

        count_cols = [col for col in df.columns if col.split("_")[-1] in ["count", "sum", "nunique"]]
        #print("Fillna these columns with zero:\n", count_cols)
        for col in count_cols:
            df[col] = df[col].fillna(0)

    df = change_dtypes(df)
    return df

