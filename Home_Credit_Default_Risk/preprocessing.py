"""
Define functions for data preprocessing
"""

import pandas as pd
import numpy as np


def print_long_string(string, indent=0, max_words_per_line=10):
    """
    :param string: str
    :param indent: int
    :param max_words_per_line: int
    :return: None
    """
    words = [" "*indent]
    for i, word in enumerate(string.split()):
        words.append(word)
        if (i+1) % max_words_per_line == 0:
            words.append("\n" + " "*indent)
    print(" ".join(words))
    return None


def print_col_desc(df, table_name, col_desc_df):
    """
    To print out the information of columns of df. This information is taken from col_desc_df
    :param df: dataframe
    :param table_name: str
    :param col_desc_table:
    :return: None
    """
    for i, col in enumerate(df.columns):
        mask = (col_desc_df["Table"] == table_name) & (col_desc_df["Row"] == col)
        print("Column Number:", i)
        print("Column Name:", col)
        print("Description:")

        if len(col_desc_df.loc[mask, :]) == 0:
            print(" " * 10 + "No Description, Maybe the column name does not match.")
        else:
            print_long_string(col_desc_df.loc[mask, "Description"].iloc[0], indent=10)
            print("Special:", col_desc_df.loc[mask, "Special"].iloc[0])

        print("Type:", df[col].dtype)
        print("Number of NULL(s):", np.sum(df[col].isnull()))

        if (df[col].dtype == np.object) or (df[col].dtype == np.int):
            nunique = df[col].nunique(dropna=False)
            print("Number of Unique Values:", nunique)
            if nunique <= 20:
                print_long_string(", ".join([str(s) for s in df[col].unique()]),
                                  indent=28, max_words_per_line=5)
            else:
                print_long_string(", ".join([str(s) for s in df[col].unique()[:20]]) + " ...",
                                  indent=28, max_words_per_line=5)

        if np.issubdtype(df[col].dtype, np.number):
            print("Min:", df[col].min())
            print("Max:", df[col].max())

        print("-" * 50 + "\n")
    return None
