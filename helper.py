import pandas as pd
import numpy as np


def column_to_numeric(df, start, stop):
    df.iloc[:, start:stop] = df.iloc[:, start:stop].apply(
        pd.to_numeric, errors='coerce')
    return df


def column_to_float(df, start, stop):
    df.iloc[:, start:stop] = df.iloc[:, start:stop].apply(
        pd.to_numeric, downcast='float', errors='coerce')
    return df


def impute_nan_add_vairable(df, missing_columns):
    for column in missing_columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    return df


def rectify_outliers_using_log_transformation(df, missing_columns):
    for columns in missing_columns:
        df[columns] = np.log1p(df[columns])
    return df
