# coding: utf-8
# Author: Youchen Du
import logging

import numpy as np
import pandas as pd


# Constants to control the functionality
DISCARD_THRESHOLD = 0.7
FILL_THRESHOLD = 0.3
CONST1 = 0
CONST2 = 0
INVALID_VALUES = (-8888, -9999)


def normalize_func(x, fill):
    """Normalize x if x is valid, else replace with provided value."""
    if isinstance(x, str):
        return 1 if x == '正常' else 0
    if x in INVALID_VALUES:
        return fill
    return x


def one_hot_func(x):
    """Used to one-hot value."""
    if isinstance(x, str):
        return 1 if x == '正常' else 0
    return 1 if x not in INVALID_VALUES else 0


def strong_clean_func(series, one_hot=True, fill=CONST2, normalize_func=normalize_func,
                      one_hot_func=one_hot_func):
    """Clean and normalize series and return one-hot series or raw series."""
    if one_hot:
        series = series.apply(one_hot_func)
    else:
        series = series.apply(lambda x: normalize_func(x, fill))
    # TODO(youchen): if the column has object as its dtype, how to handle it?
    if series.dtype == object:
        series = series.astype(np.int64)
    return series


def strong_clean(df, strong_clean_columns):
    df = df[strong_clean_columns]
    one_hot_df = df.apply(lambda series: strong_clean_func(series, one_hot=True))
    one_hot_df = one_hot_df.rename(mapper=lambda x: '{}(One-Hot)'.format(x), axis=1)

    numerical_df = df.apply(lambda series: strong_clean_func(series, one_hot=False))
    numerical_df = numerical_df.rename(mapper=lambda x: '{}(Constant)'.format(x), axis=1)

    return pd.concat([one_hot_df, numerical_df], axis=1)
    

def weak_clean_func(series, fill=CONST1, normalize_func=normalize_func):
    """Clean and normalize series."""
    series = series.apply(lambda x: normalize_func(x, fill))
    return series


def weak_clean(df, weak_clean_columns):
    weak_clean_df = df[weak_clean_columns]
    return weak_clean_df.apply(weak_clean_func)


def drop_non_feature_columns(df):
    drop_columns = ['APPLYDATE', 'applydate', 'APPLYCD',
                    'applyid', 'label', 'app_worst_loan_five_rate_is_8']
    return df.drop(drop_columns, axis=1, errors='ignore')


def clean_data(df_or_fpath, clean_columns=None):
    """Clean provided DataFrame or DataFrame read from provided fpath."""
    if isinstance(df_or_fpath, str):
        df = pd.read_csv(df_or_fpath, encoding='gbk')
    else:
        df = df_or_fpath
        
    df = drop_non_feature_columns(df)
        
    # Calculate invalid rate of columns
    invalid_rate = df.isin(INVALID_VALUES).apply(pd.value_counts)
    invalid_rate = invalid_rate.fillna(0)
    invalid_rate = invalid_rate.loc[True] / invalid_rate.sum()

    # Determine columns should be cleaned
    if clean_columns is not None:
        discard_columns, strong_clean_columns, weak_clean_columns = clean_columns
    else:
        discard_columns = invalid_rate.index[invalid_rate > DISCARD_THRESHOLD]
        logging.debug('Discard columns: {}'.format(discard_columns))

        strong_clean_columns = invalid_rate.index[invalid_rate.between(FILL_THRESHOLD+1e-6, DISCARD_THRESHOLD)]
        logging.debug('Strong clean columns: {}'.format(strong_clean_columns))

        weak_clean_columns = invalid_rate.index[invalid_rate <= FILL_THRESHOLD]
        logging.debug('Weak clean columns: {}'.format(weak_clean_columns))

        logging.debug('Total columns: {}, Discard columns: {}, Strong clean columns: {}, Weak clean columns: {}'.format(
            len(invalid_rate.index), len(discard_columns), len(strong_clean_columns), len(weak_clean_columns)))

    # Case 1:
    # Invalid rate of specific column is higher than DISCARD_THRESHOLD
    # Action:
    # Delete this column
    clean_df = df.drop(discard_columns, axis=1, errors='ignore')
    logging.debug('DataFrame shape for case 1: {}'.format(clean_df.shape))

    # Case 2:
    # Invalid rate of specific column is less or equal than DISCARD_THRESHOLD and larger than FILL_THRESHOLD
    # Action:
    # Split this column into two columns:
    #   1. one as one-hot column, 1 for valid value, 0 for invalid value
    #   2. the other copies data from the original column, but use normalization func to normalize valid value,
    #      and replace invalid value with CONST2
    strong_clean_df = strong_clean(clean_df, strong_clean_columns)
    logging.debug('DataFrame shape for case 2: {}'.format(strong_clean_df.shape))

    # Case 3:
    # Invalid rate of specific column is less or equal than FILL_THRESHOLD
    # Action:
    # Normalize valid values, replace invalid values with CONST1
    weak_clean_df = weak_clean(clean_df, weak_clean_columns)
    logging.debug('DataFrame shape for case 3: {}'.format(weak_clean_df.shape))

    # Concatenate cleaned data frame with apply id and apply date series
    final_df = pd.concat([strong_clean_df, weak_clean_df], axis=1)
    final_df = final_df.reindex(sorted(final_df.columns), axis=1)
    logging.debug('DataFrame shape after cleaned: {}'.format(final_df.shape))
    
    return final_df, (discard_columns, strong_clean_columns, weak_clean_columns)