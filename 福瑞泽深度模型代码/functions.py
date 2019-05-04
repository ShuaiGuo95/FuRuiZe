
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'


def calc_KS_AR(df, col, label):
    """
    :param df: the dataframe that contains probability and bad indicator
    :param score:
    :return:
    """
    total = pd.DataFrame({'total': df.groupby(col)[label].count()})
    bad = pd.DataFrame({'bad': df.groupby(col)[label].sum()})
    regroup = total.merge(bad, how='left', left_index=True, right_index=True)
    regroup['good'] = regroup['total'] - regroup['bad']
    regroup.reset_index(inplace=True)
    regroup['goodCumPer'] = regroup['good'].cumsum() / regroup['good'].sum()
    regroup['badCumPer'] = regroup['bad'].cumsum() / regroup['bad'].sum()
    # regroup['totalPer'] = regroup['total'] / regroup['total'].sum()

    KS = regroup.apply(lambda x: x.goodCumPer - x.badCumPer, axis=1)
    return KS.max(), regroup


def plot_ks():
    file_name = 'train/train_result.csv'
    df = pd.read_csv(file_name)
