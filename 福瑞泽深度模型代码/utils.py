# coding=utf-8

import pandas as pd
import numpy as np
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.preprocessing import MinMaxScaler
from mxnet import ndarray as nd
        

def evaluate(label, prob, pos_label = 0):
    auc = calculateAUC(label, prob, pos_label = pos_label)
    ks_their, ks_mine = calculateKS(label, prob)
    return auc, ks_their


def calculateAUC(label, prob, pos_label):
    fpr, tpr, thresholds = metrics.roc_curve(label, prob, pos_label = pos_label)  # fpr, tpr 是2个array
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


def calculateKS(label, prob):
    df = pd.DataFrame(data = {'label': label, 'prob': prob})
    df['prob'] = df['prob'].map(lambda x: round(x, 3))
    total = pd.DataFrame({'total': df.groupby('prob')['label'].count()})  
    bad = pd.DataFrame({'bad': df.groupby('prob')['label'].sum()})
    all_data = total.merge(bad, how = 'left', left_index = True, right_index = True)
    all_data['good'] = all_data['total'] - all_data['bad']
    all_data.reset_index(inplace = True)
    all_data.sort_index(ascending = False, inplace = True)
    all_data['goodCumPer'] = all_data['good'].cumsum() / all_data['good'].sum()
    all_data['badCumPer'] = all_data['bad'].cumsum() / all_data['bad'].sum()
    KS_m = all_data.apply(lambda x: x.goodCumPer - x.badCumPer, axis = 1)
    KS_t = all_data.apply(lambda x: x.badCumPer - x.goodCumPer, axis = 1)
    return max(KS_t), max(KS_m)


def transform_func(u, sigma, x):
    y = (np.exp(-(x - u)**2/(2*sigma**2)))##均值为u，标准差为sigma的正态分布
    return y


def generate_normal_distribution(N):
    u = 0.5
    sigma = 0.125
    res = []
    count = 0
    while count < N:
            x = np.random.uniform()
            y = transform_func(u, sigma, x)
            x_need = np.random.uniform()
            if x_need <= y:
                res.append(x)
                count+=1
    res = sorted(res)
    return res


def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)