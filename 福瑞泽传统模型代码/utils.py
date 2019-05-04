import pandas as pd
import numpy as np
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

def train_val_test_split(data, train_rate, validate_rate, test_rate, random_state=None, time_factors=False):
    '''
    训练集,验证集,测试集分割
    :param data: 数据(DataFrame格式)
    :train_rate: 训练集所占有的比例(0-1)
    :validate_rate: 验证集所占有的比例(0-1)
    :test_rate: 测试集所占有的比例(0-1)
    :param time_factors: 是否考虑时间因素(False)
    :return: train_data, val_data, test_data,数据格式都是numpy
    '''

    assert  1 == (train_rate + validate_rate + test_rate), \
        '(train_rate + validate_rate + test_rate) must equal 1'
    if time_factors == False:
        # data.pop('applymonth')
        data = data.copy()
        data_y = data.pop('label').as_matrix()
        data_X = data.as_matrix()
        # #PCA
        # pca = PCA()
        # data_X = pca.fit_transform(data_X)
        
        train_X, val_test_X, train_y, val_test_y = train_test_split(data_X, data_y,
                                                test_size=(validate_rate + test_rate),
                                                stratify=data_y,
                                                random_state=random_state)
        val_X, test_X, val_y, test_y = train_test_split(val_test_X, val_test_y,
                                        test_size=(test_rate/(test_rate + validate_rate)),
                                        stratify=val_test_y,
                                        random_state=random_state)
        return train_X, val_X, test_X, train_y, val_y, test_y
    else:
        data = data.sort_values(by='applymonth', axis=0)
        sample_num = data.shape[0]
        train_sample_index = range(round(sample_num * train_rate))
        val_sample_index = range(len(train_sample_index), round(sample_num * (train_rate + validate_rate)))
        test_sample_index = range((len(train_sample_index) + len(val_sample_index)), sample_num)
        # data.pop('applymonth')
        data_y = data.pop('label').as_matrix()
        data_X = data.as_matrix()
        train_X = data_X[train_sample_index, :]
        train_y = data_y[train_sample_index]
        val_X = data_X[val_sample_index, :]
        val_y = data_y[val_sample_index]
        test_X = data_X[test_sample_index, :]
        test_y = data_y[test_sample_index]
        return train_X, val_X, test_X, train_y, val_y, test_y
    
def data_under_sampling(X, y, under_sample_type, ratio, random_state=None):
    '''
    数据欠采样
    :param X: 数据
    :param y: 标签
    :param under_sample_type: 欠采样类型(str)
    :param ratio: ratio = (少类样本)/(多类样本)
    :param random_state: 随机采样的种子
    :return: 采样后的数据(nparray)
    '''
    under_sampler = None
    if under_sample_type == 'Random':
        from imblearn.under_sampling import RandomUnderSampler
        under_sampler = RandomUnderSampler(ratio=ratio, random_state=random_state)
    elif under_sample_type == 'NearMiss':
        from imblearn.under_sampling import NearMiss
        under_sampler = NearMiss(ratio=ratio, random_state=random_state, version=1)
    elif under_sample_type == 'EditedNearestNeighbours':
        from imblearn.under_sampling import EditedNearestNeighbours
        under_sampler = EditedNearestNeighbours(random_state=random_state)
    X_resampled, y_resampled = under_sampler.fit_sample(X, y)
    return X_resampled, y_resampled  

def get_predict_labels(predict_prob):
    predict_labels = []
    for pro in predict_prob:
        if pro > 0.5:
            predict_labels.append(1)
        else:
            predict_labels.append(0)
    return np.array(predict_labels)

def get_ks_score(prob, label):
    '''
    计算ks得分
    :param pro: 属于坏人的概率
    :param label: 真实标签
    :return: ks得分
    '''
    df = pd.DataFrame(data = {'label': label, 'prob': prob})
    df['prob'] = df['prob'].map(lambda x: round(x, 3))
    total = pd.DataFrame({'total': df.groupby('prob')['label'].count()})
    bad = pd.DataFrame({'bad': df.groupby('prob')['label'].sum()})
    all_data = total.merge(bad, how = 'left', left_index = True, right_index = True)
    all_data['good'] = all_data['total'] - all_data['bad']
    all_data.reset_index(inplace = True)
    all_data['goodCumPer'] = all_data['good'].cumsum() / all_data['good'].sum()
    all_data['badCumPer'] = all_data['bad'].cumsum() / all_data['bad'].sum()
    KS_m = all_data.apply(lambda x: x.goodCumPer - x.badCumPer, axis = 1)
    return max(KS_m)

def get_auc_score(test_y_predict_prob, test_y):
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(test_y, test_y_predict_prob)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    return roc_auc

def model_key_performance(test_y_predict_prob, test_y):
    '''
    获取模型的precision, recall, F1_score, confusion_matrix, ks_score
    :param test_y_predict_prob: 测试集预测概率
    :param test_y: 测试集标签
    '''
    
    test_y_predict_labels = get_predict_labels(test_y_predict_prob)
    
    precision_score = metrics.precision_score(test_y, test_y_predict_labels)
    recall_score = metrics.recall_score(test_y, test_y_predict_labels)
    f1_score = metrics.f1_score(test_y, test_y_predict_labels)
    confusion_matrix = metrics.confusion_matrix(test_y, test_y_predict_labels)

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(test_y, test_y_predict_prob)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    ks_score = get_ks_score(test_y_predict_prob, test_y)
    
    #print('\npercision: %.4f recall: %.4f f1: %.4f auc: %.4f ks: %.4f confusion_matrix: ' % (precision_score, recall_score, 
    #                                                           f1_score, roc_auc, ks_score))
    #print('{0:<7}{1:<7}'.format(confusion_matrix[0, 0], confusion_matrix[0, 1]))
    #print('{0:<7}{1:<7}'.format(confusion_matrix[1, 0], confusion_matrix[1, 1]))
    return roc_auc, ks_score

def model_key_performance_print(test_y_predict_prob, test_y):
    '''
    获取模型的precision, recall, F1_score, confusion_matrix, ks_score
    :param test_y_predict_prob: 测试集预测概率
    :param test_y: 测试集标签
    '''
    
    test_y_predict_labels = get_predict_labels(test_y_predict_prob)
    
    precision_score = metrics.precision_score(test_y, test_y_predict_labels)
    recall_score = metrics.recall_score(test_y, test_y_predict_labels)
    f1_score = metrics.f1_score(test_y, test_y_predict_labels)
    confusion_matrix = metrics.confusion_matrix(test_y, test_y_predict_labels)

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(test_y, test_y_predict_prob)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    ks_score = get_ks_score(test_y_predict_prob, test_y)
    
    print('\npercision: %.4f recall: %.4f f1: %.4f auc: %.4f ks: %.4f confusion_matrix: ' % (precision_score, recall_score, 
                                                               f1_score, roc_auc, ks_score))
    print('{0:<7}{1:<7}'.format(confusion_matrix[0, 0], confusion_matrix[0, 1]))
    print('{0:<7}{1:<7}'.format(confusion_matrix[1, 0], confusion_matrix[1, 1]))
    #return roc_auc, ks_score

def kfold(X, y, num_fold):
    kf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state = 1)
    for train_val_index, test_index in kf.split(X, y):
        X_train_val = X[train_val_index]
        y_train_val = y[train_val_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, 
                                                         test_size=(1.0/(num_fold-1)), 
                                                         stratify=y_train_val, random_state = 2)
        X_test = X[test_index]
        y_test = y[test_index]
        yield X_train, X_val, X_test, y_train, y_val, y_test

def data_init(data): # level6与缺失值填充
    features = pd.read_csv('./cmh_level/feat_importance.csv')
    imp_level = features['fscore'].drop_duplicates()
    label_col = data['label']
    # level6
    level = 6
    imp, unimp = [], []
    for index, row in features.iterrows():
        if row[1] > level:
            imp.append(row[0])
        else:
            unimp.append(row[0])
    imp_counts = len(imp)
    unimp_counts = len(unimp)
    missValue_1_imp = []
    missValue_2_imp = []
    missValue_all_imp = []
    missValue_1_unimp = []
    missValue_2_unimp = []
    missValue_all_unimp = []
    missValue_1_all = []
    missValue_2_all = []
    missValue_all_all = []
    imp_data = data.loc[:, imp]
    #将数据分隔成重要特征数据和不重要特征数据
    unimp_data = data.loc[:, unimp]
    total_count = 113
    for row in range(0, len(data)):
        imp_tmp_record = imp_data.loc[row,:]
        unimp_tmp_record = unimp_data.loc[row,:]
        count_imp_1 = len(imp_tmp_record[imp_tmp_record == -1])
        #缺失值-1的重要特征占比82
        count_imp_2 = len(imp_tmp_record[imp_tmp_record == -2])
        #缺失值-2的重要特征占比
        missValue_1_imp.append(count_imp_1 / imp_counts)
        missValue_2_imp.append(count_imp_2 / imp_counts)
        missValue_all_imp.append((count_imp_1 + count_imp_2) / imp_counts)
        count_un_1 = len(unimp_tmp_record[unimp_tmp_record == -1])
        #缺失值-1的不重要特征占比31
        count_un_2 = len(unimp_tmp_record[unimp_tmp_record == -2])
        #缺失值-2的不重要特征占比
        missValue_1_unimp.append(count_un_1 / unimp_counts)
        missValue_2_unimp.append(count_un_2 / unimp_counts)
        missValue_all_unimp.append((count_un_1 + count_un_2) / unimp_counts)
        missValue_1_all.append((count_imp_1 + count_un_1) /  (imp_counts + unimp_counts))
        missValue_2_all.append((count_imp_2 + count_un_2) / (imp_counts + unimp_counts))
        missValue_all_all.append((count_imp_1 + count_un_1 + count_imp_2 + count_un_2) / (imp_counts + unimp_counts))

    data['missValue_1_imp'] = missValue_1_imp
    data['missValue_2_imp'] = missValue_2_imp
    data['missValue_all_imp'] = missValue_all_imp
    data['missValue_1_unimp'] = missValue_1_unimp
    data['missValue_2_unimp'] = missValue_2_unimp
    data['missValue_all_unimp'] = missValue_all_unimp
    data['missValue_1_all'] = missValue_1_all
    data['missValue_2_all'] = missValue_2_all
    data['missValue_all_all'] = missValue_all_all
    data['label'] = label_col
    #......
    data.insert(0, 'untitled', np.arange(data.shape[0]))
    # median 填充
    raw_data = data.replace({-1 : np.nan, -2 : np.nan})
    median =  raw_data.median(axis = 1)
    miss_value_map = {}
    for col in raw_data.columns.values:
        miss_value_map[col] = median[0]
    res = raw_data.fillna(miss_value_map)
    return res

def predict_prob_distribution_adjust(y_predict_prob, factor=math.log(60), offset=600): 
    '''
    调整预测概率的分布
    :param y_predict_prob 预测概率
    :param factor 线性变换的系数，通常是一个对数值
    :param offset 调整常数
    return 调整后的概率分布
    '''
    y_predict_prob_need = [(factor * math.log(pr) + offset) for pr in y_predict_prob]
    y_predict_prob_need = np.array(y_predict_prob_need)
    
    pr_need_max = np.max(y_predict_prob_need)
    pr_need_min = np.min(y_predict_prob_need)
    y_predict_prob_need = (y_predict_prob_need - pr_need_min) / (pr_need_max - pr_need_min)
    
    return y_predict_prob_need

def create_feature_map(features):  
    outfile = open('xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1  
    outfile.close()  
