import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import PredefinedSplit
from collections import Counter
import sys
sys.path.append('.')
import utils

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

def my_scoring(self, X, y, sample_weight=None):
    scoring_prob = self.predict_proba(X)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y, scoring_prob)
    auc = metrics.auc(false_positive_rate, true_positive_rate)
    ks = get_ks_score(scoring_prob, y)
    
    score_need = 2*auc*ks/(auc + ks)
    return score_need

def data_init(X_train, X_val, X_test, y_train, y_val, y_test, k):
    # 归一化
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_train = min_max_scaler.transform(X_train)
    X_val = min_max_scaler.transform(X_val)
    X_test = min_max_scaler.transform(X_test)
    # 集成学习分割数据
    X_train_1 = X_train[y_train == 1]
    y_train_1 = y_train[y_train == 1]
    X_train_0 = X_train[y_train == 0]
    y_train_0 = y_train[y_train == 0]
    step_size = X_train_0.shape[0] // k
    
    X_train_need = []
    y_train_need = []
    for i in range(k):
        tmp_x = X_train_0[i*step_size:min(X_train_0.shape[0]-1, i*step_size+step_size)]
        tmp_y = y_train_0[i*step_size:min(X_train_0.shape[0]-1, i*step_size+step_size)]
        X_train_need.append(np.concatenate((X_train_1, tmp_x), axis=0))
        y_train_need.append(np.concatenate((y_train_1, tmp_y), axis=0))

    C_params = [i/10 for i in range(1, 10)] + [i for i in range(1, 10, 1)] + [i for i in range(10, 100, 10)]
    cw_params = [i for i in range(1, 10, 1)] + [i for i in range(10, 100, 10)]
    params = {
        'C': C_params,
        'class_weight': [{1:w} for w in cw_params],
    }

    train_pred_prob_record = []
    val_pred_prob_record = []
    test_pred_prob_record = []
    for i in range(k):
        print(i)
        train_val_features = np.concatenate((X_train_need[i], X_val), axis = 0)
        train_val_labels = np.concatenate((y_train_need[i], y_val), axis = 0)
        test_fold = np.zeros(train_val_features.shape[0])
        test_fold[:X_train_need[i].shape[0]] = -1
        ps = PredefinedSplit(test_fold = test_fold)
        
        model = GridSearchCV(estimator=LogisticRegression(), param_grid=params, 
                            scoring=my_scoring, n_jobs=-1, cv=ps, verbose=0)
        model.fit(train_val_features, train_val_labels)
        print(model.best_params_ )
        print(model.best_score_ )
        train_pr = model.predict_proba(X_train)[:, 1]
        val_pr = model.predict_proba(X_val)[:, 1]
        test_pr = model.predict_proba(X_test)[:, 1]
        
        utils.model_key_performance(train_pr, y_train)
        utils.model_key_performance(val_pr, y_val)
        utils.model_key_performance(test_pr, y_test)
        
        train_pred_prob_record.append(train_pr)
        val_pred_prob_record.append(val_pr)
        test_pred_prob_record.append(test_pr)
        
        train_ensum = np.array(train_pred_prob_record).T
        val_ensum = np.array(val_pred_prob_record).T
        test_ensum = np.array(test_pred_prob_record).T
    return train_ensum, val_ensum, test_ensum

def bagging(train_ensum, val_ensum, test_ensum, y_train, y_val, y_test):

    #bagging
    bagging_train_prob = np.mean(train_ensum, axis = 1)
    bagging_val_prob = np.mean(val_ensum, axis = 1)
    bagging_test_prob = np.mean(test_ensum, axis = 1)
    #result
    a, b = utils.model_key_performance(bagging_train_prob, y_train)
    c, d = utils.model_key_performance(bagging_val_prob, y_val)
    e, f = utils.model_key_performance(bagging_test_prob, y_test)
    return a, b, c, d, e, f

def stacking(train_ensum, val_ensum, test_ensum, y_train, y_val, y_test):
    C_params = [i/10 for i in range(1, 10)] + [i for i in range(1, 10, 1)] + [i for i in range(10, 100, 10)]
    cw_params = [i for i in range(1, 10, 1)] + [i for i in range(10, 100, 10)]
    params = {
        'C': C_params,
        'class_weight': [{1:w} for w in cw_params],
    }
    train_val_features = np.concatenate((train_ensum, val_ensum), axis = 0)
    train_val_labels = np.concatenate((y_train, y_val), axis = 0)
    test_fold = np.zeros(train_val_features.shape[0])
    test_fold[:train_ensum.shape[0]] = -1
    ps = PredefinedSplit(test_fold = test_fold)

    lr_stack = GridSearchCV(estimator=LogisticRegression(), param_grid=params, 
                         scoring=my_scoring, n_jobs=-1, cv=ps, verbose=0)
    lr_stack.fit(train_val_features, train_val_labels)

    lr_stack_train_pred_prob = lr_stack.predict_proba(train_ensum)[:, 1]
    lr_stack_val__pred_prob = lr_stack.predict_proba(val_ensum)[:, 1]
    lr_stack_test_pred_prob = lr_stack.predict_proba(test_ensum)[:, 1]
    a, b = utils.model_key_performance(lr_stack_train_pred_prob, y_train)
    c, d = utils.model_key_performance(lr_stack_val__pred_prob, y_val)
    e, f = utils.model_key_performance(lr_stack_test_pred_prob, y_test)
    return a, b, c, d, e, f