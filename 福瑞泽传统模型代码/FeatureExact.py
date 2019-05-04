# coding: utf-8

# basic packages
import operator
import pandas as pd
import numpy as np
from collections import Counter

# classifier packages
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier

# preprocessing datasets packages
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler, OneHotEncoder
from imblearn.under_sampling import ClusterCentroids, NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# self_defination tools
import sys
sys.path.append('.')
import utils 

def feature_select(data_file_name='./data.csv', selected_dim=30):
    # read the sourse dataset
    data = pd.read_csv(data_file_name)

    # spliting the the dataset into trainset, and testset
    y = data.pop('label')
    X = data.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state = 3)# 3 6

    # minmax preprocessing 
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_train = min_max_scaler.transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    # select fatures by the model of xgboost
    dtrain = xgb.DMatrix(X_train, label=y_train) # xgboost data style
    dtest = xgb.DMatrix(X_test, label=y_test) 

    # super-parameters configure for xgboost model
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 3,
        'lambda': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'eta': 0.1,
        'silent': 0,
        'scale_pos_weight': 3
    }
    watchlist = [(dtrain, 'train'), (dtest, 'test')]  # print the evaluating metric of dtrain and dtest during the training 
    
    # training the xgboost model
    xg_model = xgb.train(params, dtrain, num_boost_round=10, evals=watchlist)
    # select the important features from raw features
    features = data.columns
    utils.create_feature_map(features)
    importance = xg_model.get_fscore(fmap='xgb.fmap')
    # ranking the fscores
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    # construct the dframe for fscores of features
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df.to_csv('feat_importance.csv', index=False) 

    # select the features according the fscores
    selected_data = data[df['feature'][:selected_dim]]
    selected_data.insert(selected_dim, 'label', y)
    #selected_data.to_csv('feature_seleted.csv',  encoding='utf-8', index=False)

    return selected_data

def split_dataset(selected_data):    
    # splitting the seleted_data into trainset, valset, testset, and normalization
    train_size = 0.74
    val_size = 0.13
    test_size = 0.13

    X_train, X_val, X_test, y_train, y_val, y_test = utils.train_val_test_split(selected_data, train_size, val_size, test_size, 
                                                                            random_state=0, time_factors=False)
    # minmax preprocessing
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_train = min_max_scaler.transform(X_train)
    X_val = min_max_scaler.transform(X_val)
    X_test = min_max_scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def test_lr_model(X_train, X_val, X_test, y_train, y_val, y_test):
    # LR model
    lr = LogisticRegression(penalty='l2', C=1, class_weight={1:50})
    lr.fit(X_train, y_train)

    # validate the  LR model
    y_train_pre = lr.predict_proba(X_train)[:,1]
    y_val_pre = lr.predict_proba(X_val)[:,1]
    y_test_pre = lr.predict_proba(X_test)[:,1]
    print('lr-model validation result: ')
    utils.model_key_performance_print(y_train_pre, y_train)
    utils.model_key_performance_print(y_val_pre, y_val)
    utils.model_key_performance_print(y_test_pre, y_test)

def test_lr_model_shizhe(X_train, X_val, X_test, y_train, y_val, y_test):
    # LR model
    lr = LogisticRegression(penalty='l2', C=1, class_weight={1:50})
    lr.fit(X_train, y_train)

    # validate the  LR model
    y_train_pre = lr.predict_proba(X_train)[:,1]
    y_val_pre = lr.predict_proba(X_val)[:,1]
    y_test_pre = lr.predict_proba(X_test)[:,1]
    print('lr-model validation result: ')
    utils.model_key_performance(y_train_pre, y_train)
    utils.model_key_performance(y_val_pre, y_val)
    utils.model_key_performance(y_test_pre, y_test)
    
    
def test_xgboost_model(X_train, X_val, X_test, y_train, y_val, y_test, max_depth=3):
    # xg-boost model
    # super-parameters configure for xgboost model
    dtrain = xgb.DMatrix(X_train, label=y_train) # xgboost data style
    dval = xgb.DMatrix(X_val, label=y_val) 
    dtest = xgb.DMatrix(X_test, label=y_test) 
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': max_depth,
        'lambda': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'eta': 0.1,
        'silent': 0,
        'scale_pos_weight': 3
    }

    watchlist = [(dtrain, 'train'), (dval, 'val'), (dtest, 'test')]  # print the evaluating metric of dtrain and dtest during the training 

    # training the xgboost model
    xg_model = xgb.train(params, dtrain, num_boost_round=80, evals=watchlist)

    # validate the xg_model
    y_train_pre = xg_model.predict(dtrain)
    y_val_pre = xg_model.predict(dval)
    y_test_pre = xg_model.predict(dtest)
    print('xg-boost validation result: ')
    utils.model_key_performance(y_train_pre, y_train)
    utils.model_key_performance(y_val_pre, y_val)
    utils.model_key_performance(y_test_pre, y_test)


def test_rf_model(X_train, X_val, X_test, y_train, y_val, y_test, n_estimators=50, max_depth=3, return_model=False):
    # random forest model
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth= max_depth, class_weight='balanced', bootstrap=True, random_state=42)
    rf.fit(X_train, y_train)

    y_train_pre = rf.predict_proba(X_train)[:,1]
    y_val_pre = rf.predict_proba(X_val)[:,1]
    y_test_pre = rf.predict_proba(X_test)[:,1]

    utils.model_key_performance_print(y_train_pre, y_train)
    utils.model_key_performance_print(y_val_pre, y_val)
    utils.model_key_performance_print(y_test_pre, y_test)
        
def test_etc_model(X_train, X_val, X_test, y_train, y_val, y_test, n_estimators=50, max_depth=3, return_model=False):
    #Extra Trees Classifier模型
    etc = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', bootstrap=False, random_state=42)
    etc.fit(X_train, y_train)

    y_train_pre = etc.predict_proba(X_train)[:,1]
    y_val_pre = etc.predict_proba(X_val)[:,1]
    y_test_pre = etc.predict_proba(X_test)[:,1]

    utils.model_key_performance(y_train_pre, y_train)
    utils.model_key_performance(y_val_pre, y_val)
    utils.model_key_performance(y_test_pre, y_test)

def code_by_tree_model(tr_model, X):
    return tr_model.apply(X)


def generate_one_hot(X_train, X_val, X_test):
    # define a one-hot code model
    enc = OneHotEncoder()
    enc.fit(X_train)
    
    # generate one-hot code
    return enc.transform(X_train).toarray(), enc.transform(X_val).toarray(), enc.transform(X_test).toarray()
    
def generate_cross_feature(X_train, X_val, X_test):
    poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
    poly.fit(X_train)
    X_train_poly = poly.transform(X_train)
    X_val_poly = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)
    
    return X_train_poly, X_val_poly, X_test_poly

def concat_features(multi_features):
    # concat all features
    concat_fea = []
    for fea in multi_features:
        concat_fea.append(np.concatenate(fea, axis=1))
    # pass
    return concat_fea

def test(data_file = 'feature_seleted.csv'):
    data = pd.read_csv(data_file)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(data)
    rf = RandomForestClassifier(n_estimators=50, max_depth= 3, class_weight='balanced', bootstrap=True, random_state=42)
    rf.fit(X_train, y_train)
    X_train_code, X_val_code, X_test_code = [code_by_tree_model(rf, x) for x in [X_train, X_val, X_test]]
    X_train_onehot_code, X_val_onehot_code, X_test_onehot_code = generate_one_hot(X_train_code, X_val_code, X_test_code)
    
    test_rf_model(X_train_onehot_code, X_val_onehot_code, X_test_onehot_code, y_train, y_val, y_test, n_estimators=50, max_depth=3, return_model=False)

def test_cross_feature(data_file = './feature_selected.csv'):
    data = pd.read_csv(data_file)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(data)
    X_train_poly, X_val_poly, X_test_poly = generate_cross_feature(X_train, X_val, X_test)
    
    test_xgboost_model(X_train_poly, X_val_poly, X_test_poly, y_train, y_val, y_test, max_depth=3)

'''
if __name__ == '__main__':
    # test()
    test_cross_feature()
'''