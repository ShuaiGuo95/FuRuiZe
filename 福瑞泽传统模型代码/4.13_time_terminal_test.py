
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append('.')
import utils


# In[2]:


X_train = pd.read_csv('./time_split/train_data_6_10_feature_selected.csv')
X_val = pd.read_csv('./time_split/val_data_11_feature_selected.csv')
X_test = pd.read_csv('./time_split/test_data_feature_selected_12.csv')
X_train = utils.data_init(X_train)
X_val = utils.data_init(X_val)
X_test = utils.data_init(X_test)

y_train = X_train.pop('label').as_matrix()
y_val = X_val.pop('label').as_matrix()
y_test = X_test.pop('label').as_matrix()
X_train = X_train.as_matrix()
X_val = X_val.as_matrix()
X_test = X_test.as_matrix()

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X_train)
X_train = min_max_scaler.transform(X_train)
X_val = min_max_scaler.transform(X_val)
X_test = min_max_scaler.transform(X_test)


# In[3]:


train_val_features = np.concatenate((X_train, X_val), axis = 0)
train_val_labels = np.concatenate((y_train, y_val), axis = 0)


# In[4]:


dtrain = xgb.DMatrix(train_val_features, label=train_val_labels) # xgboost data style
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    'booster': 'gbtree',
    'objective': 'binary:logitraw',
    'eval_metric': 'auc',
    'max_depth': 3,
    'max_delta_step': 3,
    'lambda': 1,
    'subsample': 0.5,
    'colsample_bytree': 0.8,
    'min_child_weight': 2,
    'eta': 0.1,
    'silent': True,
    'scale_pos_weight': 3,
    'gamma': 0.1
}
watchlist = [(dtrain, 'train'), (dval, 'val'), (dtest, 'test')]
xg_model = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)


# In[7]:


y_train_pred_prob = xg_model.predict(dtrain)
y_val_pred_prob = xg_model.predict(dval)
y_test_pred_prob = xg_model.predict(dtest)
utils.model_key_performance_print(y_train_pred_prob, train_val_labels)
utils.model_key_performance_print(y_val_pred_prob, y_val)
utils.model_key_performance_print(y_test_pred_prob, y_test)


# In[8]:


y_test_prob_need = y_test_pred_prob - np.amin(y_test_pred_prob) + 0.01
y_test_prob_need = y_test_prob_need/(np.max(y_test_prob_need)+1)
y_test_prob_need = np.log(60)*np.log(y_test_prob_need/(1-y_test_prob_need)) + 600
pr_need_max = np.max(y_test_prob_need)
pr_need_min = np.min(y_test_prob_need)
y_test_prob_need = (y_test_prob_need - pr_need_min + 0.01) / (pr_need_max - pr_need_min + 0.01)
