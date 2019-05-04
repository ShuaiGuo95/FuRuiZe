import mxnet as mx
import numpy as np
import pandas as pd
import math
from collections import OrderedDict
from mxnet import ndarray as nd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from mxnet import gluon as gl
from mxnet import init
from mxnet.gluon import nn
from mxnet import autograd as ag
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn import feature_selection
from imblearn.ensemble import EasyEnsemble, BalanceCascade
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm_notebook as tqdm
from config import params
import re_rank
import itertools
import os
import utils
import dataset
import random
#np.random.seed(0)
#mx.random.seed(0)


def CNN():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Conv1D(channels=32, kernel_size=33),
            nn.BatchNorm(axis=1),
            nn.Activation('relu'),
            nn.MaxPool1D(pool_size=13),
            
            nn.Flatten(),
            nn.Dense(33, activation='relu'),
            nn.Dropout(0.2),
            nn.Dense(1, activation='sigmoid'),
        )
    return net


def dataIter(X, *data, batch_size, shuffle):
    num_examples = X.shape[0]
    index = list(range(num_examples))
    if shuffle: random.shuffle(index)
    for i in range(0, num_examples, batch_size):
        j = index[i: min(i + batch_size, num_examples)]
        yield [nd.array(X[j])] + [nd.array(d[j]) for d in data] + [len(j)]


def predictProb(params, net, ctx, X):
    pred_probs = []
    num_examples = X.shape[0]
    for data, bs in dataIter(X, batch_size=params['bs'], shuffle=False):
        data = nd.array(data).as_in_context(ctx)
        pred_prob = net(data)
        pred_probs.append(pred_prob)

    pred_probs = nd.concatenate(pred_probs, axis=0).asnumpy()
    pred_probs = pred_probs.squeeze()
    return pred_probs


def evaluateAll(params, net, ctx, *, X, Y, rerank=True, with_preds=False):
    preds = predictProb(params, net, ctx, X)
    if rerank:
        preds = re_rank.rerank(preds)
    auc, ks = utils.evaluate(Y, preds, pos_label=1)
    if with_preds:
        return auc, ks, preds
    return auc, ks


if dataset.SCALER_AVAILABLE is True:
    X_0612, Y_0612 = dataset.get_0612()
    X_0102, Y_0102 = dataset.get_0102()
    X_0305, _ = dataset.get_0305()
    X_0305_v2, _ = dataset.get_0305(version='v2')
def save_best_model(model, info, train_preds):
    WHOLE_0612 = True
    if WHOLE_0612:
        preds_0612 = predictProb(params, model, mx.gpu(), X_0612)
    else:
        preds_0612 = train_preds
    rerank_basis = np.vstack([
        sorted(preds_0612),
        utils.generate_normal_distribution(len(preds_0612))])
    
    # Inference
    preds_0612 = re_rank.rerank(preds_0612, rerank_basis)
    preds_0102 = re_rank.rerank(predictProb(params, model, mx.gpu(), X_0102), rerank_basis)
    preds_0305 = re_rank.rerank(predictProb(params, model, mx.gpu(), X_0305), rerank_basis)
    preds_0305_v2 = re_rank.rerank(predictProb(params, model, mx.gpu(), X_0305_v2), rerank_basis)
    
    # Calculate PSI w.r.t 0612
    psi_0612_0102 = utils.calculate_psi(preds_0612, preds_0102, buckettype='bins', buckets=50, axis=0)
    psi_0612_0305 = utils.calculate_psi(preds_0612, preds_0305, buckettype='bins', buckets=50, axis=0)
    psi_0612_0305_v2 = utils.calculate_psi(preds_0612, preds_0305_v2, buckettype='bins', buckets=50, axis=0)
    
    suffix = ('train_auc_{:.3f}_train_ks_{:.3f}_test_auc_{:.3f}'
              '_test_ks_{:.3f}_psi_0102_{:.5f}_0305_{:.5f}_0305v2_{:.5f}').format(
        info['train_auc'], info['train_ks'], info['test_auc'], info['test_ks'],
        psi_0612_0102, psi_0612_0305, psi_0612_0305_v2)
    with open('model_data/detailed/rerank_basis_{}.npy'.format(suffix), 'wb') as f:
        np.save(f, rerank_basis)
    model.save_parameters('model_data/detailed/cnn_params_{}.mxnet'.format(suffix))
    

def train(train_X, train_y, params, test_X=None, test_y=None):
    ctx = params['ctx']
    net = CNN()
    net.initialize(ctx=ctx, init=init.Xavier())
    trainer = gl.Trainer(net.collect_params(), params['optimizer'],
                         {'learning_rate': params['lr'], 'wd': params['wd'], 'momentum': params['momentum']})
    loss_func = gl.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    indicator = -1
    history = []

    with tqdm(range(1, params['epochs'] + 1)) as progress:
        for epoch in progress:
            train_loss = 0
            train_acc = 0
            for data, label, bs in tqdm(dataIter(train_X, train_y, batch_size=params['bs'], shuffle=True), 
                                        total=train_X.shape[0] // params['bs'], leave=False):
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)

                with ag.record():
                    output = net(data)
                    loss = loss_func(output, label)
                loss.backward()

                trainer.step(params['bs'])
                train_loss += loss.sum().asscalar()
            trainer.set_learning_rate(params['lr'] - (params['lr'] - params['elr']) / params['epochs'] * epoch)

            train_auc, train_ks, train_preds = evaluateAll(
                params, net, ctx, X=train_X, Y=train_y, rerank=False, with_preds=True)
            
            info = OrderedDict(
                loss=train_loss / train_X.shape[0],
                train_auc=train_auc,
                train_ks=train_ks)
            
            if test_X is not None:
                test_auc, test_ks, test_preds = evaluateAll(
                    params, net, ctx, X=test_X, Y=test_y, rerank=False, with_preds=True)
                info['test_auc'] = test_auc
                info['test_ks'] = test_ks
                f1 = test_auc * test_ks / (test_auc + test_ks)
                # Meet certain criterions
                if test_ks > 0.4 and train_auc - test_auc < 0.05 and train_ks - test_ks < 0.06:
                    save_best_model(net, info, train_preds)
            else:
                f1 = train_auc * train_ks / (train_auc + train_ks)
                if f1 > indicator:
                    indicator = f1
                    net.save_parameters(params['model_path'].format(train_X.shape[2]))
            info['f1'] = f1
            progress.set_postfix(**info)
            history.append(info)

    return history


def unbalanceProcess(params, X_train, y_train):
    pos_num = np.sum(y_train == 0)
    neg_num = y_train.shape[0] - pos_num
    ratio = {0: int(pos_num * 0.2),
             1: int(neg_num * 1)}
    y_train = y_train.astype("int")
    sm = BalanceCascade(sampling_strategy=ratio,# replacement=True,
                        random_state=params['random-state'], n_max_subset=10,
                        estimator=LogisticRegression(solver='sag', max_iter=200, random_state=0))

    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
    X_train_res = X_train_res[0];
    y_train_res = y_train_res[0]

    return X_train_res, y_train_res