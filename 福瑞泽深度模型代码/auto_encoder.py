import mxnet as mx
import numpy as np
import pandas as pd
import random
import pickle
from mxnet import ndarray as nd
from mxnet import gluon as gl
from mxnet import init
from mxnet.gluon import nn
from mxnet import autograd as ag
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn import feature_selection
from sklearn import preprocessing
from tqdm import tqdm_notebook as tqdm
import config
import itertools
np.random.seed(0)


def encoder():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Dense(230, activation='relu'),
            nn.Dense(150)
        )
    return net


def decoder(size):
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Dense(230, activation= 'relu'),
            nn.Dense(size, activation= 'sigmoid')
        )
    return net


class BackpropEncode(gl.Block):
    def __init__(self, size, is_extractor = False, **args):
        super(BackpropEncode, self).__init__(**args)
        self.is_extractor = is_extractor
        with self.name_scope():
            self.encoder = encoder()
            self.decoder = decoder(size)

    def forward(self, x):
        x1 = self.encoder(x)
        if self.is_extractor:
            return x1
        else:
            return self.decoder(x1)


def data_iter(X, y, batch_size, shuffle = True):
    num_examples = y.shape[0]
    index = list(range(num_examples))
    if shuffle: random.shuffle(index)
    for i in range(0, num_examples, batch_size):
        j = index[i: min(i+batch_size, num_examples)]
        yield nd.array(X[j]), nd.array(y[j])


def train_auto_encoder(X_train, net):
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    squre_loss = gl.loss.L2Loss()
    lr = 0.001
    epoch = 100
    trainer = gl.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    num_examples = X_train.shape[0]
    batch_size = 64
    ctx = config.params['ctx']
    with tqdm(range(epoch)) as progress:
        for e in progress:
            total_loss = 0
            noise_X_train = nd.random.normal(loc=0, scale=0.001, shape=X_train.shape)
            noise_X_train1 = (noise_X_train) + nd.array(X_train)
            with tqdm(data_iter(noise_X_train1, X_train, 64),
                      total=num_examples // batch_size) as batch_progress:
                for data, label in batch_progress:
                    data = nd.array(data).as_in_context(ctx)
                    label = nd.array(label).as_in_context(ctx)

                    with ag.record():
                        output = net(data)
                        loss = squre_loss(output, label)
                    trainer.set_learning_rate(lr - e * 1.0 / epoch * lr)
                    loss.backward()
                    trainer.step(batch_size)
                    total_loss += nd.sum(loss).asscalar()
                    batch_progress.set_postfix(total_loss='{:.6f}'.format(total_loss))
            progress.set_postfix(loss=total_loss / num_examples)
    return net, scaler