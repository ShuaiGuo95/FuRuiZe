import pickle
import warnings
import numpy as np
from mxnet import ndarray as nd

import auto_encoder
from config import params


DIM = 311
SCALER_AVAILABLE = True
with open(params['auto_encoder_scaler_checkpoint'].format(DIM), 'rb') as f:
    encoder_scaler = pickle.load(f)
encoder = auto_encoder.BackpropEncode(DIM, True)
encoder.initialize()
encoder.load_parameters(params['auto_encoder_checkpoint'].format(DIM))
try:
    with open(params['cnn_scaler_checkpoint'].format(DIM + 150), 'rb') as f:
        cnn_scaler = pickle.load(f)
except Exception:
    warnings.warn('No CNN scaler')
    SCALER_AVAILABLE = False


def get_0612(return_raw=False):
    Xraw = np.load('clean_data/train_0612.npy').astype(np.float32)
    with open('clean_data/train_0612_labels.npy', 'rb') as f:
        Y = np.load(f)
    Xraw_scaled = encoder_scaler.transform(Xraw)
    Xenc = encoder(nd.array(Xraw_scaled)).asnumpy()
    X = np.column_stack((Xraw, Xenc))
    if return_raw:
        return X, Y
    X = cnn_scaler.transform(X)
    X = np.expand_dims(X, axis=1)
    return X, Y


def get_0102():
    Xraw = np.load('clean_data/test_0102.npy').astype(np.float32)
    with open('clean_data/test_0102_labels.npy', 'rb') as f:
        Y = np.load(f)
    Xraw_scaled = encoder_scaler.transform(Xraw)
    Xenc = encoder(nd.array(Xraw_scaled)).asnumpy()
    X = np.column_stack((Xraw, Xenc))
    X = cnn_scaler.transform(X)
    X = np.expand_dims(X, axis=1)
    return X, Y


def get_0305(version='v1'):
    if version == 'v1':
        Xraw = np.load('clean_data/test_0305.npy').astype(np.float32)
    else:
        Xraw = np.load('clean_data/test_0305_v2.npy').astype(np.float32)
    Xraw_scaled = encoder_scaler.transform(Xraw)
    Xenc = encoder(nd.array(Xraw_scaled)).asnumpy()
    X = np.column_stack((Xraw, Xenc))
    X = cnn_scaler.transform(X)
    X = np.expand_dims(X, axis=1)
    return X, None