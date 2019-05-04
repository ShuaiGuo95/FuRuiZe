import pandas as pd
import numpy as np
import config


def rerank(preds, rerank_basis=None):
    '''
    :param preds: (N,), Probabilities to be bad person
    :return: Re-ranked probabilities to be bad person
    '''
    if rerank_basis is None:
        rerank_basis = np.load(config.params['rerank_basis'])
    mapping_to = rerank_basis[1]
    mapping_from = rerank_basis[0]

    rerank_preds = np.zeros((preds.shape[0],))
    for idx, prob in enumerate(preds):
        dist = np.square(prob - mapping_from)
        mapping_idx = dist.argmin()
        rerank_preds[idx] = mapping_to[mapping_idx]
    return rerank_preds