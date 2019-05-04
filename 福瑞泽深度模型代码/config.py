import mxnet as mx

params = {
    'lr': 0.01,
    'elr': 0.01,
    'bs': 64,
    'ctx': mx.gpu(0),
    'optimizer': 'sgd',
    'epochs': 200,
    'wd': 0.0005,
    'random-state': 42,
    'momentum': 0.5,
    'auto_encoder_checkpoint': 'model_data/auto_encoder_params_{}.mxnet',
    'auto_encoder_scaler_checkpoint': 'model_data/auto_encoder_scaler_{}.skl',
    'cnn_scaler_checkpoint': 'model_data/cnn_scaler_{}.skl',
    'rerank_basis': 'model_data/rerank_basis.npy',
    'model_path': 'model_data/cnn_params_{}.mxnet',
    # Threshold-Moving refined
    'refined_0612': 'tmp/refined_0612.npy',
    'refined_0102': 'tmp/refined_0102.npy',
    'refined_0305': 'tmp/refined_0305.npy',
    'refined_0305_v2': 'tmp/refined_0305_v2.npy',
    # Normal distribution reranked
    'reranked_0612': 'tmp/reranked_0612.npy',
    'reranked_0102': 'tmp/reranked_0102.npy',
    'reranked_0305': 'tmp/reranked_0305.npy',
    'reranked_0305_v2': 'tmp/reranked_0305_v2.npy',
}