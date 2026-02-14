import os
from config.common import cfg
hparams={ 'dct_size': 'auto',
        'nhead': 16,
        'embed_dim': 768,
        'encoder_size': 256,
        'v_size': 200,
        'model_path':os.path.join(cfg['data_path'],'ckpt','nrms.pth'),
        "news_file":cfg['news_file'],
        "behaviors_file":os.path.join(cfg['train_data_path'],'behaviors.parquet'),
        "behaviors_dev_file": os.path.join(cfg['dev_data_path'],'behaviors.parquet'),
        "w2v_file": os.path.join(cfg['data_path'],'word2vec/W2vec.pkl'),
        'vocab_file':os.path.join(cfg['data_path'],'word2vec/glove.txt'),
        'lr':1e-3,
        'epochs':100,
        'epoch_step':20,
        'log_dir':cfg['log_dir']
}