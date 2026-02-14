import os
from config.common import cfg as config
cfg={
    "in_dim": 768,
    "num_emb_list": [ 256,256,256 ],
    "beta": 0.25,
    "layers": [ 512,256,128,64],
    "quant_loss_weight": 1.0,
    "e_dim": 32,
    "dropout_prob": 0.0,
    "bn": False,
    "loss_type": "mse",
    "kmeans_init": True,
    "kmeans_iters": 100,
    "sk_epsilons": [ 0.0, 0.0, 0.003 ],
    "sk_iters": 50,

    "data_path": os.path.join(config['data_path'],'item_emb.parquet'),
    "embed_path":config['embed_path'],
    "batch_size": 1024,
    "num_workers": 4,

    "learner": "AdamW",
    "lr": 1e-3,
    "lr_scheduler_type": "linear",
    "warmup_epochs": 50,
    "eval_step": 50,
    "device": "cuda:0",
    "save_limit": 50,
    "ckpt_dir": os.path.join(config["data_path"],"ckpt"),
    "epochs": 3000,
    "weight_deacy": 1e-4,

    "log_dir": config['log_dir'],
}