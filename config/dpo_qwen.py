from copy import deepcopy

from config.qwen import cfg as qwen_cfg


cfg = deepcopy(qwen_cfg)
cfg.update({
    "model_name": "Qwen_DPO",

    # 使用 --checkpoint 传入训练好的 GR_Qwen 权重。
    "sft_checkpoint": None,

    "dpo_beta": 0.1,
    "dpo_lr": 1e-5,
    "dpo_epochs": 3,
    "dpo_batch_size": 64,
    "pairs_per_click": 1,
    "num_workers": 0,
    # 0 表示只在 DPO 前后各评估一次；正数表示额外按 epoch 间隔评估。
    "eval_every": 0
})