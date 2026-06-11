import os
import json
import logging
from datetime import datetime

def create_exp_dir(cfg):
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_name = (
        f"{time_str}_"
        f"TIGER_"
        f"lr{cfg['lr']}_"
        f"bs{cfg['batch_size']}_"
        f"beam{cfg['beam_size']}_"
        f"code{cfg['code_len']}"
    )

    exp_dir = os.path.join(cfg.get("exp_root", "runs"), exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    cfg["exp_dir"] = exp_dir
    cfg["log_path"] = os.path.join(exp_dir, "train.log")
    cfg["save_path"] = os.path.join(exp_dir, "best_model.pth")
    cfg["config_save_path"] = os.path.join(exp_dir, "config.json")

    with open(cfg["config_save_path"], "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False, default=str)

    return cfg