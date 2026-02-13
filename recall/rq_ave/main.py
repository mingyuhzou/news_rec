from typing import Optional
from torch.utils.data import DataLoader
from config.model.rq_ave import cfg
from data.rqave_dataset import EmbDataset
from model.RQ_VAE import RQVAE
from trainer.rqave_trainer import Trainer
import logging
import os

def main(cfg):
    log_dir = cfg["log_dir"]
    log_file = os.path.join(log_dir, "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="a",  # 追加写
    )

    model = RQVAE(
            cfg["in_dim"],
            cfg["num_emb_list"],
            cfg["e_dim"],
            cfg["layers"],
            cfg["dropout_prob"],
            cfg["bn"],
            cfg["loss_type"],
            cfg["quant_loss_weight"],
            cfg["beta"],
            cfg["kmeans_init"],
            cfg["kmeans_iters"],
            cfg["sk_epsilons"],
            cfg["sk_iters"],
    )

    dataset =EmbDataset(os.path.join(cfg['embed_path'],'item_emb_title.parquet'))

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"]
    )


    trainer = Trainer(
        model,
        len(dataloader),
        cfg["epochs"],
        cfg["weight_deacy"],
        cfg["learner"],
        cfg["lr"],
        cfg["lr_scheduler_type"],
        cfg["warmup_epochs"],
        cfg["eval_step"],
        cfg["device"],
        cfg["save_limit"],
        cfg["ckpt_dir"],
    )

    best_loss, best_collision_rate = trainer.fit(dataloader)

    print("Best Loss:", best_loss)
    print("Best Collision Rate:", best_collision_rate)


if __name__ == "__main__":
    main(cfg)