import logging
from collections import Counter

import numpy as np
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import torch
import os
from torch.utils.tensorboard import SummaryWriter

from data.rqave_dataset import EmbDataset
from model.RQ_VAE import RQVAE
from recall.utils.rqave import ensure_dir, delete_file, get_local_time
from torch.utils.data import DataLoader


class Trainer(object):

    def __init__(self, model, data_num, epochs, weight_deacy, learner,
                 lr, lr_scheduler_type, warmup_epochs, eval_step, device, save_limit, ckpt_dir):
        self.logger = logging.getLogger()  # 获取日志记录器

        # 学习率相关参数
        self.learner = learner  # 优化器名称
        self.lr = lr  # 学习略
        self.lr_scheduler_type = lr_scheduler_type  # 学习率调度器的类型

        self.epochs = epochs
        self.weight_deacy = weight_deacy
        self.warmup_epochs = warmup_epochs * data_num  # 学习率预热阶段的 epoch数，在训练初期学习率从较小值提升到设定的基准学习率
        self.max_steps = epochs * data_num

        self.save_limit = save_limit  # 日志容量大小
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(eval_step, epochs)  # 验证的迭代步数
        self.device = torch.device(device)

        self.ckpt_dir = ckpt_dir  # 日志目录
        self.model = model
        saved_model_dir = f'{get_local_time()}'
        self.ckpt_dir = os.path.join(self.ckpt_dir, saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

        self.writer = SummaryWriter(
            log_dir=os.path.join(
                self.ckpt_dir,
                "tensorboard"
            )
        )

    def _build_optimizer(self):
        """根据优化器名称返回优化器"""
        params = self.model.parameters()
        learner = self.learner
        lr = self.lr
        weight_decay = self.weight_deacy

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=lr)
        return optimizer

    def _get_scheduler(self):
        """返回迭代器"""
        if self.lr_scheduler_type.lower() == 'linear':
            # 线性学习器，从0到lr,再从lr到0
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_epochs,
                                                           num_training_steps=self.max_steps)
        else:
            # 常数
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _train_epoch(self, train_data, epoch_idx):
        """
        epoch训练
        """

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0

        for batch_idx, data in enumerate(train_data):
            step = epoch_idx * len(train_data) + batch_idx

            data = data.to(self.device)

            self.optimizer.zero_grad()

            # RQVAE forward
            out, rq_loss, indices = self.model(data)

            # loss
            loss, loss_recon = self.model.compute_loss(
                out,
                rq_loss,
                xs=data
            )

            self._check_nan(loss)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1
            )

            self.optimizer.step()

            self.scheduler.step()

            # =====================
            # accumulate
            # =====================

            total_loss += loss.item()

            total_recon_loss += loss_recon.item()

            total_quant_loss += rq_loss.item()

        return (
            total_loss / len(train_data),
            total_recon_loss / len(train_data),
            total_quant_loss / len(train_data)
        )

    @torch.no_grad()
    def _valid_epoch(
            self,
            valid_data
    ):

        self.model.eval()

        # SID集合
        indices_set = set()

        # 每层code频率
        codebook_counter = [
            Counter()
            for _ in range(len(self.model.num_emb_list))
        ]

        num_sample = 0

        for batch_idx, data in enumerate(valid_data):

            num_sample += len(data)

            data = data.to(self.device)

            indices = self.model.get_indices(data)

            indices = indices.view(
                -1,
                indices.shape[-1]
            ).cpu().numpy()

            for index in indices:

                # =====================
                # SID collision
                # =====================

                code = "-".join(
                    [
                        str(int(x))
                        for x in index
                    ]
                )

                indices_set.add(code)

                # =====================
                # code frequency
                # =====================

                for level, idx in enumerate(index):
                    codebook_counter[level][int(idx)] += 1

        # =====================
        # SID collision rate
        # =====================

        collision_rate = (
                                 num_sample - len(indices_set)
                         ) / num_sample

        # =====================
        # perplexity
        # =====================

        perplexity = []

        normalized_perplexity = []

        for level, counter in enumerate(codebook_counter):
            counts = np.array(
                list(counter.values())
            )

            probs = counts / counts.sum()

            ppl = np.exp(
                -np.sum(
                    probs *
                    np.log(probs + 1e-10)
                )
            )

            perplexity.append(ppl)

            # normalize到0-1
            normalized_perplexity.append(
                ppl / self.model.num_emb_list[level]
            )

        return {

            "collision_rate": collision_rate,

            "perplexity": perplexity,

            "normalized_perplexity": normalized_perplexity

        }

    def _save_checkpoint(self, epoch, collision_rate=1):
        """存储日志"""

        ckpt_path = os.path.join(
            os.path.dirname(self.ckpt_dir),
            'best_collision_rate.pth'
        )

        state = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            "Saving current" + ckpt_path
        )
        return ckpt_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        """"打印输出"""
        return (
            f"epoch {epoch_idx} training "
            f"[time: {e_time - s_time:.2f}s, "
            f"train loss: {loss:.4f}, "
            f"reconstruction loss: {recon_loss:.4f}]"
        )

    def fit(self, data):

        for epoch_idx in tqdm(range(self.epochs)):

            train_start_time = time()

            train_loss, train_recon_loss, train_quant_loss = (
                self._train_epoch(
                    data,
                    epoch_idx
                )
            )

            train_end_time = time()
            
            self.writer.add_scalar(
                "Loss/Total",
                train_loss,
                epoch_idx
            )

            self.writer.add_scalar(
                "Loss/Reconstruction",
                train_recon_loss,
                epoch_idx
            )

            self.writer.add_scalar(
                "Loss/Quantization",
                train_quant_loss,
                epoch_idx
            )
            self.logger.info(
                f"""
                    epoch {epoch_idx}
                
                    time:
                    {train_end_time - train_start_time:.2f}s
                
                    total loss:
                    {train_loss:.6f}
                
                    reconstruction loss:
                    {train_recon_loss:.6f}
                
                    quantization loss:
                    {train_quant_loss:.6f}
                    """
            )

            # =====================
            # validation
            # =====================

            if (epoch_idx + 1) % self.eval_step == 0:

                valid_result = self._valid_epoch(
                    data
                )

                collision_rate = (
                    valid_result["collision_rate"]
                )

                perplexity = (
                    valid_result["perplexity"]
                )

                normalized_ppl = (
                    valid_result["normalized_perplexity"]
                )

                self.logger.info(
                    f"""
                        SID collision:
                        {collision_rate}
                    
                        Perplexity:
                        {perplexity}
                    
                        Normalized perplexity:
                        {normalized_ppl}
                        """
                )

                # =====================
                # TensorBoard
                # =====================

                self.writer.add_scalar(
                    "Metric/SID_Collision",
                    collision_rate,
                    epoch_idx
                )

                for level in range(len(perplexity)):
                    self.writer.add_scalar(
                        f"Metric/Perplexity_Level_{level}",
                        perplexity[level],
                        epoch_idx
                    )

                    self.writer.add_scalar(
                        f"Metric/Normalized_Perplexity_Level_{level}",
                        normalized_ppl[level],
                        epoch_idx
                    )

                # =====================
                # save best model
                # =====================

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate

                    self._save_checkpoint(
                        epoch_idx,
                        collision_rate
                    )

        self.writer.close()

        return (
            self.best_loss,
            self.best_collision_rate
        )


def train(cfg):
    log_dir = cfg["log_dir"]

    os.makedirs(
        log_dir,
        exist_ok=True
    )

    log_dir = "/news_rec/log"

    os.makedirs(
        log_dir,
        exist_ok=True
    )

    logging.basicConfig(
        filename=os.path.join(
            log_dir,
            "train.log"
        ),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
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

    dataset = EmbDataset(os.path.join(cfg['embed_path'], 'item_emb_title.parquet'))

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