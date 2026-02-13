import logging
import numpy as np
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import torch
import os
import heapq
from recall.utils.rqave import ensure_dir,delete_file,get_local_time

class Trainer(object):

    def __init__(self, model, data_num, epochs, weight_deacy, learner,
                 lr, lr_scheduler_type, warmup_epochs, eval_step, device, save_limit, ckpt_dir):
        self.logger=logging.getLogger() # 获取日志记录器

        # 学习率相关参数
        self.learner = learner # 优化器名称
        self.lr = lr # 学习略
        self.lr_scheduler_type = lr_scheduler_type # 学习率调度器的类型

        self.epochs = epochs
        self.weight_deacy = weight_deacy
        self.warmup_epochs = warmup_epochs* data_num # 学习率预热阶段的 epoch数，在训练初期学习率从较小值提升到设定的基准学习率
        self.max_steps =epochs * data_num

        self.save_limit=save_limit # 日志容量大小
        self.best_save_heap=[]
        self.newest_save_queue=[]
        self.eval_step=min(eval_step,epochs) # 验证的迭代步数
        self.device = torch.device(device)

        self.ckpt_dir=ckpt_dir # 日志目录
        self.model = model
        saved_model_dir=f'{get_local_time()}'
        self.ckpt_dir = os.path.join(self.ckpt_dir, saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss=np.inf
        self.best_collision_rate=np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model=self.model.to(self.device)

    def _build_optimizer(self):
        """根据优化器名称返回优化器"""
        params=self.model.parameters()
        learner=self.learner
        lr=self.lr
        weight_decay=self.weight_deacy

        if learner.lower()=='adam':
            optimizer=optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif learner.lower()=='sgd':
            optimizer=optim.SGD(params, lr=lr, weight_decay=weight_decay)
        elif learner.lower()=='adamw':
            optimizer=optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=lr)
        return optimizer

    def _get_scheduler(self):
        """返回迭代器"""
        if self.lr_scheduler_type.lower()=='linear':
            # 线性学习器，从0到lr,再从lr到0
            lr_scheduler=get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.warmup_epochs,
                                                         num_training_steps=self.max_steps)
        else:
            # 常数
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler
    def _check_nan(self,loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _train_epoch(self,train_data,epoch_idx):
        """epoch内训练"""
        self.model.train()

        total_loss=0
        total_recon_loss=0

        for batch_idx,data in tqdm(enumerate(train_data),total=len(train_data)):
            data=data.to(self.device)
            self.optimizer.zero_grad()
            # 量化损失和承诺损失
            out,rq_loss,indices=self.model(data)
            # 计算总损失
            loss,loss_recon=self.model.compute_loss(out,rq_loss,xs=data)
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1) # 做梯度裁剪，防止梯度爆炸
            self.optimizer.step()
            self.scheduler.step()
            total_loss+=loss.item()
            total_recon_loss+=loss_recon.item()
        return total_loss,total_recon_loss

    @torch.no_grad()
    def _valid_epoch(self,valid_data):
        """每一步验证"""
        self.model.eval()

        indices_set=set()
        num_samples=0
        for batch_idx,data in tqdm(enumerate(valid_data),total=len(valid_data)):
            # 总长度
            num_samples+=len(data)
            data=data.to(self.device)
            indices=self.model.get_indices(data) # 映射后的各层索引
            indices=indices.view(-1,indices.shape[-1]).cpu().numpy()

            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)
        # 计算碰撞率
        collision_rate = (num_samples - len(list(indices_set))) / num_samples

        return collision_rate


    def _save_checkpoint(self,epoch,collision_rate=1,ckpt_file=None):
        """存储日志"""
        ckpt_path=os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))

        state={
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(state,ckpt_path,pickle_protocol=4)

        self.logger.info(
            "Saving current"+ckpt_path
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

    def fit(self,data):
        cur_eval_step=0
        for epoch_idx in range(self.epochs):
            # train
            train_start_time=time()
            train_loss,trian_recon_loss=self._train_epoch(data,epoch_idx)
            train_end_time=time()
            train_loss_output=self._generate_train_loss_output(epoch_idx,train_start_time,train_end_time,train_loss,trian_recon_loss)
            self.logger.info(train_loss_output)

            # eval
            if (epoch_idx+1)%self.eval_step==0:
                valid_start_time=time()
                collision_rate = self._valid_epoch(data)

                # 只保留最佳的
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1

                valid_end_time=time()
                valid_score_output = (
                    "epoch %d evaluating [time: %.2fs, collision_rate: %f]"
                    % (epoch_idx, valid_end_time - valid_start_time, collision_rate)
                )

                self.logger.info(valid_score_output)
                ckpt_path=self._save_checkpoint(epoch_idx,collision_rate=collision_rate)

                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])
        return self.best_loss,self.best_collision_rate