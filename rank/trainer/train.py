import os
import torch
import torch.nn as nn
from config.model.nrms import hparams
from rank.model.NRMS import NRMS
import pytorch_optimizer as optim_
from rank.data.dataset import trainDataset, ValidDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from rank.utils.metric import ndcg,auc
from rank.utils.processData import  prepare_embedding_matrix
from config.common import cfg
import logging
import numpy as np

def get_logger(log_dir, name):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler
    if not logger.handlers:
        # 控制台输出
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 文件输出
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding='utf-8')
        fh.setLevel(logging.INFO)

        # 格式设置
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

logger = get_logger(log_dir=hparams['log_dir'], name="nrms")

nlp_matrix = prepare_embedding_matrix(cfg['embed_path'] + '/item_emb_title.parquet', cfg['item_dict'])
hparams['embed_dim'] = nlp_matrix.shape[1]
hparams['item_num'] = nlp_matrix.shape[0]

# 初始化模型时传入此矩阵
model = NRMS(hparams, weight=nlp_matrix)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset=trainDataset(hparams['behaviors_file'],cfg['item_dict'],max_hist_len=25,neg_num=4)
train_loader=DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,num_workers=4,pin_memory=True)

dataset = ValidDataset(hparams['behaviors_dev_file'], cfg['item_dict'], max_hist_len=25)
valid_loader = DataLoader(dataset, batch_size=1, shuffle=False)

model=model.to(device)
criterion=nn.CrossEntropyLoss()

# 效果更好
optimizer=optim_.Ranger(model.parameters(),lr=hparams['lr'])

# 缩放，防止梯度下溢
scaler = torch.amp.GradScaler('cuda')

def evaluate(model):
    model.eval()
    res_ndcg5, res_ndcg10, res_auc = [], [], []

    with torch.no_grad():
        for click_docs, cand_docs, labels in tqdm(valid_loader, desc="Evaluating"):
            click_docs = click_docs.to(device)
            cand_docs = cand_docs.to(device)
            labels = labels.to(device).squeeze(0)

            logits = model(click_docs, cand_docs)
            logits = logits.squeeze(0)

            if labels.sum() > 0 and labels.sum() < len(labels):
                res_ndcg10.append(ndcg(logits, labels, k=10))
                res_ndcg5.append(ndcg(logits, labels, k=5))
                current_auc = auc(logits, labels)
                if current_auc is not None:
                    res_auc.append(current_auc)

    mean_auc = np.mean(res_auc) if res_auc else 0
    mean_ndcg5 = np.mean(res_ndcg5) if res_ndcg5 else 0
    mean_ndcg10 = np.mean(res_ndcg10) if res_ndcg10 else 0

    # 使用 logger 记录结果
    logger.info("=" * 30)
    logger.info(f"Evaluation Results:")
    logger.info(f"AUC:      {mean_auc:.4f}")
    logger.info(f"nDCG@5:   {mean_ndcg5:.4f}")
    logger.info(f"nDCG@10:  {mean_ndcg10:.4f}")
    logger.info("=" * 30)

    return {
        'auc': mean_auc,
        'ndcg5': mean_ndcg5,
        'ndcg10': mean_ndcg10
    }


def train():
    logger.info("=" * 20 + " 开始训练 " + "=" * 20)
    best_auc = 0

    for epoch in range(1, hparams['epochs'] + 1):
        model.train()
        total_loss = 0

        # 使用 pbar 对象，方便在循环内写日志
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", postfix={"loss": 0})

        for click_docs, cand_docs, labels in pbar:
            click_docs, cand_docs, labels = click_docs.to(device), cand_docs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                scores = model(click_docs, cand_docs)
                loss = criterion(scores, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_loss = loss.item()
            total_loss += current_loss

            # 更新进度条右侧的显示
            pbar.set_postfix(loss=f"{current_loss:.4f}")

        avg_loss = total_loss / len(train_loader)
        # Epoch 结束后，使用 logger 记录到文件和控制台
        logger.info(f"Epoch {epoch} 完成 | Avg Loss: {avg_loss:.6f}")

        # 每隔指定 epoch 验证一次
        if epoch == 1 or epoch % hparams['epoch_step'] == 0:
            logger.info(f"正在进行第 {epoch} 轮验证...")

            metrics = evaluate(model)
            current_auc = metrics['auc']

            if current_auc > best_auc:
                best_auc = current_auc
                torch.save(model.state_dict(), hparams['model_path'])
                logger.info(f"找到更优模型 (AUC: {best_auc:.4f}, NDCG5: {metrics['ndcg5']}, NDCG10: {metrics['ndcg10']}), 已保存至: {hparams['model_path']}")

train()