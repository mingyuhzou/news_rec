import torch
from sklearn.metrics import roc_auc_score
import numpy as np

def ndcg(probs, labels, k):
    # probs: [num_cands], labels: [num_cands]
    num_cands=probs.shape[0]
    k=min(num_cands,k)
    _, indices = torch.topk(probs, k)
    relevant = labels[indices]  # 取出前k个对应的真实标签

    # 计算 DCG: sum(rel / log2(pos + 1))
    # 在推荐中，通常 rel 是 0 或 1
    rank_alpha = torch.arange(2, k + 2, device=probs.device).float()
    dcg = torch.sum(relevant / torch.log2(rank_alpha))

    # 计算 IDCG (假设正样本全在前面)
    idcg_relevant = torch.sort(labels, descending=True)[0][:k]
    idcg = torch.sum(idcg_relevant / torch.log2(rank_alpha))

    if idcg == 0: return 0.
    return (dcg / idcg).item()


def auc(probs, labels):
    """
    计算单个 Session 的 AUC
    probs: [num_cands] - 模型预测的概率/原始分数
    labels: [num_cands] - 真实的标签 (0 或 1)
    """
    y_true = labels.cpu().numpy()
    y_score = probs.cpu().numpy()

    # AUC 要求样本中必须同时包含 0 和 1
    if len(np.unique(y_true)) < 2:
        return None

    return roc_auc_score(y_true, y_score)