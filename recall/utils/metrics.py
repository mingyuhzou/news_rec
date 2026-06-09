import torch

def recall_at_k(pos_index,k=5):
    return pos_index[:,:k].sum(dim=1).cpu().float()

def ndcg_at_k(pos_index, k):
    # Assume only one ground truth item per example
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    # 只有一个正确答案的时候，iDCG=1
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()
