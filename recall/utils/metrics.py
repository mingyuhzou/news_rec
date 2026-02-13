import torch


def calculate_pos_index(preds,labels,maxk=20):
    '''

    :param preds: (batch_size, maxk, seq_len)
    :param labels: (batch_size, seq_len)
    :return: (batch_size,  maxk)
    '''

    # preds = torch.tensor([
    #     [ [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12] ], # 第一个用户的 3 个预测
    #     [ [10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42] ] # 第二个用户的 3 个预测
    # ])
    preds=preds.detach().cpu()

    # labels = torch.tensor([
    #     [5, 6, 7, 8],
    #     [12, 22, 32, 42]
    # ])
    labels=labels.detach().cpu()

    # 确保预测的个数刚好等于k
    assert (
        preds.shape[1] == maxk
    ), f'preds.shape[1] = {preds.shape[1]} != {maxk}'

    pos_index=torch.zeros((preds.shape[0],maxk),dtype=torch.bool)

    for i in range(preds.shape[0]):
        cur_label=labels[i].tolist()
        for j in range(maxk):
            cur_pred=preds[i,j].tolist()
            if cur_pred==cur_label:
                pos_index[i,j]=True
                break
    # tensor([[False,  True, False],
    #         [False, False,  True]])
    return pos_index

def recall_at_k(pos_index,k=5):
    return pos_index[:,:k].sum(dim=1).cpu().float()

def ndcg_at_k(pos_index, k):
    # Assume only one ground truth item per example
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    # 只有一个正确答案的时候，iDCG=1
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()
