import torch
import torch.nn as nn
from rank.utils.metric import ndcg,auc
from config.model.nrms import hparams
from rank.model.NRMS import NRMS
import pickle
import numpy as np
from rank.data.dataset import ValidDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

news_file=hparams['news_file']
behaviors_file=hparams['behaviors_dev_file']
w2v_file=hparams['w2v_file']

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(w2v_file, 'rb') as f:
    data = pickle.load(f)
    w2id = data['w2id']
    matrix = torch.from_numpy(data['embedding']).float() # 这就是从 GloVe 提取的矩阵

# 加载模型
model = NRMS(hparams,matrix)
state_dict=torch.load(hparams['model_path'],map_location=device)
model.load_state_dict(state_dict)
model.to(device)

dataset=ValidDataset(news_file, behaviors_file, w2v_file,max_len=20,max_hist_len=50)
valid_loader = DataLoader(dataset, batch_size=1, shuffle=False)

def evaluate(model):
    model.eval()

    res_ndcg5 = []
    res_ndcg10 = []
    res_auc=[]


    with torch.no_grad():
        for click_docs,cand_docs,labels in tqdm(valid_loader):
            click_docs=click_docs.to(device) # 【1,max_hist_len,max_len】
            cand_docs=cand_docs.to(device) # [1,num_cands,max_len]
            labels=labels.to(device).squeeze(0) # [num_cands]

            logits=model(click_docs,cand_docs) # [1, num_cands]
            logits=logits.squeeze(0)
            if labels.sum()>0 and labels.sum()<len(labels):
                res_ndcg10.append(ndcg(logits,labels,k=10))
                res_ndcg5.append(ndcg(logits,labels,k=5))
                current_auc = auc(logits, labels)
                if current_auc is not None:
                    res_auc.append(current_auc)

    mean_auc = np.mean(res_auc) if res_auc else 0
    mean_ndcg5 = np.mean(res_ndcg5) if res_ndcg5 else 0
    mean_ndcg10 = np.mean(res_ndcg10) if res_ndcg10 else 0
    print(f"AUC:{mean_auc:.4f}")
    print(f"nDCG@5:{mean_ndcg5:.4f}")
    print(f"nDCG@10:{mean_ndcg10:.4f}")

evaluate(model)