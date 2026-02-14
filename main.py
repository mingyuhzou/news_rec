import os
from tqdm import tqdm
import torch
import polars as pl
import numpy as np
from rank.data.dataset import ValidDataset
from rank.model.NRMS import NRMS
from rank.utils.processData import prepare_embedding_matrix
from recall.transformer.data.tf_dataset import GenRecDataset, GenRecDataLoader
from config.model.tf import cfg
from recall.utils.metrics import calculate_pos_index,ndcg_at_k,recall_at_k
from recall.transformer.model.transformer import TIGER
from config.common import  cfg as hparams
from config.model.nrms import hparams as nrms_cfg
from rank.utils.metric import ndcg,auc
from torch.utils.data import DataLoader

topk_list = [5, 10, 15, 20]

device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

def recall():

    model = TIGER(cfg)
    model.load_state_dict(torch.load(cfg['save_path'],map_location=device))
    model.to(device)

    dataset = GenRecDataset(
        dataset_path=hparams['eval_data'],
        code_path=cfg['code_path'],
        mode='evaluation',
        max_len=cfg['max_len']
    )
    loader = GenRecDataLoader(dataset, batch_size=cfg['infer_size'], shuffle=False)

    code2item=dataset.code2item

    for epoch in range(cfg['num_epochs']):
        print(f"Epoch {epoch + 1}/{cfg['num_epochs']}")
        model.eval()

        recalls = {'Recall@' + str(k): [] for k in topk_list}
        ndcgs = {'NDCG@' + str(k): [] for k in topk_list}

        res = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="recall"):
                input_ids = batch["history"].to(device)  # (B,max_len,4)
                attention_mask = batch["attention_masks"].to(device)
                labels = batch["target"].to(device)
                history=batch["ori_history"]

                preds = model.generate(input_ids, attention_mask, num_beams=cfg['beam_size'])
                preds = preds[:, 1:]  # 排除第一个即decoder_start_token_id
                preds = preds.reshape(input_ids.shape[0], cfg['beam_size'], -1)  # [B,召回个数,code_nums]
                pos_index = calculate_pos_index(preds, labels, maxk=cfg['beam_size'])

                res.append({'history':history, 'preds':preds, 'labels':labels})
                for k in topk_list:
                    recall = recall_at_k(pos_index, k).mean().item()
                    ndcg = ndcg_at_k(pos_index, k).mean().item()
                    recalls['Recall@' + str(k)].append(recall)
                    ndcgs['NDCG@' + str(k)].append(ndcg)

    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    print(avg_recalls, avg_ndcgs)

    rows = []

    for item in res:
        history = item['history']  # [B, L]
        preds = item['preds']  # [B, K, ...]
        labels = item['labels']  # [B]

        B, K = preds.shape[:2]



        for b in range(B):
            hist =history[b]

            pos_code = labels[b].tolist()
            pos_item = code2item[tuple(pos_code)]
            label = [pos_item]


            impressions=[]
            for k in range(K):
                seq = preds[b, k].tolist()
                seq=code2item[tuple(seq)]
                impressions.append(seq)

            rows.append({
                "history": hist,
                "cands": impressions,
                'labels': label
            })

    pl.DataFrame(rows).write_parquet(os.path.join(hparams['data_path'],"valid_behaviors.parquet"))

def rank(data_path):
    nlp_matrix = prepare_embedding_matrix(
        os.path.join(hparams['embed_path'], 'item_emb_title.parquet'), hparams['item_dict']
    )
    nrms_cfg['embed_dim'] = nlp_matrix.shape[1]
    nrms_cfg['item_num'] = nlp_matrix.shape[0]

    rank_model = NRMS(nrms_cfg, weight=nlp_matrix)
    rank_model.load_state_dict(torch.load(nrms_cfg['model_path'], map_location=device))
    rank_model.to(device)

    rank_model.eval()
    res_ndcg5, res_ndcg10, res_auc,res_ndcg20,res_ndcg15= [], [], [],[],[]

    dataset = ValidDataset(data_path, hparams['item_dict'], max_hist_len=25,mode='recall')
    valid_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for click_docs, cand_docs, labels in tqdm(valid_loader, desc="rank"):
            click_docs = click_docs.to(device)
            cand_docs = cand_docs.to(device)
            labels = labels.to(device).squeeze(0)

            logits = rank_model(click_docs, cand_docs)
            logits = logits.squeeze(0)

            if labels.sum() > 0 and labels.sum() < len(labels):
                res_ndcg10.append(ndcg(logits, labels, k=10))
                res_ndcg5.append(ndcg(logits, labels, k=5))
                res_ndcg15.append(ndcg(logits, labels, k=15))
                res_ndcg20.append(ndcg(logits, labels, k=20))
                current_auc = auc(logits, labels)
                if current_auc is not None:
                    res_auc.append(current_auc)

    mean_ndcg5 = np.mean(res_ndcg5) if res_ndcg5 else 0
    mean_ndcg10 = np.mean(res_ndcg10) if res_ndcg10 else 0
    mean_ndcg15 = np.mean(res_ndcg15) if res_ndcg15 else 0
    mean_ndcg20 = np.mean(res_ndcg20) if res_ndcg20 else 0
    mean_auc = np.mean(res_auc) if res_auc else 0

    print({
        "NDCG@5": mean_ndcg5,
        "NDCG@10": mean_ndcg10,
        "NDCG@15": mean_ndcg15,
        "NDCG@20": mean_ndcg20,
        "AUC": mean_auc
    })

if __name__ == "__main__":
    recall()
    rank(os.path.join(hparams['data_path'],"valid_behaviors.parquet"))

