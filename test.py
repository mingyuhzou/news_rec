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

def evaluate(model,eval_loader,topk_list,beam_size,device,code2item):
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    with torch.no_grad():
        for batch in tqdm(eval_loader,desc="Evaluating"):
            input_ids=batch["history"].to(device) # (B,max_len,4)
            attention_mask=batch["attention_masks"].to(device)
            labels=batch["target"].to(device)

            demo=code2item[tuple(labels[0].tolist())]

            preds=model.generate(input_ids,attention_mask,num_beams=beam_size)

            preds=preds[:,1:] # 排除第一个即decoder_start_token_id
            preds=preds.reshape(input_ids.shape[0],beam_size,-1) # [B,召回个数,code_nums]
            pos_index=calculate_pos_index(preds,labels,maxk=beam_size)


            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls['Recall@' + str(k)].append(recall)
                ndcgs['NDCG@' + str(k)].append(ndcg)
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    print(avg_recalls, avg_ndcgs)

def evaluate_(model,eval_loader,beam_size,device,code2item):
    model.eval()

    with torch.no_grad():
        for batch in tqdm(eval_loader,desc="Evaluating"):
            input_ids=batch["history"].to(device) # (B,max_len,4)
            attention_mask=batch["attention_masks"].to(device)
            labels=batch["target"][0]

            preds=model.generate(input_ids,attention_mask,num_beams=beam_size)

            preds=preds[:,1:] # 排除第一个即decoder_start_token_id
            preds=preds.reshape(input_ids.shape[0],beam_size,-1)[0] # [B,召回个数,code_nums]
            recall=[code2item[tuple(t.tolist())] for t in preds]
            print(set(recall)&set(labels))


def main():
    model = TIGER(cfg)
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(cfg['save_path'], map_location=device))
    model.to(device)

    dataset = GenRecDataset(
        dataset_path='/home/ming/news_rec/Data/dev/valid_df_1.parquet',
        code_path=cfg['code_path'],
        mode='evaluation',
        max_len=cfg['max_len']
    )
    code2item=dataset.code2item
    dataloader = GenRecDataLoader(dataset, batch_size=cfg['infer_size'],shuffle=False)
    for epoch in range(cfg['num_epochs']):
        # evaluate_(model, dataloader, cfg['beam_size'],device,code2item)
        evaluate(model, dataloader, [5,10,15,20],cfg['beam_size'], device,code2item)
main()