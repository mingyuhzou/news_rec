from tqdm import tqdm
import torch
import torch.optim as optim
import logging
from recall.transformer.data.tf_dataset import GenRecDataset, GenRecDataLoader
from config.model.tf import cfg
from recall.utils.metrics import calculate_pos_index,ndcg_at_k,recall_at_k
from recall.transformer.model.transformer import TIGER
from config.common import  cfg as hparams
from config.model.nrms import cfg as nrms_cfg

def train(model,train_loader,optimizer,device):
    model.train()
    total_loss=0
    for batch in tqdm(train_loader,desc="Training"):
        input_ids=batch["history"].to(device)
        attention_mask=batch["attention_masks"].to(device)
        labels=batch["target"].to(device)

        optimizer.zero_grad()
        loss,_=model(input_ids,attention_mask,labels)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    return total_loss/len(train_loader)

def main():
    logging.basicConfig(
        filename=cfg['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Configuration: {cfg}")

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(device)


    model = TIGER(cfg)
    model.load_state_dict(torch.load(cfg['save_path']),map_location=device)
    model.to(device)

    nlp_matrix = prepare_embedding_matrix(
        os.path.join(hparams['embed_path'], 'item_emb_title.parquet'), hparams['item_dict']
    )
    rank_model = NRMS(nrms_cfg, weight=nlp_matrix)
    rank_model.load_state_dict(torch.load(nrms_cfg['model_path'], map_location=device))
    rank_model.to(device)

    dataset = GenRecDataset(
        dataset_path=hparams['eval_data'],
        code_path=cfg['code_path'],
        mode='evaluation',
        max_len=cfg['max_len']
    )

    loader = GenRecDataLoader(dataset, batch_size=cfg['infer_size'], shuffle=False)

    # Train the model
    best_ndcg = 0.0
    early_stop_counter = 0

    for epoch in range(cfg['num_epochs']):
        print(f"Epoch {epoch + 1}/{cfg['num_epochs']}")
        model.eval()

        recalls = {'Recall@' + str(k): [] for k in topk_list}
        ndcgs = {'NDCG@' + str(k): [] for k in topk_list}

        res = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch["history"].to(device)  # (B,max_len,4)
                attention_mask = batch["attention_masks"].to(device)
                labels = batch["target"].to(device)
                history=batch["history"].to(device)

                preds = model.generate(input_ids, attention_mask, num_beams=beam_size)
                preds = preds[:, 1:]  # 排除第一个即decoder_start_token_id
                preds = preds.reshape(input_ids.shape[0], beam_size, -1)  # [B,召回个数,code_nums]
                pos_index = calculate_pos_index(preds, labels, maxk=beam_size)

                i += 1
                res.append(preds, labels)
                for k in topk_list:
                    recall = recall_at_k(pos_index, k).mean().item()
                    ndcg = ndcg_at_k(pos_index, k).mean().item()
                    recalls['Recall@' + str(k)].append(recall)
                    ndcgs['NDCG@' + str(k)].append(ndcg)
                res.append({'history': history, 'preds': preds, 'labels':labels})
        avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
        avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}

        print(avg_recalls, avg_ndcgs)

if __name__ == "__main__":
    main()
