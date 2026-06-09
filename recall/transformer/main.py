from tqdm import tqdm
import torch
import torch.optim as optim
import logging
import os
from recall.transformer.data.tf_dataset import GenRecDataset, GenRecDataLoader
from config.model.tf import cfg
from recall.utils.metrics import ndcg_at_k,recall_at_k
from recall.transformer.model.transformer import TIGER

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

def evaluate(model, eval_loader, beam_size, device):
    model.eval()

    topk_list = [1, 10, 20]

    hits = {f"Hit@{k}": [] for k in topk_list}
    ndcgs = {f"NDCG@{k}": [] for k in topk_list}

    i = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_masks"].to(device)
            labels = batch["target"].to(device)

            preds = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
                num_return_sequences=beam_size,
            )

            B = input_ids.shape[0]
            code_len = cfg["code_len"]   # 这里是 4，不包含 EOS

            if i == 0:
                print("\n" + "=" * 30)
                print("DEBUG: Batch 0")
                print(f"input_ids shape: {input_ids.shape}")
                print(f"labels shape: {labels.shape}")
                print(f"preds raw shape: {preds.shape}")
                print(f"first sample raw beams:\n{preds[:beam_size]}")
                print(f"first label: {labels[0]}")
                print("=" * 30 + "\n")

            # raw preds: [0, code1, code2, code3, code4, eos]
            # 只取 code，不取 0，也不取 eos
            preds = preds[:, 1:1 + code_len]

            preds = preds.reshape(B, beam_size, code_len)

            # labels: [code1, code2, code3, code4, eos]
            # 只取 code，不取 eos
            labels = labels[:, :code_len]

            pos_index = (preds == labels.unsqueeze(1)).all(dim=-1)

            for k in topk_list:
                hit = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()

                hits[f"Hit@{k}"].append(hit)
                ndcgs[f"NDCG@{k}"].append(ndcg)

            i += 1

    avg_hits = {
        k: sum(v) / len(v)
        for k, v in hits.items()
    }

    avg_ndcgs = {
        k: sum(v) / len(v)
        for k, v in ndcgs.items()
    }

    return avg_hits, avg_ndcgs



def main():
    logging.basicConfig(
        filename=cfg['log_path'],
        filemode="w",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Configuration: {cfg}")

    # Initialize model
    model = TIGER(cfg)
    print(model.n_parameters)
    logging.info(model.n_parameters)

    # Check if the device is available
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

    train_dataset = GenRecDataset(
        dataset_path=os.path.join(cfg['dataset_path'], 'train/train_df.parquet'),
        code_path=cfg['code_path'],
        max_len=cfg['max_len'],
        code_len=cfg['code_len'],
    )
    test_dataset = GenRecDataset(
        dataset_path=os.path.join(cfg['dataset_path'],'test/test_df.parquet'),
        code_path=cfg['code_path'],
        max_len=cfg['max_len'],
        code_len=cfg['code_len'],
    )

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=cfg['infer_size'], shuffle=False)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

    # Train the model
    model.to(device)
    best_ndcg = 0.0
    early_stop_counter = 0

    for epoch in range(cfg['num_epochs']):
        logging.info(f"Epoch {epoch + 1}/{cfg['num_epochs']}")
        train_loss = train(model, train_dataloader, optimizer, device)
        logging.info(f"Training loss: {train_loss}")
        # Evaluate the model
        avg_recalls, avg_ndcgs = evaluate(model, test_dataloader,cfg['beam_size'], device)
        logging.info(f"Validation Dataset: {avg_recalls}")
        logging.info(f"Validation Dataset: {avg_ndcgs}")

        if avg_ndcgs['NDCG@20'] > best_ndcg:
            best_ndcg = avg_ndcgs['NDCG@20']
            torch.save(model.state_dict(), cfg['save_path'])
            logging.info(f"Best model saved to {cfg['save_path']}")


if __name__ == "__main__":

    main()