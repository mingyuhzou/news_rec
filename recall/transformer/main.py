from tqdm import tqdm
import torch
import torch.optim as optim
import logging
from recall.transformer.data.tf_dataset import GenRecDataset, GenRecDataLoader
from config.model.tf import cfg
from recall.utils.metrics import calculate_pos_index,ndcg_at_k,recall_at_k
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

def evaluate(model,eval_loader,topk_list,beam_size,device):
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    i=0
    with torch.no_grad():
        for batch in tqdm(eval_loader,desc="Evaluating"):
            input_ids=batch["history"].to(device) # (B,max_len,4)
            attention_mask=batch["attention_masks"].to(device)
            labels=batch["target"].to(device)

            preds=model.generate(input_ids,attention_mask,num_beams=beam_size)

            if i == 0:
                print("\n" + "=" * 30)
                print(f"DEBUG: Batch 0 详情")
                # preds 此时包含 decoder_start_token_id，通常在第一位
                print(f"原始预测 (含起始符) Shape: {preds.shape}")
                print(f"第一个样本的前 3 个 Beam 预测:\n{preds[:3 * beam_size:beam_size]}")
                print(f"第一个样本的真实标签 Label: {labels[0]}")
                print("=" * 30 + "\n")

            preds=preds[:,1:] # 排除第一个即decoder_start_token_id
            preds=preds.reshape(input_ids.shape[0],beam_size,-1) # [B,召回个数,code_nums]
            pos_index=calculate_pos_index(preds,labels,maxk=beam_size)

            i+=1

            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls['Recall@' + str(k)].append(recall)
                ndcgs['NDCG@' + str(k)].append(ndcg)
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs



def main():
    logging.basicConfig(
        filename=cfg['log_path'],
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
        dataset_path=cfg['dataset_path'] + '/train_df.parquet',
        code_path=cfg['code_path'],
        mode='train',
        max_len=cfg['max_len']
    )
    validation_dataset = GenRecDataset(
        dataset_path=cfg['dataset_path'] + '/valid_df.parquet',
        code_path=cfg['code_path'],
        mode='evaluation',
        max_len=cfg['max_len']
    )
    test_dataset = GenRecDataset(
        dataset_path=cfg['dataset_path'] + '/test_df.parquet',
        code_path=cfg['code_path'],
        mode='evaluation',
        max_len=cfg['max_len']
    )

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    validation_dataloader = GenRecDataLoader(validation_dataset, batch_size=cfg['infer_size'], shuffle=False)
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
        avg_recalls, avg_ndcgs = evaluate(model, validation_dataloader, cfg['topk_list'], cfg['beam_size'],
                                          device)
        logging.info(f"Validation Dataset: {avg_recalls}")
        logging.info(f"Validation Dataset: {avg_ndcgs}")

        if avg_ndcgs['NDCG@20'] > best_ndcg:
            best_ndcg = avg_ndcgs['NDCG@20']
            early_stop_counter = 0  # Reset early stop counter
            test_avg_recalls, test_avg_ndcgs = evaluate(model, test_dataloader, cfg['topk_list'],
                                                        cfg['beam_size'], device)
            logging.info(f"Best NDCG@20: {best_ndcg}")
            logging.info(f"Test Dataset: {test_avg_recalls}")
            logging.info(f"Test Dataset: {test_avg_ndcgs}")
            # Save the best model
            torch.save(model.state_dict(), cfg['save_path'])
            logging.info(f"Best model saved to {cfg['save_path']}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in NDCG@20. Early stop counter: {early_stop_counter}")
            if early_stop_counter >= cfg['early_stop']:
                logging.info("Early stopping triggered.")
                break

if __name__ == "__main__":

    main()