from tqdm import tqdm
import torch
import torch.optim as optim
import os
import logging

from rec.transformer.data.qwen_dataset import (
    QwenGenRecDataset,
    QwenGenRecDataLoader
)

from config.qwen import cfg

from rec.utils.metrics import (
    ndcg_at_k,
    recall_at_k
)

from rec.transformer.model.Qwen import Qwen

from utils.logger import create_exp_dir


def train(
        model,
        train_loader,
        optimizer,
        device
):
    model.train()

    total_loss = 0

    for batch in tqdm(
            train_loader,
            desc="Training"
    ):
        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        loss, _ = model(
            input_ids,
            attention_mask,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(
        model,
        eval_loader,
        beam_size,
        device,
        code_len
):
    model.eval()

    topk_list = [1, 10, 20]

    hits = {
        f"Hit@{k}": []
        for k in topk_list
    }

    ndcgs = {
        f"NDCG@{k}": []
        for k in topk_list
    }

    with torch.no_grad():

        for i, batch in enumerate(
                tqdm(eval_loader, desc="Evaluating")
        ):

            input_ids = batch["generate_input_ids"].to(device)
            attention_mask = batch["generate_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
                num_return_sequences=beam_size
            )

            B = input_ids.shape[0]

            #
            # Qwen输出:
            #
            # history + generated SID
            #
            # 去掉history
            #

            history_len = input_ids.shape[1]

            generated_tokens = generated[:, history_len:]
            preds = generated_tokens[:, :code_len]
            preds = preds.reshape(B, beam_size, code_len)

            # labels 末尾是 SID + EOS，因此排除 EOS
            target = labels[:, -(code_len + 1):-1]

            pos_index = (preds == target.unsqueeze(1)).all(dim=-1)

            # 只打印第一个评估 batch 中的第一条样例
            if i == 0:
                sample_history = input_ids[0][attention_mask[0].bool()]
                matched_beams = torch.nonzero(
                    pos_index[0],
                    as_tuple=False
                ).flatten()

                print("=" * 30)
                print("Evaluation sample 0")
                print("history:", sample_history.cpu().tolist())
                print("target SID:", target[0].cpu().tolist())
                print("predicted SIDs:")
                print(preds[0].cpu())
                print(
                    "matched beam ranks:",
                    (matched_beams + 1).cpu().tolist()
                )
                print(
                    "generated tokens (including EOS/PAD):"
                )
                print(generated_tokens[:beam_size].cpu())
                print("=" * 30)

            for k in topk_list:
                hit = recall_at_k(
                    pos_index,
                    k
                ).mean().item()

                ndcg = ndcg_at_k(
                    pos_index,
                    k
                ).mean().item()
                hits[f"Hit@{k}"].append(hit)

                ndcgs[f"NDCG@{k}"].append(ndcg)

    avg_hits = {
        k: sum(v) / len(v)
        for k, v in hits.items()
    }

    avg_ndcgs = {
        k: sum(v) / len(v)
        for k, v in ndcgs.items()
    }

    return avg_hits, avg_ndcgs


def main(cfg):
    cfg = create_exp_dir(cfg)

    logging.basicConfig(
        filename=cfg["log_path"],
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )

    logging.info(cfg)

    model = Qwen(cfg)

    print(model.n_parameters)

    logging.info(model.n_parameters)

    device = torch.device(
        cfg["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    train_dataset = QwenGenRecDataset(
        dataset_path=os.path.join(
            cfg["dataset_path"],
            "train/train.parquet"
        ),

        code_path=cfg["code_path"],

        max_len=cfg["max_len"],

        code_len=cfg["code_len"]
    )

    test_dataset = QwenGenRecDataset(
        dataset_path=os.path.join(
            cfg["dataset_path"],
            "test/test.parquet"
        ),

        code_path=cfg["code_path"],

        max_len=cfg["max_len"],

        code_len=cfg["code_len"]
    )

    train_loader = QwenGenRecDataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True
    )

    test_loader = QwenGenRecDataLoader(
        test_dataset,
        batch_size=cfg["infer_size"],
        shuffle=False
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["lr"]
    )

    model.to(device)

    best_ndcg = 0

    for epoch in range(cfg["num_epochs"]):

        logging.info(
            f"Epoch {epoch + 1}"
        )

        loss = train(
            model,
            train_loader,
            optimizer,
            device
        )

        logging.info(
            f"loss:{loss}"
        )

        if epoch % 10 == 0:

            recalls, ndcgs = evaluate(
                model,
                test_loader,
                cfg["beam_size"],
                device,
                cfg["code_len"]
            )

            logging.info(recalls)

            logging.info(ndcgs)

            if ndcgs["NDCG@20"] > best_ndcg:
                best_ndcg = ndcgs["NDCG@20"]

                torch.save(
                    model.state_dict(),
                    cfg["save_path"]
                )

                logging.info(
                    "best model saved"
                )


if __name__ == "__main__":
    main(cfg)
