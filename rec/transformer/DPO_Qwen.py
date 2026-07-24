import argparse
import copy
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config.dpo_qwen import cfg
from rec.transformer.GR_Qwen import evaluate
from rec.transformer.data.qwen_dataset import (
    QwenGenRecDataLoader,
    QwenGenRecDataset
)
from rec.transformer.data.qwen_dpo_dataset import (
    QwenDPODataLoader,
    QwenDPODataset
)
from rec.transformer.model.Qwen import Qwen
from utils.logger import create_exp_dir


def sequence_log_probs(model, input_ids, attention_mask, labels):
    '''模型认为回答的概率是多少 log \pi(y|x)'''

    '''
        input_ids: [B, L]
        logits:    [B, L, V] 输出选每个token的概率
        '''
    logits = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask
    ).logits

    ''' 
        输入：  [x0, x1, x2, x3]
        预测：       x1  x2  x3
        
        logits[:, 0] 预测 labels[:, 1]
        logits[:, 1] 预测 labels[:, 2]
        logits[:, 2] 预测 labels[:, 3]
        
        形状变为
        shifted_logits: [B, L-1, V]
        shifted_labels: [B, L-1]
    '''
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    # 只计算response的loss
    response_mask = shifted_labels.ne(-100)
    # [-100, -100, 66, 340, 700, 770, 1]->[   0,    0, 66, 340, 700, 770, 1]
    safe_labels = shifted_labels.masked_fill(~response_mask, 0)

    token_log_probs = F.log_softmax(
        shifted_logits,
        dim=-1
    ).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    return (token_log_probs * response_mask).sum(dim=-1)


def preference_log_probs(model, batch, device):
    '''同时计算chosen序列概率和rejected序列概率'''

    chosen_ids = batch["chosen_input_ids"].to(device)
    chosen_mask = batch["chosen_attention_mask"].to(device)
    chosen_labels = batch["chosen_labels"].to(device)
    rejected_ids = batch["rejected_input_ids"].to(device)
    rejected_mask = batch["rejected_attention_mask"].to(device)
    rejected_labels = batch["rejected_labels"].to(device)

    batch_size = chosen_ids.shape[0]
    # 拼接一次性计算
    log_probs = sequence_log_probs(
        model,
        torch.cat((chosen_ids, rejected_ids), dim=0),
        torch.cat((chosen_mask, rejected_mask), dim=0),
        torch.cat((chosen_labels, rejected_labels), dim=0)
    )
    return log_probs[:batch_size], log_probs[batch_size:]

def train_dpo_epoch(
        policy,
        reference,
        train_loader,
        optimizer,
        beta,
        device
):
    policy.train()
    reference.eval() # 参考模型冻结参数

    totals = {
        "loss": 0.0,
        "preference_accuracy": 0.0,
        "reward_margin": 0.0
    }
    total_samples = 0

    for batch in tqdm(train_loader, desc="DPO training"):
        # 计算policy模型的概率
        policy_chosen, policy_rejected = preference_log_probs(
            policy,
            batch,
            device
        )

        with torch.no_grad():
            # 计算参考模型的概率
            reference_chosen, reference_rejected = preference_log_probs(
                reference,
                batch,
                device
            )

        policy_ratio = policy_chosen - policy_rejected
        reference_ratio = reference_chosen - reference_rejected
        reward_margin = beta * (policy_ratio - reference_ratio)
        loss = -F.logsigmoid(reward_margin).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = policy_chosen.shape[0]
        total_samples += batch_size
        totals["loss"] += loss.item() * batch_size
        totals["preference_accuracy"] += (
            policy_ratio > reference_ratio
        ).float().sum().item()
        totals["reward_margin"] += reward_margin.detach().sum().item()

    return {
        key: value / total_samples
        for key, value in totals.items()
    }


def load_checkpoint(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def report_evaluation(stage, model, eval_loader, config, device):
    hits, ndcgs = evaluate(
        model,
        eval_loader,
        config["beam_size"],
        device,
        config["code_len"]
    )
    print(f"{stage} Hit: {hits}")
    print(f"{stage} NDCG: {ndcgs}")
    logging.info("%s Hit: %s", stage, hits)
    logging.info("%s NDCG: %s", stage, ndcgs)
    return hits, ndcgs


def main(config, checkpoint_path):
    config = copy.deepcopy(config)
    config["sft_checkpoint"] = checkpoint_path
    config["lr"] = config["dpo_lr"]
    config["batch_size"] = config["dpo_batch_size"]
    config = create_exp_dir(config)

    logging.basicConfig(
        filename=config["log_path"],
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    logging.info(config)
    set_seed(config["seed"])

    device = torch.device(
        config["device"] if torch.cuda.is_available() else "cpu"
    )

    policy = Qwen(config)
    load_checkpoint(policy, checkpoint_path)
    policy.to(device)

    train_dataset = QwenDPODataset(
        dataset_path=os.path.join(
            config["dataset_path"],
            "train/train.parquet"
        ),
        code_path=config["code_path"],
        max_len=config["max_len"],
        pairs_per_click=config["pairs_per_click"],
        seed=config["seed"]
    )
    train_loader = QwenDPODataLoader(
        train_dataset,
        batch_size=config["dpo_batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pad_token=config["pad_token_id"]
    )

    eval_dataset = QwenGenRecDataset(
        dataset_path=os.path.join(
            config["dataset_path"],
            "test/test.parquet"
        ),
        code_path=config["code_path"],
        max_len=config["max_len"],
        code_len=config["code_len"]
    )
    eval_loader = QwenGenRecDataLoader(
        eval_dataset,
        batch_size=config["infer_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    logging.info("DPO pairs: %s", len(train_dataset))

    # 先固定测试集测训练好的 GR_Qwen，作为强化学习前基线。
    baseline_hits, baseline_ndcgs = report_evaluation(
        "Before DPO",
        policy,
        eval_loader,
        config,
        device
    )

    # reference 必须与 DPO 开始前的 policy 完全相同并保持冻结。
    reference = copy.deepcopy(policy)
    reference.requires_grad_(False)
    reference.eval()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config["dpo_lr"]
    )
    best_ndcg = float("-inf")
    best_path = config["save_path"]
    best_saved = False

    for epoch in range(config["dpo_epochs"]):
        stats = train_dpo_epoch(
            policy,
            reference,
            train_loader,
            optimizer,
            config["dpo_beta"],
            device
        )
        print(f"DPO epoch {epoch + 1}: {stats}")
        logging.info("DPO epoch %s: %s", epoch + 1, stats)

        if (
            config["eval_every"] <= 0
            or (epoch + 1) % config["eval_every"] != 0
        ):
            continue

        _, ndcgs = report_evaluation(
            f"After DPO epoch {epoch + 1}",
            policy,
            eval_loader,
            config,
            device
        )
        if ndcgs["NDCG@20"] > best_ndcg:
            best_ndcg = ndcgs["NDCG@20"]
            torch.save(policy.state_dict(), best_path)
            best_saved = True
            logging.info("Best DPO model saved: %s", best_path)

    # 若启用了中途评估，使用其中最好的 checkpoint；默认保存最终模型。
    if not best_saved:
        torch.save(policy.state_dict(), best_path)
    load_checkpoint(policy, best_path)
    final_hits, final_ndcgs = report_evaluation(
        "After DPO",
        policy,
        eval_loader,
        config,
        device
    )

    print("=" * 50)
    print("Before DPO:", baseline_hits, baseline_ndcgs)
    print("After DPO:", final_hits, final_ndcgs)
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=cfg["sft_checkpoint"],
        help="trained GR_Qwen checkpoint"
    )
    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("--checkpoint is required")
    main(cfg, args.checkpoint)