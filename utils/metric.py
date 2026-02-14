import polars as pl
import numpy as np
import math

def build_gt(eval_path):
    df = pl.read_parquet(eval_path)
    gt = {}
    for row in df.iter_rows(named=True):
        gt[row['user_id']] = set(row['target'])
    return gt

def recall_at_k_multi(rank_res, gt, k):
    recall_sum = 0.0
    user_cnt = 0

    for uid, gt_items in gt.items():
        if uid not in rank_res or len(gt_items) == 0:
            continue
        preds = rank_res[uid][:k]
        hit = len(set(preds) & gt_items)
        recall_sum += hit / len(gt_items)
        user_cnt += 1

    return recall_sum / user_cnt if user_cnt > 0 else 0

def ndcg_at(rank_res, gt, k):
    ndcg_sum = 0.0
    user_cnt = 0

    for uid, gt_items in gt.items():
        if uid not in rank_res or len(gt_items) == 0:
            continue

        dcg = 0.0
        preds = rank_res[uid][:k]
        for i, item in enumerate(preds):
            if item in gt_items:
                dcg += 1.0 / math.log2(i + 2)

        # IDCG：最理想情况，所有 GT 排在最前
        ideal_hits = min(len(gt_items), k)
        idcg = sum(
            1.0 / math.log2(i + 2)
            for i in range(ideal_hits)
        )

        ndcg_sum += dcg / idcg if idcg > 0 else 0
        user_cnt += 1

    return ndcg_sum / user_cnt if user_cnt > 0 else 0

def auc(rank_res, gt):
    auc_sum = 0.0
    user_cnt = 0

    for uid, pos_items in gt.items():
        if uid not in rank_res:
            continue

        ranked = rank_res[uid]
        pos_items = set(pos_items)

        # 构造负样本
        neg_items = [x for x in ranked if x not in pos_items]
        if len(pos_items) == 0 or len(neg_items) == 0:
            continue

        # item -> rank（越小越好）
        rank_pos = {item: i for i, item in enumerate(ranked)}

        win = 0
        total = 0
        for p in pos_items:
            if p not in rank_pos:
                continue
            for n in neg_items:
                win += 1 if rank_pos[p] < rank_pos[n] else 0
                total += 1

        if total > 0:
            auc_sum += win / total
            user_cnt += 1

    return auc_sum / user_cnt if user_cnt > 0 else 0

def evaluate(rank_res, eval_path, ks=(5,10,20,50)):
    gt = build_gt(eval_path)
    for k in ks:
        r = recall_at_k_multi(rank_res, gt, k)
        n = ndcg_at(rank_res, gt, k)
        print(f"K={k:2d} | Recall@K={r:.4f} | nDCG@K={n:.4f}")
    a = auc(rank_res, gt)
    print(f"AUC = {a:.4f}")