import torch
from torch.utils.data import Dataset,DataLoader
import polars as pl
import pickle
import random
import numpy as np
from rank.utils.processData import processNews

class trainDataset(Dataset):
    def __init__(self, behaviors_file, item_dict_path, max_hist_len, neg_num):
        # 1. 加载行为数据
        self.behaviors = pl.read_parquet(behaviors_file)
        # 2. 加载 news_id -> ItemID 的映射表 (在 notebook 中通过 np.save 保存的)
        self.item_mapping = np.load(item_dict_path, allow_pickle=True).item()
        self.max_hist_len = max_hist_len
        self.neg_num = neg_num
        self.pad_id = 0  # 0 作为填充 ID

    def __len__(self):
        return len(self.behaviors)
    def __getitem__(self, idx):
        row = self.behaviors.row(idx, named=True)
        # 将 news_id 列表转换为 ItemID 列表
        history = [self.item_mapping.get(nid, self.pad_id) for nid in row['history']]

        # 截断与补齐历史
        if len(history) > self.max_hist_len:
            history = history[-self.max_hist_len:]
        while len(history) < self.max_hist_len:
            history.append(self.pad_id)

        # 候选集处理
        impressions = row['impressions']
        pos_cands = [i.split('-')[0] for i in impressions if i.split('-')[1] == '1']
        neg_cands = [i.split('-')[0] for i in impressions if i.split('-')[1] == '0']

        target_pos = random.choice(pos_cands)
        # ... (负采样逻辑与你原代码一致) ...
        if len(neg_cands) > self.neg_num:
            target_negs = random.sample(neg_cands, self.neg_num)
        else:
            # 不够则重复采样
            target_negs = random.choices(neg_cands, k=self.neg_num) if neg_cands else ['<PAD_NID>'] * self.neg_num

        cand_nids = [target_pos] + target_negs
        # 核心改动：将候选新闻转为 ItemID
        cand_ids = [self.item_mapping.get(nid, self.pad_id) for nid in cand_nids]

        return (
            torch.LongTensor(history),  # 现在的形状是 [max_hist_len]
            torch.LongTensor(cand_ids),  # 现在的形状是 [1 + neg_num]
            torch.tensor(0)
        )

class ValidDataset(Dataset):
    def __init__(self, behaviors_file, item_dict_path, max_hist_len):
        self.behaviors = pl.read_parquet(behaviors_file)
        self.item_mapping = np.load(item_dict_path, allow_pickle=True).item() # 将原始的new_id转换为映射后的结果
        self.max_hist_len = max_hist_len
    def __len__(self):
        return len(self.behaviors)
    def __getitem__(self, idx):
        row = self.behaviors.row(idx, named=True)

        # 1. 处理历史记录：转为 ItemID 序列
        history = [self.item_mapping.get(nid, 0) for nid in row['history']]
        if len(history) > self.max_hist_len:
            history = history[-self.max_hist_len:]
        while len(history) < self.max_hist_len:
            history.append(0)

        # 2. 处理候选集：转为 ItemID 序列
        impressions = row['impressions']
        cand_ids = [self.item_mapping.get(i.split('-')[0], 0) for i in impressions]
        labels = [int(i.split('-')[1]) for i in impressions]

        return (
            torch.LongTensor(history),
            torch.LongTensor(cand_ids),
            torch.FloatTensor(labels)
        )


class RankDataset(torch.utils.data.Dataset):
    def __init__(self, eval_df, recall_res, max_hist_len):
        self.uids = eval_df['user'].to_list()
        self.histories = eval_df['history'].to_list()
        self.recall_res = recall_res
        self.max_hist_len = max_hist_len

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        # 历史记录 padding
        hist = self.histories[idx]
        hist = [x for x in hist if x is not None][-self.max_hist_len:]
        hist = [0] * (self.max_hist_len - len(hist)) + hist

        # 召回候选集 (这里假设 recall_res[uid] 已经是过滤后的 ID 列表)
        cands = self.recall_res.get(uid)  # 默认补 0

        return {
            'uid': uid,
            'clicks': torch.tensor(hist, dtype=torch.long),
            'cands': torch.tensor(cands, dtype=torch.long)
        }