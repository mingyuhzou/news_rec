# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import polars as pl
import pickle
import random
import re
from rank.utils.processData import processNews
# %%
class trainDataset(Dataset):
    def __init__(self,news_file,behaviors_file,w2v_file,max_len,max_hist_len,neg_num):
        # new_id->embed_id
        self.news_dict = processNews(news_file,w2v_file,max_len)

        self.behaviors=pl.read_parquet(behaviors_file)
        self.max_len=max_len
        self.max_hist_len=max_hist_len
        self.neg_num=neg_num

        with open(w2v_file,'rb') as f:
            self.w2id=pickle.load(f)['w2id']

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row=self.behaviors.row(idx,named=True)

        history=row['history']
        # 用户点击长度只保留最近的
        if len(history)>self.max_hist_len:
            history=history[-self.max_hist_len:]

        # 转换
        click_docs=[self.news_dict.get(nid,self.news_dict['<PAD>']) for nid in history]
        # 补齐历史长度
        pad_vec = self.news_dict['<PAD>']
        # 截断
        while len(click_docs) < self.max_hist_len:
            click_docs.append(pad_vec)
        
        impressions=row['impressions']
        pos_cands=[i.split('-')[0] for i in impressions if i.split('-')[1]=='1']
        neg_cands=[i.split('-')[0] for i in impressions if i.split('-')[1]=='0']
        
        # 随机抽取一个正样本
        target_pos=random.choice(pos_cands)
        # 负样本采样
        if len(neg_cands)>self.neg_num:
            target_negs=random.sample(neg_cands,self.neg_num)
        else:
            # 不够则重复采样
            target_negs=random.choices(neg_cands,k=self.neg_num) if neg_cands else ['<PAD_NID>'] * self.neg_num
        
        # 训练用的候选集为1个正样本+K个负样本
        cand_nids=[target_pos]+target_negs
        # 转换
        cand_docs = [self.news_dict.get(nid, self.news_dict['<PAD>']) for nid in cand_nids]

        label=0
        return (
            torch.LongTensor(click_docs), # [max_hist_len,max_len] 
            torch.LongTensor(cand_docs), # 【num_hand,max_len】
            torch.tensor(label),
        )

class ValidDataset(Dataset):
    def __init__(self,news_file,behaviors_file,w2v_file,max_len,max_hist_len):
        self.news_dict = processNews(news_file,w2v_file,max_len)

        self.behaviors=pl.read_parquet(behaviors_file)
        self.max_len=max_len
        self.max_hist_len=max_hist_len

        with open(w2v_file,'rb') as f:
            self.w2id=pickle.load(f)['w2id']

    def __getitem__(self,idx):
        row = self.behaviors.row(idx, named=True)

        history = row['history']
        # 用户点击长度只保留最近的
        if len(history) > self.max_hist_len:
            history = history[-self.max_hist_len:]

        # 转换
        click_docs = [self.news_dict.get(nid, self.news_dict['<PAD>']) for nid in history]
        # 补齐历史长度
        pad_vec = self.news_dict['<PAD>']
        while len(click_docs) < self.max_hist_len:
            click_docs.append(pad_vec)

        # 无需采样
        impressions=row['impressions']
        cand_nids=[i.split('-')[0] for i in impressions]
        labels=[int(i.split('-')[1]) for i in impressions]

        cand_docs=[self.news_dict[nid] for nid in cand_nids]
        
        return (
            torch.LongTensor(click_docs), torch.LongTensor(cand_docs),torch.FloatTensor(labels),
        )
    def __len__(self):
        return len(self.behaviors)