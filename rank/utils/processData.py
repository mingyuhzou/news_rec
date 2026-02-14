# %%
import numpy as np
import polars as pl
import pickle
import re

import torch

from config.model.nrms import hparams

def processW2vec(news_file,W2vec_file,save_path):
    """
    从词表中提取本地数据中单词的向量，并构建一个token->id的字典
    """
    news=pl.read_parquet(news_file)

    vocab=set()
    for row in news.iter_rows(named=True):
        content=row['title'].lower()
        words = re.findall(r'\w+', content)
        for w in words:
            vocab.add(w)

    # 设置两个占位符
    w2id={'<PAD>':0,'<UNK>':1}
    vectors=[]
    # PAD 和 UNK 的初始化
    vectors.append(np.zeros(300))
    vectors.append(np.random.normal(size=300))

    cnt=0
    with open(W2vec_file,'r',encoding='utf-8') as f:
        for line in f:
            parts=line.rstrip().split()
            if len(parts)!=301:
                continue
            word=parts[0].lower()
            if word in vocab:
                if word not in w2id:
                    w2id[word]=len(w2id)
                    vectors.append(np.array(parts[1:],dtype='float32'))
                    cnt+=1
    embeddings=np.stack(vectors)
    with open(save_path,'wb') as f:
        pickle.dump({'w2id':w2id,'embedding':embeddings},f)
    print(f"共有{len(vocab)}个单词，匹配到GloVe词汇: {cnt}, 未匹配到的数量为{len(vocab)-cnt}")

#

def processNews(news_path,w2v_path,max_len=20):
    """
    构建news_id->token_ids的字典
    """
    df=pl.read_parquet(news_path)

    with open(w2v_path,'rb') as f:
        w2id=pickle.load(f)['w2id']
    def tokenize_and_map(title):
        tokens=re.findall(r'\w+',title.lower()) # 只保留单词的字母，并小写
        token_ids=[w2id.get(t,1) for t in tokens] # 找不到则置为占位符

        token_ids=token_ids[:max_len]
        # 补齐
        token_ids+=[0]*(max_len-len(token_ids))
        return token_ids

    news_dict={}
    for row in df.iter_rows(named=True):
        news_dict[row['news_id']]=tokenize_and_map(row['title'])
    news_dict['<PAD>']=[0]*max_len

    return news_dict

def prepare_embedding_matrix(item_emb_path, item_dict_path):
    # 1. 加载数据
    item_emb_df = pl.read_parquet(item_emb_path)
    # 这里的 item_dict 是你在 notebook 中生成的映射表 {news_id: ItemID}
    item_id_mapping = np.load(item_dict_path, allow_pickle=True).item()

    # 2. 确定维度
    max_id = max(item_id_mapping.values())
    emb_dim = len(item_emb_df['embedding'][0])

    # 3. 初始化矩阵 (Index 0 留给 <PAD>)
    matrix = np.zeros((max_id + 1, emb_dim), dtype='float32')

    # 4. 填充矩阵
    for row in item_emb_df.iter_rows(named=True):
        idx = row['ItemID']
        matrix[idx] = np.array(row['embedding'])

    return torch.from_numpy(matrix)