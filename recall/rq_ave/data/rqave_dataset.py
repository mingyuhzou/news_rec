import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
class EmbDataset(Dataset):

    def __init__(self, data_path,**kwargs):
        self.data_path = data_path
        self.embedding=pd.read_parquet(data_path)['embedding'].values # array([array([...]), array([...]), ...], dtype=object)
        self.embedding =np.stack(self.embedding,axis=0) # [D,] ->[N,D]
        self.dim = self.embedding.shape[-1]
    def __getitem__(self, index):
        emb=self.embedding[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb
    def __len__(self):
        return len(self.embedding)
