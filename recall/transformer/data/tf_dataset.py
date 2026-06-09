from torch.utils.data import Dataset,DataLoader
from recall.utils.tf_processor import process_data,item2code
import numpy as np
import torch

class GenRecDataset(Dataset):
    def __init__(self,dataset_path,code_path,max_len,code_len,PAD_TOKEN=0):
        self.data_path = dataset_path
        self.code_path = code_path
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.code_len = code_len

        self.item2code,self.code2item=item2code(code_path)
        self.data=self._prepare_data()

    def _prepare_data(self):
        processed_data=process_data(self.data_path,self.item2code,self.max_len,self.code_len,self.PAD_TOKEN)
        return processed_data

    def __getitem__(self,idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

class GenRecDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collat_fn=None
    ):
        collate_fn = collat_fn if collat_fn is not None else self.collate_fn

        super(GenRecDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

    def collate_fn(self, batch):
        pad_token = self.dataset.PAD_TOKEN

        max_token_len = self.dataset.max_len * self.dataset.code_len

        histories = []
        for item in batch:
            h = item["history"]

            # 兼容两种情况：
            # 1. h = [1, 2, 3, 4]
            # 2. h = [[1, 2], [3, 4]]
            if len(h) > 0 and isinstance(h[0], list):
                h = [x for sub in h for x in sub]

            h = h[:max_token_len]

            if len(h) < max_token_len:
                h = h + [pad_token] * (max_token_len - len(h))

            histories.append(h)

        histories = torch.tensor(histories, dtype=torch.long)

        targets = torch.tensor(
            [item["target"] for item in batch],
            dtype=torch.long
        )

        attention_masks = (histories != pad_token).long()

        ori_history = [
            item.get("ori_history", None)
            for item in batch
        ]

        ret = {
            "history": histories,
            "target": targets,
            "attention_masks": attention_masks,
        }

        # 如果数据里有 ori_history，就返回；没有就不返回也可以
        if any(x is not None for x in ori_history):
            ret["ori_history"] = ori_history

        return ret

