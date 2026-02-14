from torch.utils.data import Dataset,DataLoader
from recall.utils.tf_processor import process_data,pad_or_truncate,item2code
import numpy as np
import torch

class GenRecDataset(Dataset):
    def __init__(self,dataset_path,code_path,mode,max_len,PAD_TOKEN=0):
        self.data_path = dataset_path
        self.code_path = code_path
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.mode = mode

        self.item2code,self.code2item=item2code(code_path)
        self.data=self._prepare_data()

    def _prepare_data(self):
        processed_data=process_data(self.data_path,self.mode,self.max_len,self.PAD_TOKEN)
        # 转换为编码
        for item in processed_data:
            item['history'] = [self.item2code.get(x, np.array([self.PAD_TOKEN] * 4)) for x in item['history']]
            if self.mode!='recall':item['target'] = self.item2code.get(item['target'], np.array([self.PAD_TOKEN] * 4))

        return processed_data

    def __getitem__(self,idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

class GenRecDataLoader(DataLoader):
    def __init__(self,dataset,batch_size=32,shuffle=True,num_workers=4,collat_fn=None):
        collate_fn=self.collate_fn
        super(GenRecDataLoader,self).__init__(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,collate_fn=collate_fn)

    def collate_fn(self,batch,pad_token=0):
        '''
        batch = [
            {
                'history': [[0,0,0,0], [1,257,513,769], [5,261,517,773]], # 长度为 max_len(3) 的序列
                'target': [10, 266, 522, 778] # 目标物品的 code
            },
            {
                'history': [[0,0,0,0], [0,0,0,0], [9,265,521,777]], # 填充了两个 PAD 的序列
                'target': [12, 268, 524, 780]
            }
        ]
        '''

        histories = [item['history'] for item in batch]  # [[],[],[],[]...]

        target=[item['target'] for item in batch]# [[],[],[],[]...]

        ori_history=[item['ori_history'] for item in batch]
        # 将每个样本的code合并为一个列表，拉成一维
        flattened_histories=torch.stack(
            [torch.tensor([elem for sublist in history for elem in sublist],dtype=torch.int64) for history in histories]
        )

        attention_masks=torch.stack(
            [torch.tensor([1 if elem!=pad_token else 0 for elem in h],dtype=torch.int64) for h in flattened_histories]
        )

        flattened_targets = torch.stack(
            [torch.tensor(item['target'], dtype=torch.int64) for item in batch]
        )
        return {
            'history': flattened_histories,
            'target': flattened_targets,
            'attention_masks': attention_masks,
            'ori_history':ori_history,
        }

