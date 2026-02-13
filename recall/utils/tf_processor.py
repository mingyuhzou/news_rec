import pandas as pd
import numpy as np


def process_data(filt_path,mode,max_len, PAD_TOKEN=0):

    data=pd.read_parquet(filt_path)
    data['sequence']=data['history'].apply(lambda x:list(x))+data['target'].apply(lambda x:[x])
    if mode=='train':
        processed_data=[]
        # 滑动窗口划分
        for row in data.itertuples(index=False):
            sequence=row.sequence
            for i in range(1,len(sequence)):
                processed_data.append({
                    'history': sequence[:i],
                    'target': sequence[i]
                })
    elif mode == 'evaluation':
        # Use the last item as target and the rest as history
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            processed_data.append({
                'history': sequence[:-1],
                'target': sequence[-1]
            })
    else:
        raise ValueError("Mode must be 'train' or 'evaluation'.")

    for item in processed_data:
        item['history']=pad_or_truncate(item['history'],max_len)
    return processed_data


def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    if len(sequence) > max_len:
        # 截断
        return sequence[-max_len:]
    else:
        # 用0填充
        return [PAD_TOKEN] * (max_len - len(sequence)) + sequence


def item2code(code_path, codebook_size=256):
    """
    构建物品索引到code的映射
    """
    data = np.load(code_path, allow_pickle=True)
    item_to_code = {}
    code_to_item = {}

    for index, code in enumerate(data):
        offsets = [c + i * codebook_size + 1 for i, c in enumerate(code)]
        # 0空出来
        item_to_code[index + 1] = offsets
        code_to_item[tuple(offsets)] = index + 1

    return item_to_code, code_to_item