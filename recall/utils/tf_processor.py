import pandas as pd
import numpy as np

def process_data(filt_path,mode,max_len, PAD_TOKEN=0):

    data=pd.read_parquet(filt_path)
    if mode!='recall':
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
            target = sequence[-1]
            if isinstance(target, (list, np.ndarray)):
                target = target[-1]

            processed_data.append({
                'history': sequence[:-1],
                'target': target
            })
    elif mode == 'recall':
        processed_data = []
        for row in data.itertuples(index=False):
            processed_data.append({
                'history': list(row.history),
                'target':row.target
            })
    else:
        raise ValueError("Mode must be 'train' or 'evaluation'.")

    for item in processed_data:
        item['ori_history']=item['history']
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
    构建物品索引到code的映射，从[c1, c2, c3, c4】到[c1+1, c2+256+1, c3+512+1, c4+768+1] 第一：让模型明确知道当前生成的 code 是属于编码序列的第几位。如果不加偏移，模型在生成时可能会混淆第一层和第二层的语义信息
    第二：处理“未登录/填充”物品
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