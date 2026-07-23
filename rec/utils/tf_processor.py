import pandas as pd
import numpy as np

def process_data(file_path,item2code,max_len,code_len, PAD_TOKEN=0):

    data=pd.read_parquet(file_path)

    processed_data = []
    for row in data.itertuples(index=False):
        sequence = sum(
            [item2code[int(x)] for x in row.history[-max_len:]],
            []
        ) # 拼接token

        target = item2code[int(row.target)]

        processed_data.append({
            'history': sequence+[PAD_TOKEN] * (max_len*code_len - len(sequence)) if len(sequence) < max_len * code_len else sequence,
            'target': target+[1]
        })
    return processed_data

def process_data_qwen(
    file_path,
    item2code,
    max_len,
    code_len,
    EOS_TOKEN=1
):

    data = pd.read_parquet(file_path)

    processed_data = []


    for row in data.itertuples(index=False):

        # history SID
        history = sum(
            [
                item2code[int(x)]
                for x in row.history[-max_len:]
            ],
            []
        )


        # target SID
        target = item2code[int(row.target)]

        target = target + [EOS_TOKEN]


        # ===== training =====
        input_ids = history + target

        labels = (
            [-100] * len(history)
            +
            target
        )


        # ===== generation =====
        generate_input_ids = history


        processed_data.append({

            # train
            "input_ids": input_ids,
            "labels": labels,

            # inference
            "generate_input_ids": generate_input_ids

        })


    return processed_data

def item2code(code_path, codebook_size=256):
    """
    code: [c1, c2, c3, c4]
    offset 后:
    [c1 + 2, c2 + 256 + 2, c3 + 512 + 2, c4 + 768 + 2]

    0: PAD
    1: EOS
    2 开始才是真实 code token
    """
    data = np.load(code_path, allow_pickle=True)

    item_to_code = {}
    code_to_item = {}

    for index, code in enumerate(data):
        offsets = [
            int(c) + i * codebook_size + 2
            for i, c in enumerate(code)
        ]

        item_id = index + 1

        item_to_code[item_id] = offsets
        code_to_item[tuple(offsets)] = item_id

    return item_to_code, code_to_item