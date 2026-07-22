# qwen_processor.py

import pandas as pd


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


        # Qwen causal LM格式
        input_ids = history + target


        labels = (
            [-100] * len(history)
            +
            target
        )


        processed_data.append({

            "input_ids": input_ids,

            "labels": labels

        })


    return processed_data