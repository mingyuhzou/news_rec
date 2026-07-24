import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from rec.utils.tf_processor import item2code


class QwenDPODataset(Dataset):

    def __init__(
            self,
            dataset_path,
            code_path,
            max_len,
            pairs_per_click=1,
            seed=2025,
            eos_token=1
    ):
        self.item2code, _ = item2code(code_path)
        data = pd.read_parquet(dataset_path)

        required = {"history", "click", "unclick"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(
                f"DPO dataset is missing columns: {sorted(missing)}. "
                "Run preprocessData.py first."
            )

        rng = random.Random(seed)
        self.samples = []

        for row in data.itertuples(index=False):
            history_items = [
                int(item)
                for item in row.history[-max_len:]
                if int(item) in self.item2code
            ]
            clicked = [
                int(item)
                for item in row.click
                if int(item) in self.item2code
            ]
            unclicked = [
                int(item)
                for item in row.unclick
                if int(item) in self.item2code
            ]

            if not history_items or not clicked or not unclicked:
                continue

            history = sum(
                (self.item2code[item] for item in history_items),
                []
            )

            for chosen_item in clicked:
                for _ in range(pairs_per_click):
                    rejected_item = rng.choice(unclicked)
                    self.samples.append({
                        "history": history,
                        "chosen": (
                            self.item2code[chosen_item] + [eos_token]
                        ),
                        "rejected": (
                            self.item2code[rejected_item] + [eos_token]
                        )
                    })

        if not self.samples:
            raise ValueError(
                "No DPO pairs found. Every preference sample needs "
                "history, click and unclick items in the SID codebook."
            )

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

class QwenDPODataLoader(DataLoader):

    def __init__(
            self,
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pad_token=0
    ):
        self.pad_token = pad_token
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        max_len = max(
            len(item["history"]) + max(
                len(item["chosen"]),
                len(item["rejected"])
            )
            for item in batch
        )

        def left_pad(history, response):
            ids = history + response
            labels = [-100] * len(history) + response
            padding_len = max_len - len(ids)
            return (
                [self.pad_token] * padding_len + ids,
                [0] * padding_len + [1] * len(ids),
                [-100] * padding_len + labels
            )

        result = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "chosen_labels": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "rejected_labels": []
        }

        for item in batch:
            chosen = left_pad(item["history"], item["chosen"])
            rejected = left_pad(item["history"], item["rejected"])

            result["chosen_input_ids"].append(chosen[0])
            result["chosen_attention_mask"].append(chosen[1])
            result["chosen_labels"].append(chosen[2])
            result["rejected_input_ids"].append(rejected[0])
            result["rejected_attention_mask"].append(rejected[1])
            result["rejected_labels"].append(rejected[2])

        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in result.items()
        }