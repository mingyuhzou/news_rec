from recall.utils.qwen_processor import process_data_qwen


class QwenGenRecDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        code_path,
        max_len,
        code_len
    ):

        self.item2code,self.code2item=item2code(code_path)

        self.data=process_data_qwen(
            dataset_path,
            self.item2code,
            max_len,
            code_len
        )

        self.PAD_TOKEN=0


    def __getitem__(self,idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)

class QwenGenRecDataLoader(DataLoader):

    def __init__(
        self,
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    ):

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )


    def collate_fn(self, batch):

        pad_token = self.dataset.PAD_TOKEN


        input_ids = [
            item["input_ids"]
            for item in batch
        ]

        labels = [
            item["labels"]
            for item in batch
        ]


        max_len = max(
            len(x)
            for x in input_ids
        )


        batch_input_ids = []
        batch_labels = []
        batch_attention = []


        for x, y in zip(input_ids, labels):

            padding_len = max_len - len(x)


            batch_input_ids.append(
                x + [pad_token] * padding_len
            )


            batch_labels.append(
                y + [-100] * padding_len
            )


            batch_attention.append(
                [1] * len(x)
                +
                [0] * padding_len
            )


        return {

            "input_ids": torch.tensor(
                batch_input_ids,
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                batch_attention,
                dtype=torch.long
            ),

            "labels": torch.tensor(
                batch_labels,
                dtype=torch.long
            )

        }