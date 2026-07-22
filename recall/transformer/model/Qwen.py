import torch
import torch.nn as nn

from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM
)


class Qwen(nn.Module):

    def __init__(self, config):

        super().__init__()


        qwen_config = Qwen2Config(
            vocab_size=config["vocab_size"],

            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],

            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],

            max_position_embeddings=config["max_position_embeddings"],

            pad_token_id=config["pad_token_id"],
            eos_token_id=config["eos_token_id"],

        )


        self.code_len=config["code_len"]


        self.model=Qwen2ForCausalLM(
            qwen_config
        )


    @property
    def n_parameters(self):

        total=sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        embedding=sum(
            p.numel()
            for p in self.model.get_input_embeddings().parameters()
        )


        return (
            f"#Embedding parameters: {embedding}\n"
            f"#Non-embedding parameters: {total-embedding}\n"
            f"#Total trainable parameters: {total}\n"
        )


    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None
    ):

        outputs=self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )


        return (
            outputs.loss,
            outputs.logits
        )


    def generate(
        self,
        input_ids,
        attention_mask=None,
        num_beams=20,
        num_return_sequences=20
    ):


        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,

            max_length=
            input_ids.shape[1]+self.code_len+1,
    
            num_beams=num_beams,

            num_return_sequences=num_return_sequences,

            pad_token_id=self.model.config.pad_token_id,

            eos_token_id=self.model.config.eos_token_id
        )