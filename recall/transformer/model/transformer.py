import torch
from transformers import T5ForConditionalGeneration,T5Config
import torch.nn as nn

class TIGER(nn.Module):
    def __init__(self,config):
        super(TIGER, self).__init__()
        t5config=T5Config(
            num_layers=config['num_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            vocab_size=config['vocab_size'],
            pad_token_id=config['pad_token_id'],
            eos_token_id=config['eos_token_id'],
            decoder_start_token_id=config['pad_token_id'],
            feed_forward_proj=config['feed_forward_proj'],
        )
        self.model=T5ForConditionalGeneration(t5config)

    @property
    def n_parameters(self):
        '''统计并格式化输出模型的参数量'''

        # numel计算单个张量中的参数个数
        num_params=lambda ps:sum(p.numel() for p in ps if  p.requires_grad)
        total_params=num_params(self.parameters()) # 总训练参数

        emb_params=num_params(self.model.get_input_embeddings().parameters()) # 嵌入层参数
        return (
            f'#Embedding parameters: {emb_params}\n'
            f'#Non-embedding parameters: {total_params - emb_params}\n'
            f'#Total trainable parameters: {total_params}\n'
        )

    def forward(self,input_ids,attention_mask=None,labels=None):
        outputs=self.model(input_ids,attention_mask=attention_mask,labels=labels)
        return outputs.loss,outputs.logits

    def generate(self,input_ids,attention_mask=None,labels=None,num_beams=20):
        # 使用beam search生成预测物品的编码，输入维度为[B,history_len*code_num]，在dataloader中展平过
        # 返回[B*num_return_sequences(候选物品数),max_len] max_len的第一位是decoder_start_token_id被设置为了0
        return self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=5, #
            num_beams=num_beams,
            num_return_sequences=num_beams,
        )