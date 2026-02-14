import torch
import torch.nn.functional as F
import torch.nn as nn

class AdditiveAttention(nn.Module):
    # 加性注意力池化，将N个词向量聚合为一个代表整个文章的向量
    def __init__(self,in_dim,v_size):
        super(AdditiveAttention,self).__init__()

        self.in_dim=in_dim
        self.v_size=v_size

        self.proj=nn.Sequential(nn.Linear(self.in_dim,self.v_size),nn.Tanh())
        self.proj_v=nn.Linear(self.v_size,1)
    def forward(self,context,mask=None):
        """
        加性注意力机制
        :param context: [batch_size,seq_len,in_dim]
        :return: outputs [batch_size，seq_len,out_dim],weights[batch_size,seq_len]
        """

        # proj->[B,seq_len,v_size], proj_v->[B,seq_len,1]
        # 因为下一步要用softmax打分得到每个token的权重，需要去除最后一维度
        weights=self.proj_v(self.proj(context)).squeeze(-1)
        # mask掩码，因为会对标题和点击序列填充，所以需要将这些填充位置的值设置为极小的负数，这样就不会影响到softmax的输出，这里没有选择-inf是因为要搭配半精度加速
        if mask is not None:
            weights=weights.masked_fill(mask,-65500.0)
        weights=torch.softmax(weights,dim=-1) # [B,seq_len]

        # bmm批量矩阵乘法，要求两个输入都必须是3D张量，且第一维相等
        # unsqueeze升维 weights->[B,1,seq_len]，最终得到[B,seq_len]
        return torch.bmm(weights.unsqueeze(1),context).squeeze(1),weights


class TextEncoder(nn.Module):
    def __init__(self, hparams, pretrained_vectors=None):
        super(TextEncoder, self).__init__()
        # 使用预训练好的新闻矩阵
        if pretrained_vectors is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_vectors,
                freeze=True, # 通常建议冻结，因为 NLP 模型产出的语义已经很稳健
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(hparams['item_num'], hparams['embed_dim'], padding_idx=0)

        # 投影层：将 NLP 向量维度（如 768）转为模型的 encoder_size（如 256）
        self.input_projection = nn.Linear(hparams['embed_dim'], hparams['encoder_size'])

    def forward(self, news_ids):
        """
        :param news_ids: [Batch, num_docs] 形状的 ID 张量
        :return: [Batch, num_docs, encoder_size]
        """
        # 1. 查表获取向量 [Batch, num_docs, 768]
        news_vecs = self.embedding(news_ids)
        # 2. 线性转换 [Batch, num_docs, encoder_size]
        return self.input_projection(news_vecs)