import torch
import torch.nn as nn
from rank.model.TextEncoder import TextEncoder,AdditiveAttention
import torch.nn.functional as F

# class NRMS(nn.Module):
#     def __init__(self,hparams,weight=None):
#         super(NRMS, self).__init__()
#         self.hparams = hparams
#         self.doc_encoder=TextEncoder(hparams,weight=weight)
#         self.mha=nn.MultiheadAttention(hparams['encoder_size'],hparams['nhead'],dropout=0.1,batch_first=True)
#         self.additive_attn=AdditiveAttention(hparams['encoder_size'],hparams['v_size'])
#         self.criterion=nn.CrossEntropyLoss()
#     def forward(self,clicks,cands,labels=None):
#         """
#         :param clicks: [num_user,num_click_docs(历史长度),seq_len] token已经被转化id了
#         :param cands:  [num_user,num_cand_docs(候选数量),seq_len]
#         :param labels:
#         :return:
#         """
#         num_click_docs=clicks.shape[1]
#         num_cand_docs=cands.shape[1]
#         num_user=clicks.shape[0]
#         seq_len=clicks.shape[2]
#
#         click_mask = (clicks.sum(dim=-1) == 0)
#         # 防止全被遮蔽
#         click_mask[click_mask.all(dim=1), 0] = False
#
#         # 把所有新闻看作一个大批次，DocEncoder只能处理二维输入
#         clicks=clicks.reshape(-1,seq_len) # 【num_user*num_click_docs,seq_len】
#         cands=cands.reshape(-1,seq_len) # [num_user*num_cand_docs,seq_len]
#
#         click_emb=self.doc_encoder(clicks) # [num_user*num_click_docs,encoder_size】
#         cand_emb=self.doc_encoder(cands) # [num_user*num_cand_docs,encoder_size]
#
#         # 转换回来
#         click_emb=click_emb.reshape(num_user,num_click_docs,-1) #【num_user,num_click_docs,encoder_size】
#         cand_emb=cand_emb.reshape(num_user,num_cand_docs,-1) # [num_user,num_cand_docs,encoder_size]
#
#         # 对用户兴趣建模
#         click_output,_=self.mha(click_emb,click_emb,click_emb,key_padding_mask=click_mask) # [num_user,num_clicks_docs,encoder_size]
#         click_output=F.dropout(click_output,p=0.2,training=self.training)
#         click_repr,_=self.additive_attn(click_output,mask=click_mask) # [num_user,encoder_size]
#
#         # 点击预测
#         logits=torch.bmm(click_repr.unsqueeze(1),cand_emb.permute(0,2,1)).squeeze(1) # [num_user,1,encoder_size] * [num_user,encoder_size,num_cand_docs] ->[nums_user,num_cand_docs]
#         if labels is not None:
#             loss=self.criterion(logits,labels)
#             return loss,logits
#         return logits


class NRMS(nn.Module):
    def __init__(self, hparams, weight=None):
        super(NRMS, self).__init__()
        # 使用你新定义的、基于查表的 TextEncoder
        self.doc_encoder = TextEncoder(hparams, pretrained_vectors=weight)
        self.mha = nn.MultiheadAttention(hparams['encoder_size'], hparams['nhead'], dropout=0.1, batch_first=True)
        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, clicks, cands, labels=None):
        """
        :param clicks: [batch, max_hist_len] 只有新闻ID
        :param cands:  [batch, 1 + neg_num] 只有新闻ID
        """
        # 1. 直接通过 doc_encoder 获取新闻表示 (不需要 reshape seq_len)
        click_emb = self.doc_encoder(clicks)  # [batch, max_hist_len, encoder_size]
        cand_emb = self.doc_encoder(cands)  # [batch, 1+neg_num, encoder_size]

        # 2. 构造 Mask (ID 为 0 的地方即为 padding)
        click_mask = (clicks == 0)
        click_mask[click_mask.all(dim=1), 0] = False

        # 3. 用户兴趣建模 (逻辑保持不变)
        click_output, _ = self.mha(click_emb, click_emb, click_emb, key_padding_mask=click_mask)
        click_repr, _ = self.additive_attn(click_output, mask=click_mask)

        # 4. 点击预测
        logits = torch.bmm(click_repr.unsqueeze(1), cand_emb.permute(0, 2, 1)).squeeze(1)

        if labels is not None:
            return self.criterion(logits, labels), logits
        return logits