import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .components.layers import MLPLayers
from .components.rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
        ):
        super(RQVAE, self).__init__()

        # RQ
        self.in_dim = in_dim # 物品embedding的维度
        self.num_emb_list = num_emb_list # 每层码本中向量的个数
        self.e_dim = e_dim #　码本向量的纬度

        # MLP
        self.layers = layers # mlp中间层的纬度
        self.dropout_prob = dropout_prob
        self.bn = bn #　是否层归一化
        self.loss_type = loss_type # 损失函数
        self.quant_loss_weight=quant_loss_weight # 量化损失在总损失中的权重：
        self.beta = beta #　的权重。
        self.kmeans_init = kmeans_init # 是否用k-means初始化码本
        self.kmeans_iters = kmeans_iters # k-means的迭代次数
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        # 编码器
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        # 量化器
        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)
        # 解码器
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x,use_sk=use_sk) #量化后的结果，损失，索引
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    # 计算总损失，重建损失+量化损失
    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon