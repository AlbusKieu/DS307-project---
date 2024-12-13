
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)

        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        n = h.size(0) # h is of size n x f_in
        h_prime = torch.matmul(h.unsqueeze(0), self.w) #  n_head x n x f_out
        attn_src = torch.bmm(h_prime, self.a_src) # n_head x n x 1
        attn_dst = torch.bmm(h_prime, self.a_dst) # n_head x n x 1
        attn = attn_src.expand(-1, -1, n) + attn_dst.expand(-1, -1, n).permute(0, 2, 1) # n_head x n x n

        attn = self.leaky_relu(attn)
        attn.data.masked_fill_(1 - adj, float("-inf"))
        attn = self.softmax(attn) # n_head x n x n
        attn = self.dropout(attn)
        output = torch.bmm(attn, h_prime) # n_head x n x f_out

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2] # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w) # bs x n_head x n x f_out
        attn_src = torch.matmul(F.tanh(h_prime), self.a_src) # bs x n_head x n x 1
        attn_dst = torch.matmul(F.tanh(h_prime), self.a_dst) # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2) # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = 1 - adj.unsqueeze(1) # bs x 1 x n x n
        attn.data.masked_fill_(mask.bool(), float("-inf"))
        attn = self.softmax(attn) # bs x n_head x n x n
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime) # bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class BatchGAT(nn.Module):
    def __init__(self, pretrained_emb, vertex_feature, use_vertex_feature,
            n_units=[1433, 8, 7], n_heads=[8, 1],
            dropout=0.1, attn_dropout=0.0, fine_tune=False,
            instance_normalization=False):
        super(BatchGAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)

        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/2
        self.embedding = nn.Embedding(pretrained_emb.size(0), pretrained_emb.size(1))
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.requires_grad = fine_tune
        n_units[0] += pretrained_emb.size(1)

        self.use_vertex_feature = use_vertex_feature
        if self.use_vertex_feature:
            self.vertex_feature = nn.Embedding(vertex_feature.size(0), vertex_feature.size(1))
            self.vertex_feature.weight = nn.Parameter(vertex_feature)
            self.vertex_feature.weight.requires_grad = False
            n_units[0] += vertex_feature.size(1)

        self.layer_stack = nn.ModuleList()
        for i in range(self.n_layer):
            # consider multi head from last layer
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                    BatchMultiHeadGraphAttention(n_heads[i], f_in=f_in,
                        f_out=n_units[i + 1], attn_dropout=attn_dropout)
                    )

    def forward(self, x, vertices, adj):
        emb = self.embedding(vertices)
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((x, emb), dim=2)
        if self.use_vertex_feature:
            vfeature = self.vertex_feature(vertices)
            x = torch.cat((x, vfeature), dim=2)
        bs, n = adj.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = gat_layer(x, adj) # bs x n_head x n x f_out
            if i + 1 == self.n_layer:
                x = x.mean(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)
