import dgl.nn.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from model.layer import *
from model.GCN import GraphConvolution
from torch_geometric.nn import conv

device = torch.device('cuda')


class GMAMDA(nn.Module):
    def __init__(self, args):
        super(GMAMDA, self).__init__()
        self.args = args

        self.mlp = nn.Sequential(
            nn.Linear(args.nhid, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

        #  -----------多重卷积--------------
        self.dropout_ratio = 0.2
        self.nhid = args.nhid
        self.kernels = 2
        self.alpha = 0.9
        self.beta1 = 0.3
        self.beta2 = 0.35
        self.beta3 = 0.35
        self.node_encoder = torch.nn.Linear(2249, self.nhid)
        self.node_encoder2 = torch.nn.Linear(2249, self.nhid)
        self.gconv1 = MKGC(self.kernels, 2249, self.nhid)
        # self.gconv1 = MKGC(self.kernels, self.nhid, self.nhid)
        self.gconv2 = MKGC(self.kernels, self.nhid, self.nhid)
        self.gconv3 = MKGC(self.kernels, self.nhid, self.nhid)
        self.weight = GLAPool(self.nhid, self.alpha)

    def reset_parameters(self):
        self.node_encoder1.reset_parameters()
        self.node_encoder2.reset_parameters()
        self.gconv1.reset_parameters()
        self.gconv2.reset_parameters()
        self.gconv3.reset_parameters()

    @staticmethod
    def to_tensor(data, dtype=torch.float32):
        """统一的张量转换函数"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(dtype)
        elif isinstance(data, torch.Tensor):
            return data.clone().detach().to(dtype)
        else:
            return torch.tensor(data, dtype=dtype)

    def node_feature(self, adj, adj_edge_index):

        adj = self.to_tensor(adj, dtype=torch.float32).to(device)
        adj_edge_index = adj_edge_index.to(device)
        # x = self.node_encoder(adj)
        x = F.dropout(adj, p=self.dropout_ratio, training=self.training)
        x1 = self.gconv1(x, adj_edge_index)
        x1 = F.dropout(x1, p=self.dropout_ratio, training=self.training)
        x2 = self.gconv2(x1, adj_edge_index)
        x2 = F.dropout(x2, p=self.dropout_ratio, training=self.training)
        x3 = self.gconv3(x2, adj_edge_index)

        weight = torch.cat((self.weight(x1, adj_edge_index, None, None, 1),
                            self.weight(x2, adj_edge_index, None, None, 1),
                            self.weight(x3, adj_edge_index, None, None, 1)), dim=-1)
        weight = torch.softmax(weight, dim=-1)
        node_feature = weight[:, 0].view(-1, 1) * x1 + weight[:, 1].view(-1, 1) * x2 + weight[:, 2].view(-1, 1) * x3
        return node_feature

    def forward(self, sample, adj, adj_edge_index1, adj_edge_index2, adj_edge_index3):

        node_feature1 = self.node_feature(adj, adj_edge_index1)
        node_feature2 = self.node_feature(adj, adj_edge_index2)
        node_feature3 = self.node_feature(adj, adj_edge_index3)
        node_feature = self.beta1 * node_feature1 + self.beta2 * node_feature2 + self.beta3 * node_feature3
        dr_gcn = node_feature[:2033]
        di_gcn = node_feature[2033:]
        dr = dr_gcn
        di = di_gcn

        drdi_embedding = torch.mul(dr[sample[:, 0]], di[sample[:, 1]])

        output = self.mlp(drdi_embedding)

        return drdi_embedding, output
