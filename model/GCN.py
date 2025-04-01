import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

class GraphConvolution(Module):

    # in_features 878， out 56
    def __init__(self, in_features, out_features, stdv, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  #
        # print(self.weight)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  #
        else:
            self.register_parameter('bias', None)
        # 随机初始化参数
        self.reset_parameters(stdv)

    def reset_parameters(self, stdv):

        # print("stdv:", stdv)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, adj):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(input.shape)
        try:
            input1 = input1.float()
            adj = adj.to(device)
        except:
            pass

        support = torch.mm(input1.to(device), self.weight)

        output = torch.mm(adj.to(torch.double), support.to(torch.double))
        if self.bias is not None:
            return output + self.bias
        else:
            return output