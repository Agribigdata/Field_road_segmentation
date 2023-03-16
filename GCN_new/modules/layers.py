import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.left_gcn_layers = nn.ModuleList(
            [nn.Linear((input_size if i == 0 else hidden_size), hidden_size, bias=False) \
            for i in range(num_layers)])
        self.right_gcn_layers = nn.ModuleList(
            [nn.Linear((input_size if i == 0 else hidden_size), hidden_size, bias=False) \
            for i in range(num_layers)])
        self.up_gcn_layers = nn.ModuleList(
            [nn.Linear((input_size if i == 0 else hidden_size), hidden_size, bias=False) \
             for i in range(num_layers)])
        self.down_gcn_layers = nn.ModuleList(
            [nn.Linear((input_size if i == 0 else hidden_size), hidden_size, bias=False) \
             for i in range(num_layers)])
        self.another1_gcn_layers = nn.ModuleList(
            [nn.Linear((input_size if i == 0 else hidden_size), hidden_size, bias=False) \
             for i in range(num_layers)])
        self.another2_gcn_layers = nn.ModuleList(
            [nn.Linear((input_size if i == 0 else hidden_size), hidden_size, bias=False) \
             for i in range(num_layers)])

        self.self_loof_layers = nn.ModuleList(
            [nn.Linear((input_size if i == 0 else hidden_size), hidden_size, bias=False) \
            for i in range(num_layers)])
        self.dropout = nn.Dropout(0.3)
    def forward(self, adj, x):
        #nadj = adj[:,:]
        first_neigh = (adj == 0.1).float()
        second_neigh = (adj == 0.2).float()
        third_neigh= (adj == 0.3).float()
        four_neigh= (adj == 0.4).float()
        forth_neigh = (adj == 0.5).float()
        sixth_neigh = (adj == 0.6).float()
        denom =   first_neigh.sum(-1, keepdim=True)  \
                + second_neigh.sum(-1, keepdim=True)\
                + third_neigh.sum(-1, keepdim=True)\
                + four_neigh.sum(-1, keepdim=True)\
                + forth_neigh.sum(-1, keepdim=True)\
                +sixth_neigh.sum(-1, keepdim=True)+1

        for l in range(self.num_layers):
            self_node = self.self_loof_layers[l](x)
            left_neigh_Ax = self.left_gcn_layers[l](torch.einsum('kl, lz -> kz', first_neigh, x))
            right_neigh_Ax = self.right_gcn_layers[l](torch.einsum('kl, lz -> kz', second_neigh, x))
            up_neigh_Ax = self.up_gcn_layers[l](torch.einsum('kl, lz -> kz', third_neigh, x))
            down_neigh_Ax = self.down_gcn_layers[l](torch.einsum('kl, lz -> kz', four_neigh, x))
            another1_neigh_Ax= self.another1_gcn_layers[l](torch.einsum('kl, lz -> kz', forth_neigh, x))
            another2_neigh_Ax= self.another2_gcn_layers[l](torch.einsum('kl, lz -> kz', sixth_neigh, x))
            if l != self.num_layers - 1:
                AxW = (self_node + left_neigh_Ax + right_neigh_Ax + up_neigh_Ax + down_neigh_Ax +another1_neigh_Ax+another2_neigh_Ax) / denom

            else:
                AxW = self_node + left_neigh_Ax + right_neigh_Ax + up_neigh_Ax + down_neigh_Ax +another1_neigh_Ax+another2_neigh_Ax
            gAxWb = torch.relu(AxW)
            x = self.dropout(gAxWb) if l < self.num_layers - 1 else gAxWb
        return x

