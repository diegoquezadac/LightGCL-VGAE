import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.spmm(x, self.weight)
        x = torch.spmm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def GRDPG_decode(Z1, Z2, q=0):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    I_pq = torch.eye(Z1.shape[1], device=device)
    if q > 0:
        I_pq[:, -q:] = -I_pq[:, -q:]
    A_pred = torch.sigmoid((Z1) @ I_pq @ Z2.T)
    return A_pred


def glorot_init(input_dim, output_dim):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim, device=device) * 2 * init_range - init_range
    return nn.Parameter(initial)


class VBGAE(nn.Module):
    def __init__(self, adj, GRDPG=0):
        super(VBGAE, self).__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # User matrix
        self.input_dim1 = adj.size()[0]
        self.hidden1_dim1 = 16
        self.hidden2_dim1 = 4

        # Item matrix
        self.input_dim2 = adj.size()[1]
        self.hidden1_dim2 = 16
        self.hidden2_dim2 = 4

        self.base_gcn1 = GraphConvSparse(self.input_dim1, self.hidden1_dim1, adj.t())
        self.gcn_mean1 = GraphConvSparse(
            self.hidden1_dim1, self.hidden2_dim1, adj, activation=lambda x: x
        )
        self.gcn_logstddev1 = GraphConvSparse(
            self.hidden1_dim1, self.hidden2_dim1, adj, activation=lambda x: x
        )

        self.base_gcn2 = GraphConvSparse(self.input_dim2, self.hidden1_dim2, adj)
        self.gcn_mean2 = GraphConvSparse(
            self.hidden1_dim2, self.hidden2_dim2, adj.t(), activation=lambda x: x
        )
        self.gcn_logstddev2 = GraphConvSparse(
            self.hidden1_dim2, self.hidden2_dim2, adj.t(), activation=lambda x: x
        )
        self.GRDPG = GRDPG

    def encode1(self, X1):
        hidden1 = self.base_gcn1(X1)
        self.mean1 = self.gcn_mean1(hidden1)
        self.logstd1 = self.gcn_logstddev1(hidden1)
        gaussian_noise1 = torch.randn(X1.size(0), self.hidden2_dim1, device=self.device)

        sampled_z1 = gaussian_noise1 * torch.exp(self.logstd1) + self.mean1
        return sampled_z1

    def encode2(self, X2):
        hidden2 = self.base_gcn2(X2)
        self.mean2 = self.gcn_mean2(hidden2)
        self.logstd2 = self.gcn_logstddev2(hidden2)
        gaussian_noise2 = torch.randn(X2.size(0), self.hidden2_dim2, device=self.device)
        sampled_z2 = gaussian_noise2 * torch.exp(self.logstd2) + self.mean2
        return sampled_z2

    def forward(self, X1, X2):
        Z1 = self.encode1(X1)
        Z2 = self.encode2(X2)
        A_pred = GRDPG_decode(Z1, Z2, self.GRDPG)
        return A_pred, Z1, Z2
