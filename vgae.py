import numpy as np
import sys
import pickle as pkl
import networkx as nx
import torch
print(torch.__version__)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from scipy.sparse import csr_matrix, dia_matrix, lil_matrix, eye, vstack, isspmatrix_coo, coo_matrix, diags, triu
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class VGAE(nn.Module):
    def __init__(self, adj):
        super(VGAE,self).__init__()
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x:x)
        self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, adj, activation=lambda x:x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), hidden2_dim)
        sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred

    
class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim) 
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

    