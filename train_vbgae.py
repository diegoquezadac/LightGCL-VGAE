import torch
import pickle
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
from parser import args
from utils import scipy_sparse_mat_to_torch_sparse_tensor
from vbgae import VBGAE


def get_scores(val_edges, val_edges_false, A_pred):

    pos_pred = A_pred[val_edges].detach().numpy()
    neg_pred = A_pred[val_edges_false].detach().numpy()

    preds_all = np.hstack([pos_pred, neg_pred])
    labels_all = np.hstack([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):  # adj is a csr matrix
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    colsum = np.array(adj.sum(0))
    D1 = sp.diags(np.power(rowsum + 1, -0.5).flatten())
    D2 = sp.diags(np.power(colsum + 1, -0.5).flatten())
    adj_normalized = D1.dot(adj).dot(D2).tocoo()
    return sparse_to_tuple(adj_normalized)


def mask_test_edges(adj):
    # Function to build test set with 10% positive links

    edges = np.where(adj > 0)
    non_edges = np.where(adj == 0)

    permut_edges = np.random.permutation(edges[0].shape[0])
    edges = edges[0][permut_edges], edges[1][permut_edges]

    permut_non_edges = np.random.permutation(non_edges[0].shape[0])
    non_edges = non_edges[0][permut_non_edges], non_edges[1][permut_non_edges]

    num_test = int(np.floor(edges[0].shape[0] / 10.0))
    num_val = int(np.floor(edges[0].shape[0] / 20.0))

    edges = np.split(edges[0], [num_test, num_test + num_val]), np.split(
        edges[1], [num_test, num_test + num_val]
    )
    non_edges = np.split(non_edges[0], [num_test, num_test + num_val]), np.split(
        non_edges[1], [num_test, num_test + num_val]
    )

    train_edges, val_edges, test_edges = (
        (edges[0][2], edges[1][2]),
        (edges[0][1], edges[1][1]),
        (edges[0][0], edges[1][0]),
    )
    val_edges_false, test_edges_false = (non_edges[0][1], non_edges[1][1]), (
        non_edges[0][0],
        non_edges[1][0],
    )

    data = np.ones(train_edges[0].shape[0])
    adj_train = sp.csr_matrix((data, train_edges), shape=adj.shape)

    return (
        adj_train,
        train_edges,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    )


if __name__ == "__main__":

    device = "cuda:" + args.cuda if torch.cuda.is_available() else "cpu"


    print("Defining matrices...")

    path = "data/" + args.data + "/"
    f = open(path + "trnMat.pkl", "rb")
    train = pickle.load(f)
    train_csr = (train != 0).astype(np.float32)  # adjacency matrix in csr format

    #f = open(path + "tstMat.pkl", "rb")
    #test = pickle.load(f)
    #test_csr = (test != 0).astype(np.float32)  # adjacency matrix in csr format

    norm = train_csr.shape[0] * train_csr.shape[1] / float((train_csr.shape[0] * train_csr.shape[1] - train_csr.sum()) * 2)


    
    adj_norm = preprocess_graph(train_csr)

    adj_norm = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm[0].T),
        torch.FloatTensor(adj_norm[1]),
        torch.Size(adj_norm[2]),
    )

    print("Defining model, optimizar and features...")
    model = VBGAE(adj_norm, GRDPG=0)
    optimizer = Adam(model.parameters())

    X1 = torch.eye(adj_norm.size()[0]).cuda(torch.device(device)).to_sparse()
    X2 = torch.eye(adj_norm.size()[1]).cuda(torch.device(device)).to_sparse()

    n_epochs = 100
    roclist = []
    loss_list = []

    for epoch in range(1, n_epochs + 1):

        print("Running epoch", epoch)

        A_pred, Z1, Z2 = model(X1, X2)  

        optimizer.zero_grad()
        loss = norm * F.binary_cross_entropy(A_pred.view(-1), adj_norm.to_dense().view(-1))

        kl_divergence = 0.5/ A_pred.size(0) *( (1 + 2*model.logstd1 - model.mean1**2 - torch.exp(model.logstd1)**2).sum(1).mean()+
                                            (1 + 2*model.logstd2 - model.mean2**2 - torch.exp(model.logstd2)**2).sum(1).mean())
        loss -= kl_divergence
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        if epoch % 10 == 0:
            print("Epoch:", epoch, "Loss:", loss.item())