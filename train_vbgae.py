import torch
import pickle
import os
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from parser import args
from utils import scipy_sparse_mat_to_torch_sparse_tensor
from vbgae import VBGAE


def get_scores(val_edges, val_edges_false, A_pred, threshold=0.5):
    # Extract predictions
    pos_pred = A_pred[val_edges].detach().cpu().numpy()
    neg_pred = A_pred[val_edges_false].detach().cpu().numpy()

    # Combine positive and negative predictions
    preds_all = np.hstack([pos_pred, neg_pred])
    labels_all = np.hstack([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])

    # Classification threshold
    binary_preds = (preds_all >= threshold).astype(int)

    # Metrics
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    accuracy = accuracy_score(labels_all, binary_preds)
    f1 = f1_score(labels_all, binary_preds)
    precision = precision_score(labels_all, binary_preds)
    recall = recall_score(labels_all, binary_preds)

    return roc_score, ap_score, accuracy, f1, precision, recall


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


def split_train_val_edges(adj, val_ratio=0.2):
    """
    Splits the adjacency CSR matrix into train and validation sets.

    Parameters:
        adj (sp.csr_matrix): Sparse adjacency matrix of the graph.
        val_ratio (float): Proportion of edges to use for validation.

    Returns:
        adj_train (sp.csr_matrix): Training adjacency matrix.
        train_edges (np.ndarray): Training edges as an array of [row, col].
        val_edges (np.ndarray): Validation edges as an array of [row, col].
    """
    # Extract edges (non-zero entries)
    row, col = adj.nonzero()
    edges = np.vstack((row, col)).T

    # Shuffle edges
    np.random.shuffle(edges)

    # Determine split index
    num_val = int(np.floor(len(edges) * val_ratio))

    # Split edges into validation and training sets
    val_edges = edges[:num_val]
    train_edges = edges[num_val:]

    # Create training adjacency matrix
    data = np.ones(len(train_edges))
    adj_train = sp.csr_matrix(
        (data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape
    )

    return adj_train, train_edges, val_edges


def mask_test_edges(adj, adj_total):
    # Function to build test set with 10% positive links

    edges = np.where(adj > 0)
    non_edges = np.where(adj_total == 0)

    permut_edges = np.random.permutation(edges[0].shape[0])
    edges = edges[0][permut_edges], edges[1][permut_edges]

    permut_non_edges = np.random.permutation(non_edges[0].shape[0])
    non_edges = non_edges[0][permut_non_edges], non_edges[1][permut_non_edges]

    num_test_non_edges = int(np.floor(edges[0].shape[0] / 10.))
    num_test_edges = 0
    num_val  = int(np.floor(edges[0].shape[0] / 20.))


    edges = np.split(edges[0], [num_test_edges, num_test_edges + num_val]), np.split(
        edges[1], [num_test_edges, num_test_edges + num_val]
    )
    non_edges = np.split(non_edges[0], [num_test_non_edges, num_test_non_edges + num_val]), np.split(
        non_edges[1], [num_test_non_edges, num_test_non_edges + num_val]
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
        # test_edges,
        test_edges_false,
    )


if __name__ == "__main__":

    device = "cuda:" + args.cuda if torch.cuda.is_available() else "cpu"

    print("Defining matrices...")

    path = "data/" + args.data + "/"
    f = open(path + "trnMat.pkl", "rb")
    train = pickle.load(f)
    train_csr = (train != 0).astype(np.float32)  # adjacency matrix in csr format

    f = open(path + "tstMat.pkl", "rb")
    test = pickle.load(f)
    test_csr = (test != 0).astype(np.float32)  # adjacency matrix in csr format
    
    pos_weight = (
        float(train_csr.shape[0] * train_csr.shape[1] - train_csr.sum())
        / train_csr.sum()
    )
    norm = (
        train_csr.shape[0]
        * train_csr.shape[1]
        / float((train_csr.shape[0] * train_csr.shape[1] - train_csr.sum()) * 2)
    )

    file_paths = {
        "adj_train": "adj_train.pkl",
        "train_edges": "train_edges.pkl",
        "val_edges": "val_edges.pkl",
        "val_edges_false": "val_edges_false.pkl",
        "test_edges": "test_edges.pkl",
        "test_edges_false": "test_edges_false.pkl",
    }

    if all(os.path.exists(path) for path in file_paths.values()):
        print("Loading data from local files...")
        with open(file_paths["adj_train"], "rb") as f:
            adj_train = pickle.load(f)
        with open(file_paths["train_edges"], "rb") as f:
            train_edges = pickle.load(f)
        with open(file_paths["val_edges"], "rb") as f:
            val_edges = pickle.load(f)
        with open(file_paths["val_edges_false"], "rb") as f:
            val_edges_false = pickle.load(f)
        with open(file_paths["test_edges"], "rb") as f:
            test_edges = pickle.load(f)
        with open(file_paths["test_edges_false"], "rb") as f:
            test_edges_false = pickle.load(f)
    else:

        adj_total = train_csr + test_csr

        print("Computing data...")
        # Compute the outputs
        (
            adj_train,
            train_edges,
            val_edges,
            val_edges_false,
            # test_edges,
            test_edges_false,
        ) = mask_test_edges(torch.from_numpy(train_csr.toarray()), torch.from_numpy(adj_total.toarray()))

        sparse_mx = test_csr.tocoo()
        coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
        test_edges = sparse_mx.row, sparse_mx.col

        # Save the outputs
        with open(file_paths["adj_train"], "wb") as f:
            pickle.dump(adj_train, f)  # Save as sparse matrix
        with open(file_paths["train_edges"], "wb") as f:
            pickle.dump(train_edges, f)
        with open(file_paths["val_edges"], "wb") as f:
            pickle.dump(val_edges, f)
        with open(file_paths["val_edges_false"], "wb") as f:
            pickle.dump(val_edges_false, f)
        with open(file_paths["test_edges"], "wb") as f:
            pickle.dump(test_edges, f)
        with open(file_paths["test_edges_false"], "wb") as f:
            pickle.dump(test_edges_false, f)


    print(test_edges)
    print(test_edges_false)

    print(type(test_edges))
    print(type(test_edges_false))

    weight_mask = torch.from_numpy(adj_train.toarray()).view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0), device=device)
    weight_tensor[weight_mask] = pos_weight

    adj_norm = preprocess_graph(adj_train)

    adj_norm = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm[0].T),
        torch.FloatTensor(adj_norm[1]),
        torch.Size(adj_norm[2]),
    )
    adj_norm = adj_norm.cuda(torch.device(device))
    torch.save(adj_norm, "adj_norm.pth")

    print("Defining model, optimizar and features...")
    model = VBGAE(adj_norm, GRDPG=1)
    optimizer = Adam(model.parameters(), lr=1e-2)

    X1 = torch.eye(adj_norm.size()[0]).cuda(torch.device(device)).to_sparse()
    X2 = torch.eye(adj_norm.size()[1]).cuda(torch.device(device)).to_sparse()

    n_epochs = 200
    metrics = []
    loss_list = []
    
    adj_train_as_tensor = torch.from_numpy(adj_train.toarray()).to(dtype=torch.float32, device=torch.device(device))


    for epoch in range(1, n_epochs + 1):

        A_pred, Z1, Z2 = model(X1, X2)

        optimizer.zero_grad()

        loss = norm * F.binary_cross_entropy(
            A_pred.view(-1), adj_train_as_tensor.view(-1), weight=weight_tensor
        )

        kl_divergence = (
            0.5
            / A_pred.size(0)
            * (
                (1 + 2 * model.logstd1 - model.mean1**2 - torch.exp(model.logstd1) ** 2)
                .sum(1)
                .mean()
                + (
                    1
                    + 2 * model.logstd2
                    - model.mean2**2
                    - torch.exp(model.logstd2) ** 2
                )
                .sum(1)
                .mean()
            )
        )
        loss -= kl_divergence
        loss.backward()
        optimizer.step()

        roc_score, ap_score, accuracy, f1, precision, recall = get_scores(
            val_edges, val_edges_false, A_pred
        )

        metrics.append(
            {
                "roc": roc_score,
                "ap": ap_score,
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
        )
        loss_list.append(loss.item())

        if epoch % 10 == 0:

            print("Epoch:", epoch, "Loss:", loss.item())
            print(
                "ROC:",
                roc_score,
                "AP:",
                ap_score,
                "Accuracy:",
                accuracy,
                "Recall:",
                recall,
                "Precision:",
                precision,
                "F1:",
                f1,
            )


    print("TEST TIME")

    A_pred, Z1, Z2 = model(X1, X2)

    roc_score, ap_score, accuracy, f1, precision, recall = get_scores(
        test_edges, test_edges_false, A_pred
    )
    
    print("ROC:", roc_score, "AP:", ap_score, "Accuracy:", accuracy, "Recall:", recall, "Precision:", precision, "F1:", f1)

    torch.save(model.state_dict(), "model.pth")