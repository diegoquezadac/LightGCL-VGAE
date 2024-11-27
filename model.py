import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F
from vbgae import VBGAE

class LightGCL(nn.Module):
    def __init__(self,
    n_u, # NOTE: Number of users
    n_i, # NOTE: Number of items
    d,   # NOTE: Embedding size
    u_mul_s, # NOTE: Precomputed matrix multiplication
    v_mul_s, # NOTE: Precomputed matrix multiplication
    ut,  # NOTE: SVD U transposed
    vt,  # NOTE: SVD V transposed
    train_csr, # NOTE: User-Item matrix
    adj_norm,  # NOTE: User-Item matrix coalesced
    l,    # NOTE: Number of gnn layers
    temp, # NOTE: Temperature in cl loss
    lambda_1, # NOTE: weight of cl loss
    lambda_2, # NOTE: l2 reg weight
    dropout,
    batch_user,
    use_vbgae,
    A_vbgae,
    device):
        super(LightGCL,self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user
        self.use_vbgae = use_vbgae
        self.A_vbgae = A_vbgae

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

    def forward(self, uids, iids, pos, neg, test=False):
        if test==True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1-mask) - 1e8 * mask
            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase
            for layer in range(1,self.l+1):
                # GNN propagation
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1])) # (I, J) * (J, d) = (I, d)
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1])) # (J, I) * (I, d) = (J, d)

                # svd_adj propagation
                if self.use_vbgae:
                    
                    self.G_u_list[layer] = (torch.mm(F.dropout(self.A_vbgae,self.dropout), self.E_i_list[layer-1])) # (I, J) * (J, d) = (I, d)
                    self.G_i_list[layer] = (torch.mm(F.dropout(self.A_vbgae,self.dropout).transpose(0,1), self.E_u_list[layer-1])) # (J, I) * (I, d) = (J, d)

                    print(self.G_u_list[layer])

                else:
                    vt_ei = self.vt @ self.E_i_list[layer-1] # (q, J) * (J, d) = (q, d)
                    self.G_u_list[layer] = (self.u_mul_s @ vt_ei) # (I, q) * (q,d) = (I,d)
                    ut_eu = self.ut @ self.E_u_list[layer-1] # (q, I) * (I, d) = (q, d)
                    self.G_i_list[layer] = (self.v_mul_s @ ut_eu) # (J, q) * (q, d)  = (J, d)
                    
                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer] # (I, d)
                self.E_i_list[layer] = self.Z_i_list[layer] # (J, d)

            self.G_u = sum(self.G_u_list) # [ (I,d), (I,d), (I,d), (I,d), ... ] (I,d)
            self.G_i = sum(self.G_i_list) # (J,d)

            # aggregate across layers
            self.E_u = sum(self.E_u_list) # (I, d)
            self.E_i = sum(self.E_i_list) # (J, d)

            # cl loss
            G_u_norm = self.G_u # (I,d)
            E_u_norm = self.E_u # (I, d)
            G_i_norm = self.G_i # (J, d)
            E_i_norm = self.E_i # (J, d)

            # a = G_u_norm[uids] @ E_u_norm.T # (b, d) * (d, I) = (b, I) Similitud entre usuarios del batch contra todos los usuarios
            # torch.exp(a / self.temp).sum(1) # (b, 1)

            # DENOMINADOR
            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean() 
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()

            # NUMERADOR
            # a = G_u_norm[uids] * E_u_norm[uids] # (b, d) * (b, d) = (b, d)
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()

            loss_s = -pos_score + neg_score

            # bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # total loss
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            #print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
            return loss, loss_r, self.lambda_1 * loss_s