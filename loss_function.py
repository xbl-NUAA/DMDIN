import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from scipy.io import loadmat
from sklearn import preprocessing
import seaborn as sns
import scipy.io as sio
import math
import itertools
from sklearn.metrics import  roc_auc_score
from torch.utils.data import DataLoader


def GCCA(M_list, rank_m = 10, r = 1e-8):
    UD_list = []
    for M in M_list:
        assert torch.isnan(M).sum().item() == 0
        U, S, V = M.svd(some=True, compute_uv=True)
        S = S[:rank_m]
        U = U[:,:rank_m]
        assert torch.isnan(S).sum().item() == 0
        S_thin = S[:rank_m]
        S2_inv = 1. / (torch.mul(S_thin, S_thin) + r)
        assert torch.isnan(S2_inv).sum().item() == 0
        T2 = torch.mul(torch.mul(S_thin, S2_inv), S_thin)
        assert torch.isnan(T2).sum().item() == 0
        D = torch.diag(torch.sqrt(T2))
        assert torch.isnan(D).sum().item() == 0
        UD = torch.matmul(U, D)
        UD_list.append(UD)

    Q = torch.cat(UD_list, dim=1)
    assert torch.isnan(Q).sum().item() == 0
    G, H, E = Q.svd(some=True, compute_uv=True)
    G = G[:,:rank_m]
    assert torch.isnan(G).sum().item() == 0
    R_list = []
    views = M_list
    F = [M.shape[0] for M in M_list]

    for idx, (f, view) in enumerate(zip(F, views)):
        MM_inv = torch.inverse((view.T.mm(view) + r * torch.eye(view.shape[1], device=view.device)))
        assert torch.isnan(MM_inv).sum().item() == 0
        R = torch.matmul(torch.matmul(MM_inv, view.T),G)
        R_list.append(R)

    S_k = H[:rank_m]
    corr = torch.sum(S_k)
    assert torch.isnan(corr).item() == 0
    loss = - corr

    return loss, G, R_list


def back(G, R_list,Z,F_list,lr,lambda1,lambda2,lambda3,A):
    z_grad1 = []
    z_grad3 = []
    FX_list = []
    loss = 0
    for i in range(len(R_list)):
        GR = torch.matmul(G,R_list[i].T)
        ZF = torch.matmul(Z,F_list[i])
        RR = torch.matmul(R_list[i],R_list[i].T)
        z_grad1.append(torch.matmul(GR - torch.matmul(ZF,RR),F_list[i].T))
        FF = torch.matmul(F_list[i],F_list[i].T)
        z_grad3.append(torch.matmul(FF,(Z - torch.eye(Z.shape[0]))))

        f_grad1 = torch.matmul(Z, GR - torch.matmul(ZF, RR))
        ZI = Z - torch.eye(Z.shape[0])
        f_grad2 = torch.matmul(torch.matmul(ZI, ZI), F_list[i])
        f_grad = 2 * (f_grad1 + lambda2 * f_grad2)
        FX = F_list[i] - lr * f_grad
        FX_list.append(FX)

    z_grad2 = Z
    z_grad4 = torch.matmul(torch.mul(A,Z),A)
    z_grad = 2 * (sum(z_grad1) + lambda1 * z_grad2 + lambda2 * sum(z_grad3) + lambda3 * z_grad4)

    for F in FX_list:
        loss += torch.sum(torch.pow(torch.matmul(Z, F) - F, 2))

    Z = Z - lr * z_grad


    return Z,FX_list,loss

def loss_function(z, block, f_list, zf_list, weight_coef, weight_selfExp, weight_block,lr):
    loss_gcca, G, R_list = GCCA(zf_list)
    loss_coef = torch.sum(torch.pow(z, 2))
    loss_self = []
    for i in range(len(f_list)):
        loss_self.append( F.mse_loss(f_list[i], zf_list[i], reduction='sum'))
    loss_block = torch.sum(torch.pow(torch.mul(block,z), 2))
    loss = loss_gcca +  weight_coef * loss_coef + weight_selfExp * sum(loss_self) + weight_block * loss_block

    z, FX, FX_loss = back(G, R_list, z, f_list, lr, weight_coef,
                                                  weight_selfExp, weight_block, block)

    return loss, FX_loss
