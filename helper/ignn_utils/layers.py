import math
import numpy as np

import torch
import torch.sparse
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function

import scipy.sparse as sp
from .utils import projection_norm_inf


class ImplicitGraph(Module):
    """
    A Implicit Graph Neural Network Layer (IGNN)
    """

    def __init__(self, in_features, out_features, num_node, kappa=0.99, b_direct=False):
        super(ImplicitGraph, self).__init__()
        self.p = in_features
        self.m = out_features
        self.n = num_node
        self.k = kappa      # if set kappa=0, projection will be disabled at forward feeding.
        self.b_direct = b_direct

        self.W = Parameter(torch.FloatTensor(self.m, self.m))
        self.Omega_1 = Parameter(torch.FloatTensor(self.m, self.p))
        self.Omega_2 = Parameter(torch.FloatTensor(self.m, self.p))
        self.bias = Parameter(torch.FloatTensor(self.m, 1))
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.Omega_1.data.uniform_(-stdv, stdv)
        self.Omega_2.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X_0, A, U, phi, A_rho=1.0, fw_mitr=300, bw_mitr=300, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        if self.k is not None: # when self.k = 0, A_rho is not required
            self.W = projection_norm_inf(self.W, kappa=self.k/A_rho)
        support_1 = (U @ self.Omega_1.T).T
        # support_1 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_1.T).T
        # support_1 = torch.spmm(torch.transpose(A, 0, 1), support_1.T).T
        # support_2 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_2.T).T
        b_Omega = support_1 #+ support_2
        return ImplicitFunction.apply(self.W, X_0, A if A_orig is None else A_orig, b_Omega, phi, fw_mitr, bw_mitr)





class ImplicitFunction(Function):
    #ImplicitFunction.apply(input, A, U, self.X_0, self.W, self.Omega_1, self.Omega_2)
    @staticmethod
    def forward(ctx, W, X_0, A, B, phi, fd_mitr=300, bw_mitr=300):
        X_0 = B if X_0 is None else X_0
        X, err, status, D = ImplicitFunction.inn_pred(W, X_0, A, B, phi, mitr=fd_mitr, compute_dphi=True)
        ctx.save_for_backward(W, X, A, B, D, X_0, torch.tensor(bw_mitr))
        if status not in "converged":
            print("Iterations not converging!", err, status)
        return X

    @staticmethod
    def backward(ctx, *grad_outputs):

        #import pydevd
        #pydevd.settrace(suspend=False, trace_only_current_thread=True)

        W, X, A, B, D, X_0, bw_mitr = ctx.saved_tensors
        bw_mitr = bw_mitr.cpu().numpy()
        grad_x = grad_outputs[0]

        dphi = lambda X: torch.mul(X, D)
        grad_z, err, status, _ = ImplicitFunction.inn_pred(W.T, X_0, A, grad_x, dphi, mitr=bw_mitr, trasposed_A=True)
        #grad_z.clamp_(-1,1)

        grad_W = grad_z @ torch.spmm(A, X.T)
        grad_B = grad_z

        # Might return gradient for A if needed
        return grad_W, None, torch.zeros_like(A), grad_B, None, None, None

    @staticmethod
    def inn_pred(W, X, A, B, phi, mitr=300, tol=3e-6, trasposed_A=False, compute_dphi=False):
        # TODO: randomized speed up
        At = A if trasposed_A else torch.transpose(A, 0, 1)
        #X = B if X is None else X

        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            # WXA
            X_ = W @ X
            support = torch.spmm(At, X_.T).T
            X_new = phi(support + B)
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new

        dphi = None
        if compute_dphi:
            with torch.enable_grad():
                support = torch.spmm(At, (W @ X).T).T
                Z = support + B
                Z.requires_grad_(True)
                X_new = phi(Z)
                dphi = torch.autograd.grad(torch.sum(X_new), Z, only_inputs=True)[0]

        return X_new, err, status, dphi