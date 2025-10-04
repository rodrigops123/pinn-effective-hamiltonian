import torch
import torch.nn as nn

def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


def mae(y_pred, y_true):
    return torch.mean(abs(y_pred - y_true))


def diagonal(A):
    return A.diagonal(offset=0, dim1=1, dim2=2)


def expected(A, B):
    return diagonal(A @ B)


def commutator(A, B):
    return torch.matmul(A, B) - torch.matmul(B, A)


def dagger(A):
    return torch.conj(A.T)


class SIN(nn.Module):
    def __init__(self):
        super(SIN, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class TANH(nn.Module):
    def __init__(self):
        super(TANH, self).__init__()

    def forward(self, x):
        return torch.tanh(x)