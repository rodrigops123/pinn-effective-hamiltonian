import torch


def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


def msa(y_pred, y_true):
    return torch.mean(abs(y_pred - y_true))


def diagonal(A):
    return A.diagonal(offset=0, dim1=1, dim2=2)


def expected(A, B):
    return diagonal(A @ B)


def commutator(A, B):
    return torch.matmul(A, B) - torch.matmul(B, A)


def dagger(A):
    return torch.conj(A.T)