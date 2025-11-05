import torch
import torch.nn as nn
import itertools
import random
import numpy as np


class Sin(torch.nn.Module):
    def forward(self, x):
        f = torch.sin(x)
        return f


class Cos(torch.nn.Module):
    def forward(self, x):
        f = torch.cos(x)
        return f


class Exp(torch.nn.Module):
    def forward(self, x):
        f = torch.exp(x)
        return f


class ExpN(torch.nn.Module):
    def forward(self, x):
        f = torch.exp(-x)
        return f


class ExpAbs(torch.nn.Module):
    def forward(self, x):
        f = torch.exp(-0.01 * abs(x))
        return f


class ExpAbsP(torch.nn.Module):
    def forward(self, x):
        f = torch.exp(0.01 * abs(x))
        return f


class Sqrt(torch.nn.Module):
    # this is an approximation of the square root function in order to avoid numerical instability
    def __init__(self):
        super(Sqrt, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        f = (0.01 + self.relu(x)) ** 0.5
        return f


class Log(torch.nn.Module):
    # this is an approximation of the log function in order to avoid numerical instability
    def __init__(self):
        super(Log, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        f = torch.log(0.1 + self.relu(x))
        return f


class Id(torch.nn.Module):
    def forward(self, x):
        f = x
        return f


class Tanh(torch.nn.Module):
    def forward(self, x):
        f = torch.tanh(x)
        return f


class Linear(nn.Module):
    def __init__(self, a=1.0, b=0.0):
        super(Linear, self).__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))

    def forward(self, x):
        f = self.a * x + self.b
        return f


functions = [Sin(), Cos(), Id()]
L = len(functions)


class Quad(nn.Module):
    def __init__(self, n_in, n_out, second_order=True):
        super(Quad, self).__init__()

        self.second_order = second_order

        # first order neurons
        if second_order is False:
            self.linear = nn.Linear(n_in, n_out)

        # second order neurons
        if second_order is True:
            L = int(n_in * (n_in - 1) / 2)
            self.linear = nn.Linear(L + n_in, n_out)
            self.ids = torch.triu_indices(n_in, n_in, 1)

    def forward(self, x):

        if self.second_order is True:
            x2 = x[:, :, None] @ x[:, None, :]  # (batch, n_in, n_in)
            x2 = x2[:, self.ids[0], self.ids[1]]  # (batch, L)

            x = torch.cat((x, x2), axis=1)  # (batch, n_in + L)
            x = self.linear(x)

            return x

        else:
            x = self.linear(x)

            return x


class Mixfun(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        normalization_function=False,
        normalization_neuron=False,
        p_drop=False,
        second_order_input=False,
        second_order_function=False,
        temperature=1.0,
        create_parameter=False,
        n_paramater=1,
    ):
        super(Mixfun, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.p_drop = p_drop
        if p_drop is not False:
            self.dropout = nn.Dropout(p=p_drop)
        self.second_order_function = second_order_function
        self.temperature = temperature

        # first order projection
        self.project1 = Quad(n_in, L * n_out, second_order=second_order_input)

        if second_order_function is True:
            # second order projection
            self.l = int(L * (L + 1) / 2)
            self.project2_1 = Quad(n_in, L * n_out, second_order=second_order_function)
            self.project2_2 = Quad(n_in, L * n_out, second_order=second_order_function)
            self.ids = torch.triu_indices(L, L, 0)

        # neuron output
        self.normalization_function = (
            normalization_function  # forces each neuron to choose a single function
        )
        self.normalization_neuron = normalization_neuron  # forces each neuron to have a different function from the others
        if second_order_function is True:
            if normalization_function is True and normalization_neuron is True:
                self.p1 = nn.Parameter(torch.ones(n_out, L + self.l))
                self.p2 = nn.Parameter(torch.ones(n_out, L + self.l))
            if normalization_function is False and normalization_neuron is True:
                self.p = nn.Parameter(torch.ones(n_out, L + self.l))
            if normalization_function is True and normalization_neuron is False:
                self.p = nn.Parameter(torch.ones(n_out, L + self.l))
            if normalization_function is False and normalization_neuron is False:
                self.p = nn.Parameter(torch.randn(n_out, L + self.l))
        else:
            if normalization_function is True and normalization_neuron is True:
                self.p1 = nn.Parameter(torch.ones(n_out, L))
                self.p2 = nn.Parameter(torch.ones(n_out, L))
            if normalization_function is False and normalization_neuron is True:
                self.p = nn.Parameter(torch.ones(n_out, L))
            if normalization_function is True and normalization_neuron is False:
                self.p = nn.Parameter(torch.ones(n_out, L))
            if normalization_function is False and normalization_neuron is False:
                self.p = nn.Parameter(torch.randn(n_out, L))

        if normalization_function is True or normalization_neuron is True:
            self.amplitude = nn.Parameter(torch.randn(n_out))

        if create_parameter:
            self.param = nn.Parameter(
                torch.rand(n_paramater), requires_grad=True
            )

    def forward(self, x):

        if self.second_order_function is True:
            # first order functions
            x1 = self.project1(x).reshape((x.shape[0], self.n_out, L))
            x1 = torch.stack(
                [functions[i](x1[:, :, i]) for i in range(L)], dim=1
            ).reshape((x.shape[0], self.n_out, L))

            # second order functions
            x2_1 = self.project2_1(x).reshape((x.shape[0], self.n_out, L))
            x2_1 = torch.stack(
                [functions[i](x2_1[:, :, i]) for i in range(L)], dim=1
            ).reshape((x.shape[0], self.n_out, L))
            x2_2 = self.project2_2(x).reshape((x.shape[0], self.n_out, L))
            x2_2 = torch.stack(
                [functions[i](x2_2[:, :, i]) for i in range(L)], dim=1
            ).reshape((x.shape[0], self.n_out, L))
            x2 = x2_1[:, :, :, None] @ x2_2[:, :, None, :]
            x2 = x2[:, :, self.ids[0], self.ids[1]]

            # concatenate first order and second order functions
            x = torch.cat((x1, x2), axis=2)

        else:
            # first order functions
            x = self.project1(x).reshape((x.shape[0], self.n_out, L))
            x = torch.stack(
                [functions[i](x[:, :, i]) for i in range(L)], dim=1
            ).reshape((x.shape[0], self.n_out, L))

        if self.p_drop is not False and self.training is True:
            x = self.dropout(x)

        if self.normalization_function is True and self.normalization_neuron is True:
            Z = torch.sum(torch.exp(-self.p1 / self.temperature), axis=1)
            p1 = torch.exp(-self.p1 / self.temperature) / (
                1e-8 + Z.reshape((self.n_out, 1))
            )

            Z = torch.sum(torch.exp(-self.p2 / self.temperature), axis=0)
            if self.second_order_function is True:
                p2 = torch.exp(-self.p2 / self.temperature) / (
                    1e-8 + Z.reshape((1, L + self.l))
                )
            else:
                p2 = torch.exp(-self.p2 / self.temperature) / (1e-8 + Z.reshape((1, L)))

            x = self.amplitude * torch.sum(p2 * p1 * x, axis=2)

        elif self.normalization_function is True and self.normalization_neuron is False:
            Z = torch.sum(torch.exp(-self.p / self.temperature), axis=1)
            p = torch.exp(-self.p / self.temperature) / (
                1e-8 + Z.reshape((self.n_out, 1))
            )
            x = self.amplitude * torch.sum(p * x, axis=2)

        elif self.normalization_function is False and self.normalization_neuron is True:
            Z = torch.sum(torch.exp(-self.p / self.temperature), axis=0)
            if self.second_order_function is True:
                p = torch.exp(-self.p / self.temperature) / (
                    1e-8 + Z.reshape((1, L + self.l))
                )
            else:
                p = torch.exp(-self.p / self.temperature) / (1e-8 + Z.reshape((1, L)))
            x = self.amplitude * torch.sum(p * x, axis=2)

        else:
            x = torch.sum(self.p * x, axis=2)

        return x
