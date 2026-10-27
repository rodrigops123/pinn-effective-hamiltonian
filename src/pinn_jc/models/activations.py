import torch
import torch.nn as nn


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)
