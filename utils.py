'''
contains code for rmsnorm and SwiGLU

rmsnorm can be used in place of layernorm and SwiGLU in place of ReLU / GeLU
'''

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(data = torch.ones(1, 1, d_model)) # x.shape = (B, T, d_model) (torch.ones(d_model) also works ig)
        self.eps = eps

    def forward(self, x):

        mean_squared = x.pow(2).mean(dim= -1, keepdim = True)
        root_meaned_square = torch.sqrt(mean_squared + self.eps) # eps is a very small value that helps avoid division by zero error
        rms = (x/root_meaned_square) * self.weight

        return rms

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = self.lin1(x) # SwiGLU gated linear unit passes the input 2 times, once through a swish (input * sigmoid(input)) func and one through a linear proj
                           # and multiplies them both.
        swish = out * torch.sigmoid(x)

        swiglu = swish * self.lin2(x)

        return swiglu
    