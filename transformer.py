'''
Transformer module implementing a stack of attention and feedforward layers.
We are implementing what is called a Pre-LN Transformer, where layer
normalization is applied before attention and feedforward sub-layers.

For more on Pre-LN Transformers, see:
- "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
- https://arxiv.org/abs/2002.04745

'''

import torch
import torch.nn as nn
from attention import GroupedQueryAttention
from utils import RMSNorm, SwiGLU

class Transformer(nn.Module):
    def __init__(self, d_model, num_layers, num_q_heads, num_kv_heads, max_seq_len, eps=1e-8):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': GroupedQueryAttention(d_model, num_q_heads, num_kv_heads, max_seq_len),
                'rmsnorm1': RMSNorm(d_model),
                'ffn': SwiGLU(d_model),
                'rmsnorm2': RMSNorm(d_model)
            }) for _ in range(num_layers)
        ])
        self.max_seq_len = max_seq_len

    def forward(self, x, mask=None):
        """
        x: (B, T, d_model)
        mask: optional attention mask
        """
        for layer in self.layers:
            # attn block
            x_norm = layer['rmsnorm1'](x)
            attn_out = layer['attention'](x_norm, mask=mask)
            x = x + attn_out 

            # ffn block
            x_norm = layer['rmsnorm2'](x)
            ffn_out = layer['ffn'](x_norm)
            x = x + ffn_out  

        return x