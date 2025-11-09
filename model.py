'''
Decoder-only Transformer model implementation.

Note: using final_norm before lm_head(prediction) is optional. it is mostly used for training stability.
'''

import torch
import torch.nn as nn
from transformer import Transformer
from utils import RMSNorm

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_q_heads, 
                 num_kv_heads, max_seq_len, eps=1e-8):
        super().__init__()
        
        # input embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # transformer 
        self.transformer = Transformer(d_model, num_layers, num_q_heads, 
                                       num_kv_heads, max_seq_len, eps)
        
        # final norm (optional) and output
        self.final_norm = RMSNorm(d_model, eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids, mask=None):
        x = self.token_embedding(input_ids)
        x = self.transformer(x, mask=mask)
        x = self.final_norm(x) # this is completely optional, some implementations have it some dont. those that have it include llama, gpt-3 etc.
        logits = self.lm_head(x)
        return logits