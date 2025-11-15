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

        # weight tying, shares same weights bw input embedding and output layer
        self.lm_head.weight = self.token_embedding.weight
        
    def create_mask(self, input_ids, pad_token_id=0):
        """
        Create combined padding + causal mask
        input_ids: (B, T)
        Returns: (B, 1, T, T)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Padding mask: (B, 1, 1, T) - marks which positions are padding
        padding_mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
        
        # Causal mask: (1, 1, T, T) - prevents attending to future
        causal_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=device)).bool()
        
        # Combine: position is valid if both non-padding AND not future
        mask = padding_mask & causal_mask
        
        return mask

    def forward(self, input_ids, mask=None, use_cache = False):
        x = self.token_embedding(input_ids)

        if mask is None and not use_cache:
            mask = self.create_mask(input_ids)

        x = self.transformer(x, mask=mask, use_cache = use_cache)
        x = self.final_norm(x) # this is completely optional, some implementations have it some dont. those that have it include llama, gpt-3 etc.
        logits = self.lm_head(x)
        return logits