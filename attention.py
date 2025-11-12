import torch
import torch.nn as nn
import math
from rope import RotaryPositionalEmbedding
from kv_cache import KVCaching

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    - RoPE is nice but needs GQA for faster inference since an increasing KV cache becomes a bottleneck.
    - GQA overcomes this by computing less KV pairs per query.
    - for example: if there are 8 query heads and 2 kv head pairs, its divided into groups of 2 where the first four query heads
      tend to first kv pair and the next 4 tend to the second pair
    """
    def __init__(self, d_model, num_q_heads, num_kv_heads, max_seq_len):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // num_q_heads
        
        # linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.kv_cache = None
    
    def split_heads(self, x, num_heads):
        b, seq_len, d = x.shape
        head_dim = d // num_heads
        return x.view(b, seq_len, num_heads, head_dim).transpose(1, 2)  # (B, H, T, d_head)
    
    def combine_heads(self, x):
        b, h, t, d = x.shape
        return x.transpose(1, 2).reshape(b, t, h * d)
    
    def reset_cache(self): # reset cache after every sequence generated. 
        self.kv_cache = None
    
    def forward(self, x, mask=None, use_cache=False):
        """
        x: (B, T, d_model)
        mask: optional attention mask
        use_cache: whether to use KV caching (for inference only)
        """
        b, t, _ = x.shape
        device = x.device
        
        Q = self.split_heads(self.W_q(x), self.num_q_heads)  # (B, Hq, T, d_head)
        K = self.split_heads(self.W_k(x), self.num_kv_heads) # (B, Hkv, T, d_head)
        V = self.split_heads(self.W_v(x), self.num_kv_heads) # (B, Hkv, T, d_head)
        
        start_pos = 0
        
        if use_cache:
            if self.kv_cache is None:
                self.kv_cache = KVCaching(
                    batch=b, 
                    heads=self.num_kv_heads, 
                    max_seq_len=self.max_seq_len, 
                    d_k=self.head_dim, 
                    device=device
                )
            start_pos = self.kv_cache.cache_idx
            
            # apply rotary embeddings with position offset
            Q, K = self.rope(Q, K, pos_offset=start_pos)
            
            # update cache and get all past keys/values
            self.kv_cache.update_cache(K, V)
            K, V = self.kv_cache.get_cache()  # (B, Hkv, T_cache, d_head)
        else:
            # we dont use kv caching for training.
            Q, K = self.rope(Q, K, pos_offset=0)
        
        # expand kv heads to match query heads
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)
        
        # scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, Hq, T, d_head)
        
        out = self.W_o(self.combine_heads(attn_output))  # (B, T, d_model)
        return out