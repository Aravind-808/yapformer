import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        base = 10000

        # computing angles for each position
        pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        
        # freq[i] = base ^ (-i/dim)
        freq = torch.exp(torch.log(torch.tensor(base))* -torch.arange(0, d_model, 2)/d_model)
        angles = pos * freq

        # persistent = false bc its a learnable value but not part of state_dict (i.e not a core model param)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, q, k, seq_len=None, pos_offset = 0):
        if seq_len is None:
            seq_len = q.shape[2]

        cos = self.cos[pos_offset:pos_offset+seq_len, :].unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,dim/2]
        sin = self.sin[pos_offset:pos_offset+seq_len, :].unsqueeze(0).unsqueeze(0)

        def rotate_half(x):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).reshape_as(x)

        # interleave cos/sin along last dimension to match x
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        q_rot = (q * cos) + (rotate_half(q)*sin)
        k_rot = (k * cos) + (rotate_half(k)*sin)
        return q_rot, k_rot