"""
Configuration file for TinyStories training
"""

class Config:
    vocab_size = 50257
    d_model = 512
    num_layers = 8
    num_q_heads = 8
    num_kv_heads = 2
    max_seq_len = 512
    dropout = 0.1
    
    batch_size = 4
    gradient_accumulation_steps = 8  # simulates batch size pf 32 bc we accumulate grads over 8 steps
    learning_rate = 3e-4
    weight_decay = 0.1
    max_epochs = 9999 # set ts so high we use only max_steps
    warmup_steps = 500
    max_steps = 15000  # max steps is basically the number of times the gradient is updated
    
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    mixed_precision = True
    num_workers = 4
    
    log_interval = 500
    eval_interval = 1000
    save_interval = 2500
    checkpoint_dir = './checkpoints'