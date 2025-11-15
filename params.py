from model import DecoderOnlyTransformer
from config import Config

config = Config()
model = DecoderOnlyTransformer(
    vocab_size=config.vocab_size,
    d_model=config.d_model,
    num_layers=config.num_layers,
    num_q_heads=config.num_q_heads,
    num_kv_heads=config.num_kv_heads,
    max_seq_len=config.max_seq_len
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params/1e6:.1f}M parameters")