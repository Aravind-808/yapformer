"""
Complete training script for TinyStories dataset
Optimized for RTX 3060 6GB VRAM
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os

# Import your model and config
from model import DecoderOnlyTransformer
from config import Config
import warnings
warnings.filterwarnings("ignore")

config = Config()

def prepare_dataset():

    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=config.max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        outputs['labels'] = outputs['input_ids'].clone()
        return outputs
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"
    )
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
    
    return tokenized_dataset, tokenizer


def create_model(config):
    model = DecoderOnlyTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        max_seq_len=config.max_seq_len
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model.to(config.device)


def get_lr_scheduler(optimizer, warmup_steps, max_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_step(model, batch, optimizer, scaler, config):
    input_ids = batch['input_ids'].to(config.device)
    labels = batch['labels'].to(config.device)
    
    with autocast(enabled=config.mixed_precision): # mixed precision helps with memory
        logits = model(input_ids)
        
        shift_logits = logits[..., :-1, :].contiguous() # since we predict next token,shift by 1
        shift_labels = labels[..., 1:].contiguous()
        
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1),
            ignore_index=50256  # this is the padding token
        )
    
    scaler.scale(loss).backward()
    
    return loss.item()

@torch.no_grad()
def evaluate(model, eval_dataloader, config, max_batches=50):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for i, batch in enumerate(eval_dataloader):
        if i >= max_batches:
            break
            
        input_ids = batch['input_ids'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        with autocast(enabled=config.mixed_precision):
            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, config.vocab_size),
                shift_labels.view(-1),
                ignore_index=50256
            )
        
        total_loss += loss.item()
        num_batches += 1
    
    model.train()
    return total_loss / num_batches

def save_checkpoint(model, optimizer, scheduler, step, loss, config): #save chkpoint incase model fails during training
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': vars(config)
    }
    path = os.path.join(config.checkpoint_dir, f'checkpoint_step_{step}.pt')
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def train(config):
    """Main training function"""
    
    # prep data
    tokenized_dataset, tokenizer = prepare_dataset()
    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['validation']
    
    # data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # model
    model = create_model(config)
    
    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay
    )
    
    # scheduler for lr
    scheduler = get_lr_scheduler(optimizer, config.warmup_steps, config.max_steps)
    
    # mixed precision scaler
    scaler = GradScaler(enabled=config.mixed_precision)
    
    # train :3
    model.train()
    global_step = 0
    running_loss = 0
    
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")
    
    for epoch in range(config.max_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.max_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            loss = train_step(model, batch, optimizer, scaler, config)
            running_loss += loss
            
            # gradient accumulation simulates larger batch size while processing smaller ones by accumulating grads
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # clip gradients to norm 1.0 to prevent explosion boom boom boom
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1
                
                # simple logging
                if global_step % config.log_interval == 0:
                    avg_loss = running_loss / config.log_interval
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                        'step': global_step
                    })
                    
                    running_loss = 0
                
                # for each eval interval, run eval
                if global_step % config.eval_interval == 0:
                    eval_loss = evaluate(model, eval_dataloader, config)
                    print(f"\nStep {global_step} - Eval Loss: {eval_loss:.4f}")
                
                # saving checkpoint
                if global_step % config.save_interval == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step, loss, config)
                
                # stop training if max steps reached
                if global_step >= config.max_steps:
                    print(f"\nReached max steps ({config.max_steps}). Stopping training.")
                    save_checkpoint(model, optimizer, scheduler, global_step, loss, config)
                    return model, tokenizer
    
    save_checkpoint(model, optimizer, scheduler, global_step, loss, config)
    print("\nTraining completed!")
    
    return model, tokenizer

if __name__ == "__main__":
    train(config)