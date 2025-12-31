"""
Inference script for generating text with trained model
"""

import torch
from transformers import GPT2Tokenizer
from model import DecoderOnlyTransformer
from config import Config
import time
import warnings
warnings.filterwarnings("ignore")

config = Config()

ASCII_ART = r"""
                                 ,d8888b                                               
                                 88P'                                                  
                              d888888P                                                 
?88   d8P  d888b8b  ?88,.d88b,  ?88'     d8888b   88bd88b  88bd8b,d88b  d8888b  88bd88b
d88   88  d8P' ?88  `?88'  ?88  88P     d8P' ?88  88P'  `  88P'`?8P'?8bd8b_,dP  88P'  `
?8(  d88  88b  ,88b   88b  d8P d88      88b  d88 d88      d88  d88  88P88b     d88     
`?88P'?8b `?88P'`88b  888888P'd88'      `?8888P'd88'     d88' d88'  88b`?888P'd88'     
       )88            88P'                                                             
      ,d8P           d88                                                               
   `?888P'           ?8P                                                               
"""

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_length=200, temperature=0.4, top_k=60, device='cuda'):
    
    model.eval()

    for layer in model.transformer.layers:
        layer['attention'].reset_cache()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_ids = input_ids.clone()
    logits = model(input_ids, use_cache=True)

    for _ in range(max_length):
        
        next_token_logits = logits[0, -1, :] / temperature
        
        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
        probs = torch.softmax(top_k_logits, dim=-1)
        next_token = top_k_indices[torch.multinomial(probs, 1)]
        
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break

        logits = model(next_token.unsqueeze(0), use_cache=True) # only passing next token since we are using kv cache
    
    for layer in model.transformer.layers:
        layer['attention'].reset_cache()
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def load_model(checkpoint_path, config):
    print(f"Loading model from {checkpoint_path}...")
    
    model = DecoderOnlyTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        max_seq_len=config.max_seq_len
    ).to(config.device)
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded! (Step: {checkpoint['step']}, Loss: {checkpoint['loss']:.4f})")
    
    return model, tokenizer


def interactive_generation(model, tokenizer, config):
    print("="*70)
    print(ASCII_ART)
    print("Type your prompt and press Enter. Type 'quit' to exit.")
    print("="*70 + "\n")
    
    temperature = 0.6
    max_length = 200
    
    while True:
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not prompt:
            print("Please enter a prompt!")
            continue
        
        print("\nGenerating...\n")
        generated = generate_text(
            model, 
            tokenizer, 
            prompt,
            max_length=max_length,
            temperature=temperature,
            device=config.device
        )
        
        print("-" * 70)
        # end the generated text at the nearest fullstop when the max length is reached
        if len(generated) >= max_length:
            last_period = generated.rfind('.', len(prompt))
            if last_period != -1:
                generated = generated[:last_period + 1]
        
        disp_buf= "" # the model loves continuing the story after "the end" so ima make smth to detect "the end" and stop generating
        for char in generated:
            disp_buf += char
            if "the end" in disp_buf.lower(): # detect "the end"
                print("d.") # print "The End."
                break
            print(char, end = '')
            time.sleep(0.01)
        print("\n")

        print("-" * 70)


if __name__ == "__main__":
    # Load model
    model, tokenizer = load_model("checkpoints\checkpoint_step_15000.pt", config)
    interactive_generation(model, tokenizer, config)
