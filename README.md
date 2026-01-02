# **YapFormer : A Transformer Implementation with Modern Optimizations ✧ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧**

https://github.com/user-attachments/assets/4f6f1d3b-4388-4657-a99c-e4b7251e04f8

**YapFormer** is a transformer model built entirely from scratch, featuring modern architectural components and efficient training optimizations.  
The final model contains **~56 million parameters** and was trained for **15,000 steps** (~4.5 hours) on the **TinyStories** dataset.

Despite the small size and short training time, YapFormer produces surprisingly high-quality short stories, demonstrating that well-designed architectures can go a long way even with limited compute.

----------

#  **What is YapFormer? ૮ ◕ ﻌ ◕ა**

YapFormer is a **from-scratch GPT-style autoregressive transformer** that integrates many techniques used in contemporary LLMs:

-   Rotary Embeddings (RoPE)
    
-   Grouped Query Attention (GQA)
    
-   KV caching for fast inference
    
-   RMSNorm
    
-   SwiGLU feed-forward layers
    
-   Mixed precision training
    
-   Gradient accumulation
    
-   Cosine decay learning rate
    
-   Gradient clipping
    

This project serves as both a learning exercise and a practical lightweight generative model.

----------

#  **Working ૮₍ • ᴥ • ₎ა**

### **1. Input & Embeddings**

-   Tokens are mapped using a custom tokenizer.
    
-   **RoPE** is applied to attention queries/keys instead of absolute positional embeddings.
    

### **2. Attention (with GQA + KV Cache)**

-   **Grouped Query Attention (GQA):**  
    Multiple query heads share a smaller number of key/value heads → faster and more memory-efficient.
    
-   **KV Caching:**  
    During inference, previous keys/values are stored so the model only attends to new tokens.
    

### **3. Transformer Blocks**

Each block contains:

-   RMSNorm
    
-   Multi-Head Attention (with RoPE, GQA, KV cache)
    
-   SwiGLU feed-forward network
    
-   Residual connections
    

### **4. Output Projection**

-   Final RMSNorm
    
-   Linear layer → logits → softmax for next-token prediction
    

### **5. Training Loop**

Modern GPU-friendly techniques:

-   **AMP mixed precision** for speed + memory efficiency
    
-   **Gradient accumulation** to simulate large batch sizes
    
-   **Cosine LR decay** for smooth convergence
    
-   **Gradient clipping** to prevent instability
    

----------

#  **Architecture ૮ - ﻌ • ა⁩**

**Model Structure**

```
Token Embedding
 + RoPE
      ↓
N × Transformer Blocks:
  RMSNorm
    → Multi-Head Self-Attention (GQA, KV Cache)
    → Residual
  RMSNorm
    → SwiGLU Feed-Forward
    → Residual
      ↓
Final RMSNorm
 → Linear LM Head
```


----------

#  **Technology Stack ૮ฅ・ﻌ・აฅ** 

-   **Language:** Python
    
-   **Framework:** PyTorch
    
-   **Built With:**
    
    -   Custom attention mechanisms
        
    -   Custom embeddings
        
    -   Custom RMSNorm + SwiGLU layers
        
    -   Mixed precision training tools
        
-   **Ecosystem Tools:**
    
    -   🤗 Hugging Face (datasets/tokenization)
        
    -    PyTorch (core autograd & tensor ops)
