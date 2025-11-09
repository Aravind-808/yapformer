# Grouped Query Attention (GQA) 🧠

A beginner-friendly guide to understanding Grouped Query Attention in transformer models.

## What is Attention?

Think of attention as a way for a model to decide **what to focus on**. When reading "The cat sat on the mat", the word "sat" needs to pay attention to "cat" to know who's sitting.

In transformers, attention has three components:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What information do I have?"
- **Value (V)**: "What's the actual content?"

## Multi-Head Attention (MHA)

Traditional transformers use **Multi-Head Attention** - multiple attention mechanisms working in parallel.

With 8 heads, you get:
```
Head 1: Q₁, K₁, V₁
Head 2: Q₂, K₂, V₂
...
Head 8: Q₈, K₈, V₈
```

Each head learns to focus on different patterns (syntax, semantics, etc.).

**Problem**: During text generation, we store K and V for every past token in a cache. With 8 heads, that's a LOT of memory! 💾

## Enter Grouped Query Attention (GQA)

GQA is a clever optimization: **use fewer K/V heads than Q heads**.

Example with 8 Query heads and 2 K/V heads:
```
Queries: Q₁, Q₂, Q₃, Q₄, Q₅, Q₆, Q₇, Q₈  (8 heads)
Keys:    K₁, K₂                           (2 heads)
Values:  V₁, V₂                           (2 heads)
```

**Grouping**: 
- Q₁, Q₂, Q₃, Q₄ share K₁, V₁
- Q₅, Q₆, Q₇, Q₈ share K₂, V₂

## Why Does This Work?

Think of it like a library:
- **MHA**: 8 people, 8 different bookshelves (everyone has unique books)
- **GQA**: 8 people, 2 shared bookshelves (people share books, but each person still asks different questions)

You still get 8 different "perspectives" (queries), but they're looking at shared information (keys/values).

## Benefits

✅ **4x less memory** (in our 8→2 example)  
✅ **Faster inference** (less data to move around)  
✅ **Minimal quality loss** (research shows ~similar performance to MHA)  
✅ **Better than MQA** (Multi-Query Attention uses only 1 K/V head - too aggressive!)

## Where is GQA Used?

- **Llama 2** (8 Q heads, 1 K/V head per group)
- **Mistral** and other modern LLMs
- Any application where long-context generation matters

## Code Overview

The key implementation detail:

```python
# Project to fewer K/V heads
self.W_k = nn.Linear(d_model, num_kv_heads * head_dim)  # Only 2 heads
self.W_v = nn.Linear(d_model, num_kv_heads * head_dim)  # Only 2 heads

# Expand K/V to match Q heads before attention
K = K.repeat_interleave(group_size, dim=1)  # 2 heads → 8 heads (copied)
V = V.repeat_interleave(group_size, dim=1)
```

## Quick Comparison

| Method | Q Heads | K/V Heads | Memory | Quality |
|--------|---------|-----------|--------|---------|
| MHA    | 8       | 8         | High   | Best    |
| GQA    | 8       | 2         | Medium | ~Same   |
| MQA    | 8       | 1         | Low    | Okay    |

## When to Use GQA?

Use GQA when:
- Building chatbots or assistants (long conversations)
- Working with limited GPU memory
- Deploying models at scale
- You want a sweet spot between performance and efficiency

---

**TL;DR**: GQA lets multiple query heads share the same key/value heads, dramatically reducing memory while maintaining quality. It's the secret sauce behind efficient modern LLMs! 🚀