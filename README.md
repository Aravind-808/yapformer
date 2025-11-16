# Recreating a transformer from scratch (again) but adding stuff like
- KV caching
- Rotary Embeddings (RoPE)
- Grouped Query Attention
- RMSNorm
- SwiGLU

This time around, Ive also added some training optimizations like 
- Mixed Precision Training
- Gradient Accumulation
- Cosine Decay
- Gradient Clipping

I trained the final version of this model for 15,000 steps (I used steps and not epochs since the gradient updates were diminishing after 12000 steps).
It took me around 4 and half hours in total to finish training. 

The results are really impressive (to me, atleast), considering that this is only a ~56 million parameter model trained on very low data for such a short time.

Some remarkable examples:
![alt text](image.png)
![alt text](image-1.png)