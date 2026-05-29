# Transformer Architecture

The **Transformer** is the neural network architecture underlying virtually all modern LLMs. Introduced in the 2017 paper *"Attention Is All You Need"* by Vaswani et al.

---

## High-Level Architecture

```
Input Text
    ↓
[Tokenizer] → Token IDs
    ↓
[Input Embedding] → Dense vectors per token
    ↓
[Positional Encoding] → Adds position information
    ↓
[N × Transformer Blocks]
    ├── Multi-Head Self-Attention
    ├── Add & Norm (residual connection)
    ├── Feed-Forward Network (FFN)
    └── Add & Norm (residual connection)
    ↓
[Output Linear Layer] → Logits over vocabulary
    ↓
[Softmax] → Probability distribution
    ↓
Sample next token
```

---

## 1. Input Embedding

Converts token IDs into dense vectors of dimension `d_model` (e.g., 768 for BERT-base, 12,288 for GPT-4).

```python
# Conceptually:
embedding_layer = nn.Embedding(vocab_size=50257, d_model=768)
token_ids = [9906, 11, 358]  # "Hello, I"
vectors = embedding_layer(token_ids)  # shape: (3, 768)
```

---

## 2. Positional Encoding

Transformers have no inherent notion of order (unlike RNNs). Positional encoding injects position information into each token vector.

- **Original Transformer**: Fixed sinusoidal encoding
- **Modern LLMs**: Learned positional embeddings, or **RoPE** (Rotary Position Embedding) for better long-context handling

---

## 3. Self-Attention (The Core)

Self-attention allows each token to "look at" every other token in the sequence and weight their importance.

For each token, compute three vectors:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"  
- **V (Value)**: "What will I contribute?"

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

**Why scale by √d_k?** — For large d_k, dot products grow very large, pushing softmax into saturation (near-zero gradients). Scaling stabilizes training.

---

## 4. Multi-Head Attention

Run `h` attention operations in parallel, each with different learned projections. Each head can focus on different types of relationships (syntax, coreference, semantics, etc.).

```
Input → [Head 1: syntactic patterns]
      → [Head 2: coreference]
      → [Head 3: semantic similarity]
      → ... (h heads total)
      → Concat all heads
      → Linear projection
      → Output
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
```

---

## 5. Feed-Forward Network (FFN)

Applied independently to each token position after attention. Usually 4× wider than `d_model`.

```python
# Per token:
FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
# Or with GELU (modern):
FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
```

---

## 6. Residual Connections & Layer Norm

Residual connections (skip connections) prevent vanishing gradients and allow training very deep networks.

```
output = LayerNorm(x + Attention(x))
output = LayerNorm(x + FFN(x))
```

---

## 7. Causal (Decoder-only) Masking

For **text generation** (GPT-style), the model must not "see the future." A causal mask prevents each token from attending to tokens that come after it.

```
Token 1 can attend to:  [1]
Token 2 can attend to:  [1, 2]
Token 3 can attend to:  [1, 2, 3]
...
```

This is why LLMs are **auto-regressive** — they generate one token at a time.

---

## Encoder vs Decoder vs Encoder-Decoder

| Type | Example | Use |
|---|---|---|
| **Encoder-only** | BERT | Classification, embeddings (reads the full context, bidirectional) |
| **Decoder-only** | GPT, LLaMA | Text generation (causal, left-to-right) |
| **Encoder-Decoder** | T5, BART | Translation, summarization (encoder reads input, decoder generates) |

Modern chat LLMs (GPT-4, Claude, LLaMA) are all **decoder-only**.

---

## Key Numbers for GPT-Scale Models

| Param | GPT-2 (small) | GPT-3 | GPT-4 (est.) |
|---|---|---|---|
| Parameters | 117M | 175B | ~1.8T (MoE est.) |
| Layers | 12 | 96 | Unknown |
| Attention heads | 12 | 96 | Unknown |
| d_model | 768 | 12,288 | Unknown |
| Context length | 1,024 | 4,096 | 128,000 |
