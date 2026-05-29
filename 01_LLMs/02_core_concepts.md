# Core Concepts of LLM Inference

These are the fundamental parameters and concepts you control when calling an LLM.

---

## 1. Tokens

A **token** is the atomic unit of text that the LLM processes. It is NOT always a word.

```
"Hello, world!"          → ["Hello", ",", " world", "!"]          (4 tokens)
"unbelievable"           → ["un", "bel", "iev", "able"]            (4 tokens via BPE)
"AI"                     → ["AI"]                                   (1 token)
"supercalifragilistic"   → ["super", "cal", "if", "rag", "il", "istic"]
```

**Rough rule of thumb**: 1 token ≈ 0.75 English words, or ~4 characters.

Tokenizers like **BPE (Byte-Pair Encoding)** build a vocabulary of ~50,000–100,000 tokens from training data, merging frequent byte pairs iteratively.

```python
import tiktoken  # OpenAI's tokenizer library

enc = tiktoken.encoding_for_model("gpt-4o")
tokens = enc.encode("Hello, I love LangChain!")
print(tokens)       # [9906, 11, 358, 3021, 95890, 8581, 0]
print(len(tokens))  # 7 tokens
```

---

## 2. Context Window

The **context window** is the maximum number of tokens the model can process in a single call — input + output combined.

| Model | Context Window |
|---|---|
| GPT-3.5-turbo | 16,384 tokens |
| GPT-4o | 128,000 tokens |
| Claude 3.5 Sonnet | 200,000 tokens |
| Gemini 1.5 Pro | 1,000,000 tokens |
| LLaMA 3.1 70B | 128,000 tokens |

**Why it matters:**
- Limits how much history, context, and documents you can send
- The model can attend to everything within its window equally
- Exceeding it causes errors or truncation

---

## 3. Temperature

Temperature controls the **randomness** of token sampling by scaling logits before the softmax.

$$P'(token_i) = \text{softmax}\left(\frac{logit_i}{T}\right)$$

| Temperature | Behavior | Use Case |
|---|---|---|
| `0.0` | Always picks highest-probability token (greedy) | Factual Q&A, code generation |
| `0.1–0.3` | Near-deterministic, slight variation | Summarization, extraction |
| `0.7` | Balanced creativity (default) | General assistants |
| `1.0+` | High creativity, possible incoherence | Creative writing, brainstorming |

```python
from langchain_openai import ChatOpenAI

factual_llm = ChatOpenAI(model="gpt-4o", temperature=0)
creative_llm = ChatOpenAI(model="gpt-4o", temperature=0.9)
```

---

## 4. Top-p (Nucleus Sampling)

Restricts sampling to the **smallest set of tokens** whose cumulative probability ≥ `p`. Cuts off the long tail of improbable tokens.

- `top_p=1.0` → sample from all tokens (no restriction)
- `top_p=0.9` → sample from top 90% probability mass

```python
llm = ChatOpenAI(model="gpt-4o", top_p=0.9)
```

> **Best practice**: Use either temperature OR top_p, not both simultaneously.

---

## 5. Max Tokens

Caps the **length of the model's response**. Does not affect the input.

```python
llm = ChatOpenAI(model="gpt-4o", max_tokens=500)
# Response will be cut off at 500 tokens even if incomplete
```

Set this appropriately:
- Too low → truncated answers
- Too high → wastes money and latency on short answers

---

## 6. Frequency Penalty & Presence Penalty (OpenAI)

| Parameter | Effect |
|---|---|
| `frequency_penalty` (0-2) | Penalizes tokens proportional to how often they've already appeared. Reduces repetition. |
| `presence_penalty` (0-2) | Flat penalty for any token that has appeared at all. Encourages new topics. |

```python
llm = ChatOpenAI(
    model="gpt-4o",
    model_kwargs={"frequency_penalty": 0.5, "presence_penalty": 0.3}
)
```

---

## 7. Stop Sequences

Tell the model to stop generating when it produces a specific string.

```python
llm = ChatOpenAI(
    model="gpt-4o",
    stop=["###", "END", "\n\n"]  # stop at any of these
)
```

Useful for structured generation where you know the output format.

---

## 8. Seed (Reproducibility)

Setting a seed makes the output more reproducible (same input + same seed → same output).

```python
llm = ChatOpenAI(model="gpt-4o", seed=42)
```

---

## Complete Example

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=256,
    seed=42,
)

response = llm.invoke([HumanMessage(content="What is backpropagation?")])
print(f"Content: {response.content}")
print(f"Tokens used: {response.usage_metadata}")
# {'input_tokens': 14, 'output_tokens': 98, 'total_tokens': 112}
```
