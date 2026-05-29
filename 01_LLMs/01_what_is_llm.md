# What is a Large Language Model (LLM)?

## Definition
A **Large Language Model (LLM)** is a deep neural network — almost always **Transformer-based** — trained on billions of tokens of text data. Its objective during training is simple: **predict the next token** given the preceding tokens. Through doing this at enormous scale, the model internalizes grammar, facts, reasoning patterns, coding conventions, and common sense.

---

## Why "Large"?
| Era | Parameters | Example |
|---|---|---|
| Early NLP models | Millions | Word2Vec, GloVe |
| BERT / GPT-2 | 100M – 1.5B | BERT-base, GPT-2 |
| Modern LLMs | 7B – 1T+ | GPT-4, Claude 3.5, LLaMA 3 |

Scale unlocks **emergent capabilities** — abilities not explicitly trained for, such as chain-of-thought reasoning, few-shot learning, and code generation.

---

## Popular LLM Families

| Model | Company | Notable Strengths |
|---|---|---|
| **GPT-4o** | OpenAI | Multimodal, strong reasoning, tool calling |
| **Claude 3.5 Sonnet** | Anthropic | 200k context, safety, long-document tasks |
| **Gemini 1.5 Pro** | Google | 1M context, multimodal |
| **LLaMA 3.1** | Meta | Open source, deployable locally |
| **Mistral / Mixtral** | Mistral AI | Efficient, open weights |
| **Qwen 2.5** | Alibaba | Strong multilingual |

---

## How Training Works (Simplified)

```
1. Collect massive text corpus (web, books, code, Wikipedia…)
2. Tokenize all text into token IDs
3. For each sequence, predict next token → compare to actual next token
4. Compute cross-entropy loss
5. Backpropagate, update model weights
6. Repeat billions of times
```

After **pre-training**, models are typically aligned with human preferences via:
- **SFT** (Supervised Fine-Tuning) — train on curated instruction-response pairs
- **RLHF** (Reinforcement Learning from Human Feedback) — rank responses, train reward model, optimize policy
- **DPO** (Direct Preference Optimization) — newer, simpler alternative to RLHF

---

## What LLMs Can Do

- Text generation, summarization, translation
- Question answering
- Code generation and debugging
- Reasoning and math (with CoT)
- Classification and extraction
- Conversational interaction
- Tool/function calling (structured outputs)

---

## What LLMs Cannot Do (Without Help)

| Limitation | Solution |
|---|---|
| No access to real-time data | RAG, web search tools |
| Knowledge cutoff date | RAG or fine-tuning |
| Hallucinations (confident wrong answers) | Grounding with RAG, verification chains |
| Cannot remember past conversations | Memory (chat history, vector stores) |
| Cannot take actions in the world | Agents + Tools |
| Limited context window | Chunking, summarization, long-context models |

---

## Using an LLM in LangChain

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    max_tokens=1024,
)

# Simple invocation
response = llm.invoke("What is a transformer neural network?")
print(response.content)

# With messages
messages = [
    SystemMessage(content="You are a concise AI tutor."),
    HumanMessage(content="Explain LLMs in 3 bullet points."),
]
response = llm.invoke(messages)
print(response.content)
```

---

## Key Takeaway
> An LLM is a statistical language model. It doesn't "know" things — it has learned probability distributions over tokens. Its power comes from the breadth of patterns it has seen during training at massive scale.
