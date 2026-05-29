# Fine-tuning vs RAG vs Prompting

Three strategies to adapt LLMs to specific needs. Choosing the right one is a critical design decision.

---

## At a Glance

| Strategy | What changes | Cost | Speed to deploy | Best for |
|---|---|---|---|---|
| **Prompting** | Nothing — instructions in the request | API call cost only | Immediate | General tasks, quick iteration |
| **RAG** | External retrieval system (no model change) | Storage + retrieval infra | Hours–days | Private/fresh knowledge |
| **Fine-tuning** | Model weights are updated | High (GPU training) | Days–weeks | Style, format, domain embedding |

---

## 1. Prompting

Write better instructions — no infrastructure changes.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Persona + constraints via system prompt
messages = [
    SystemMessage(content="""
        You are a legal document reviewer. 
        - Respond only about contract law.
        - Always cite relevant clauses when possible.
        - If unsure, say "consult a lawyer."
    """),
    HumanMessage("What happens if a contractor misses a delivery deadline?"),
]
response = llm.invoke(messages)
```

**Pros:** Zero setup, instant iteration, no training cost.  
**Cons:** Knowledge limited to training cutoff, context window limits how much guidance you can inject.

---

## 2. RAG — Retrieval-Augmented Generation

Keep the model unchanged. At inference time, **retrieve relevant documents** from an external store and inject them into the prompt as context.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Build the knowledge base (offline)
vectorstore = Chroma.from_texts(
    texts=company_documents,  # your private docs
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Build the RAG chain (online)
prompt = ChatPromptTemplate.from_template("""
Answer based only on the provided context:

Context:
{context}

Question: {question}
""")

rag_chain = (
    {"context": retriever | (lambda docs: "\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

answer = rag_chain.invoke("What is our refund policy?")
```

**Pros:** Up-to-date knowledge, no retraining, cite sources, handles private data.  
**Cons:** Retrieval quality matters — bad retrieval = bad answers. Adds latency.

---

## 3. Fine-tuning

Train the model on **custom examples** to permanently adapt its behavior, style, or domain knowledge.

```python
# Conceptual — actual fine-tuning uses OpenAI API or training scripts

# Training data format (JSONL):
training_example = {
    "messages": [
        {"role": "system", "content": "You are a customer support agent for AcmeCorp."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password: go to Settings > Security > Reset Password..."},
    ]
}

# After fine-tuning, call the fine-tuned model:
llm = ChatOpenAI(model="ft:gpt-4o-mini:acmecorp:support-v1:abc123")
```

**Pros:** Baked-in knowledge, consistent style, potentially faster (smaller prompts), no retrieval needed.  
**Cons:** Expensive (GPU time), slow iteration, can forget general knowledge ("catastrophic forgetting"), static — must retrain when knowledge changes.

---

## Decision Framework

```
New information or private docs?
    ├── YES → use RAG
    └── NO → 
        Specific style/format behavior needed?
            ├── YES → Fine-tune (if data exists) or few-shot prompting
            └── NO → Just prompting
```

### Real-world combinations
Most production systems **combine** all three:
1. **Fine-tune** for correct tone and format
2. **RAG** for up-to-date knowledge
3. **Prompting** for runtime task-specific control

---

## Hallucination Comparison

| Strategy | Hallucination Risk |
|---|---|
| Prompting (no context) | High — model must rely on training data |
| RAG | Low — model is grounded in retrieved facts |
| Fine-tuning | Medium — memorized facts can become stale |
