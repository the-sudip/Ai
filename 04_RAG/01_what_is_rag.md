# RAG — Retrieval-Augmented Generation

## What is RAG?

**RAG** is a technique that enhances LLM responses by **retrieving relevant documents** from an external knowledge base and injecting them into the prompt as context — grounding the model's answer in real facts.

---

## The Core Problem RAG Solves

| Problem | Without RAG | With RAG |
|---|---|---|
| Knowledge cutoff | Model can't answer about events after training | Retrieve up-to-date docs |
| Private knowledge | Model doesn't know internal company docs | Index and retrieve your own docs |
| Hallucination | Model makes up answers | Model cites retrieved facts |
| Source attribution | No way to cite sources | Return doc metadata (filename, page) |

---

## Two Phases

### Phase 1 — Indexing (Offline, run once)
```
Raw Documents
    ↓
[Document Loader] → List of Documents
    ↓
[Text Splitter] → Smaller Chunks
    ↓
[Embedding Model] → Vectors per Chunk
    ↓
[Vector Store] → Stored + Indexed
```

### Phase 2 — Retrieval & Generation (Online, per query)
```
User Query
    ↓
[Embedding Model] → Query Vector
    ↓
[Vector Store Similarity Search] → Top-K Relevant Chunks
    ↓
[LLM with Prompt: "Given this context, answer:"] → Answer
```

---

## Full RAG Implementation

```python
# ── INDEXING PHASE ──────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. Load
loader = PyPDFLoader("knowledge_base.pdf")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Embed + Store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db",
)

# ── RETRIEVAL & GENERATION PHASE ────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )

prompt = ChatPromptTemplate.from_template("""
You are an expert assistant. Answer the question based ONLY on the provided context.
If the answer cannot be found in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:""")

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the company's refund policy?")
print(answer)
```

---

## Returning Sources

```python
from langchain_core.runnables import RunnableParallel

rag_chain_with_sources = RunnableParallel({
    "answer": rag_chain,
    "context": retriever | format_docs,
})

result = rag_chain_with_sources.invoke("What is the refund policy?")
print(result["answer"])   # the generated answer
print(result["context"])  # the retrieved documents used
```

---

## Evaluating RAG Quality

Key metrics:
| Metric | What it measures |
|---|---|
| **Context Recall** | Did we retrieve all relevant documents? |
| **Context Precision** | Were retrieved documents all relevant? |
| **Answer Faithfulness** | Is the answer supported by retrieved context? |
| **Answer Relevance** | Does the answer address the question? |

Use **RAGAS** library for automated evaluation:
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(
    dataset=test_dataset,
    metrics=[faithfulness, answer_relevancy, context_recall],
)
```
