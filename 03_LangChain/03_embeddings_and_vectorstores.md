# Embeddings & Vector Stores

Embeddings and vector stores are the backbone of RAG (Retrieval-Augmented Generation) and semantic search.

---

## What are Embeddings?

An **embedding** is a dense numerical vector that represents the **semantic meaning** of text. Text with similar meaning → vectors that are close together in vector space.

```
"dog"     → [0.12, -0.34, 0.89, ...]   # 1536-dim vector
"puppy"   → [0.13, -0.31, 0.91, ...]   # very close to "dog"
"car"     → [0.78, 0.22, -0.45, ...]   # far from "dog"
```

---

## Embedding Models

```python
from langchain_openai import OpenAIEmbeddings

# OpenAI (recommended)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536-dim
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # 3072-dim, higher quality

# Embed a single query
vector = embeddings.embed_query("What is machine learning?")
print(len(vector))   # 1536

# Embed multiple documents
vectors = embeddings.embed_documents([
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
])
print(len(vectors))     # 2
print(len(vectors[0]))  # 1536
```

### Other Embedding Options
```python
# HuggingFace (free, local)
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Google
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Ollama (local)
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

---

## Similarity Metrics

| Metric | Formula | Use |
|---|---|---|
| **Cosine similarity** | cos(θ) = (A·B)/(|A||B|) | Most common; direction matters |
| **Dot product** | A·B | Fast; used when vectors are normalized |
| **Euclidean distance** | √Σ(aᵢ-bᵢ)² | Distance in space |

Most vector stores use cosine similarity by default.

---

## Vector Stores

A **vector store** stores embeddings and supports fast nearest-neighbor search.

### Chroma (Easy, Local)
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # persists to disk
)

# Load existing
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
```

### FAISS (Fast, In-Memory)
```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
vectorstore.save_local("./faiss_index")  # save

vectorstore = FAISS.load_local("./faiss_index", embeddings)  # load
```

### Pinecone (Cloud, Production-Scale)
```python
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index",
)
```

---

## Search Methods

```python
# 1. Similarity search — top-K most similar
results = vectorstore.similarity_search("What is RAG?", k=3)
for doc in results:
    print(doc.page_content)
    print(doc.metadata)

# 2. With score — includes cosine similarity score
results = vectorstore.similarity_search_with_score("What is RAG?", k=3)
for doc, score in results:
    print(f"Score: {score:.3f} | {doc.page_content[:100]}")

# 3. MMR — Maximal Marginal Relevance (diverse results)
results = vectorstore.max_marginal_relevance_search(
    "What is RAG?",
    k=3,
    fetch_k=20,  # fetch 20, then pick 3 diverse ones
)

# 4. With filter on metadata
results = vectorstore.similarity_search(
    "refund policy",
    k=3,
    filter={"source": "faq.pdf"},  # only search within this source
)
```

---

## Using as a Retriever

A **Retriever** is a standardized interface wrapping any search mechanism:

```python
# Basic retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# MMR retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20},
)

# With threshold (only return results above similarity score)
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 5},
)

# Invoke the retriever
docs = retriever.invoke("How does transformer attention work?")
```

---

## Vector Store Comparison

| Store | Type | Best For |
|---|---|---|
| **Chroma** | Local file | Development, small-medium projects |
| **FAISS** | In-memory | High-speed batch retrieval, no persistence needed |
| **Pinecone** | Cloud | Production, large scale |
| **Weaviate** | Cloud/self-hosted | Complex filtering + semantic search |
| **Qdrant** | Cloud/local | Filtered vector search at scale |
| **pgvector** | PostgreSQL extension | Existing PostgreSQL infrastructure |
