# What is a Vector Database?

---

## The Core Problem

Traditional databases store and search **exact values** (SQL `WHERE name = 'Alice'`). They cannot answer questions like:

> "Find me documents *semantically similar* to this query."

A **vector database** stores and searches **embeddings** — dense numerical representations of meaning — enabling similarity-based retrieval.

---

## How Embeddings Work

An **embedding model** converts any content (text, image, audio) into a fixed-length vector of floats:

```python
from langchain_openai import OpenAIEmbeddings

embedder = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536 dimensions

# Same meaning → vectors are close together
v1 = embedder.embed_query("What is machine learning?")
v2 = embedder.embed_query("Explain ML to me")
v3 = embedder.embed_query("What is the capital of France?")

# v1 and v2 will be very close in vector space
# v3 will be far from v1 and v2
print(len(v1))  # 1536
```

---

## Vector Space Intuition

```
High-dimensional space (simplified to 2D):

         "dog" •    • "puppy"
                           
    "cat" •  • "kitten"
                              
                     • "Paris"
                • "France"
                                        
Semantically similar things cluster together.
```

---

## Similarity Metrics

| Metric | Formula | Best For |
|---|---|---|
| **Cosine Similarity** | $\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$ | Text (normalized embeddings) |
| **Dot Product** | $A \cdot B = \sum A_i B_i$ | When magnitude matters |
| **Euclidean Distance** | $\sqrt{\sum (A_i - B_i)^2}$ | Image embeddings, dense retrieval |

Most text embedding models use **cosine similarity** (vectors are unit-normalized).

---

## What a Vector Database Does

```
INDEXING (offline):
  Documents → Embedding Model → Vectors → Stored in Vector DB (with metadata)

QUERYING (online):
  Query text → Embedding Model → Query Vector
     → ANN Search in Vector DB (finds k nearest neighbors)
     → Returns top-k matching documents
```

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

embedder = OpenAIEmbeddings()

# Store documents
docs = [
    Document(page_content="Python is a general-purpose programming language.", metadata={"source": "wiki"}),
    Document(page_content="Neural networks learn from data.", metadata={"source": "textbook"}),
    Document(page_content="LangChain builds LLM applications.", metadata={"source": "docs"}),
]

vectorstore = Chroma.from_documents(docs, embedder)

# Query — finds semantically similar documents
results = vectorstore.similarity_search("how do deep learning models work?", k=2)
for doc in results:
    print(doc.page_content)
# → "Neural networks learn from data."  (most similar)
# → "Python is a general-purpose programming language."
```

---

## Vector DB vs Traditional DB

| Aspect | Traditional Database | Vector Database |
|---|---|---|
| **Storage** | Rows, columns, JSON | Dense vectors + metadata |
| **Search** | Exact match, B-tree index | Approximate nearest neighbor |
| **Query type** | `WHERE id = 5` | "Find most similar to X" |
| **Use case** | Transactions, lookups | Semantic search, RAG, recommendations |
| **Scalability** | Billions of rows | Billions of vectors |
| **Examples** | PostgreSQL, MySQL | Chroma, FAISS, Pinecone, Weaviate, Qdrant |

---

## Key Concepts

### Approximate Nearest Neighbor (ANN)
Exact nearest neighbor search in high dimensions is too slow (brute force = O(n·d)). ANN algorithms trade a small accuracy loss for massive speed gains:

- **HNSW** (Hierarchical Navigable Small World) — graph-based, used by Chroma, Weaviate, Qdrant
- **IVF** (Inverted File Index) — cluster-based, used by FAISS
- **LSH** (Locality-Sensitive Hashing) — hash-based, older approach

### Metadata Filtering
Every vector can have attached metadata, enabling hybrid search (vector similarity + metadata filter):

```python
# Find similar docs, but only from source="textbook"
results = vectorstore.similarity_search(
    query="neural networks",
    k=3,
    filter={"source": "textbook"}
)
```

### Dimensions
The size of the embedding vector. Common sizes:
- `text-embedding-3-small` → 1536 dims
- `text-embedding-3-large` → 3072 dims
- `text-embedding-ada-002` → 1536 dims
- `all-MiniLM-L6-v2` (HuggingFace) → 384 dims
