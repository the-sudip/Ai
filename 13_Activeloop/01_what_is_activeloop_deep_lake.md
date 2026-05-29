# What is Activeloop & Deep Lake?

---

## Activeloop

**Activeloop** is an AI data infrastructure company that builds **Deep Lake** — an open-source, AI-native dataset format and vector store designed for storing, versioning, and querying multi-modal AI data at scale.

Website: [activeloop.ai](https://activeloop.ai)

---

## Deep Lake

**Deep Lake** is more than a vector database — it is a **tensor database** and **data lakehouse** that stores:

- Embeddings (vectors)
- Original data (text, images, video, audio, 3D point clouds)
- Labels and metadata
- All in a single, versioned, cloud-native format

```
Traditional Vector DB:        Deep Lake:
  [vector]                      [vector] + [text] + [image] + [label] + [metadata]
  [vector]                      [vector] + [text] + [image] + [label] + [metadata]
     ↓                                          ↓
Finds similar vectors           Finds similar vectors AND returns raw data alongside
```

---

## Why Deep Lake?

| Pain Point | Deep Lake Solution |
|---|---|
| Embeddings stored separately from source data | Stores everything together in one dataset |
| Re-embedding on every schema change | Versioning — roll back to any dataset version |
| Can't share datasets across teams | Cloud storage on S3, GCS, Azure, or Activeloop Hub |
| Batch ML training needs different format | Same dataset works for RAG + model training |
| Large datasets don't fit in memory | Streaming — iterate over billions of samples |

---

## Core Concepts

### Tensors

Deep Lake organizes data into **tensors** — typed, named columns in a dataset:

```python
import deeplake

ds = deeplake.Dataset("./my_dataset")

# Each tensor stores one type of data
ds.create_tensor("text", htype="text")           # text content
ds.create_tensor("embedding", htype="embedding", dtype="float32", dims=1536)
ds.create_tensor("metadata", htype="json")       # arbitrary metadata
ds.create_tensor("image", htype="image", sample_compression="jpeg")  # images
```

### Dataset Storage Backends

```
./local_path          → local filesystem
s3://bucket/path      → Amazon S3
gcs://bucket/path     → Google Cloud Storage
azure://container/path → Azure Blob Storage
hub://username/dataset → Activeloop Hub (managed cloud)
```

---

## Installation

```bash
pip install deeplake langchain-community
```

---

## Quick Start

```python
import deeplake
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake

embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Create a Deep Lake vector store
vectorstore = DeepLake(
    dataset_path="./my_deep_lake_store",
    embedding=embedder,
    overwrite=False,     # set True to recreate from scratch
)

# Add documents
from langchain_core.documents import Document

docs = [
    Document(page_content="Deep Lake stores multi-modal AI datasets.", metadata={"source": "docs"}),
    Document(page_content="LangChain integrates with many vector stores.", metadata={"source": "langchain"}),
    Document(page_content="RAG retrieves relevant documents before generation.", metadata={"source": "paper"}),
]

vectorstore.add_documents(docs)
print(f"Dataset stored at: ./my_deep_lake_store")
```

---

## Dataset Versioning

Deep Lake has **git-like versioning** built in:

```python
import deeplake

ds = deeplake.load("./my_dataset")

# Commit current state
ds.commit("Added Q1 2025 documents")

# List all versions
for commit in ds.commits:
    print(commit.id, commit.message, commit.timestamp)

# Checkout a previous version (read-only)
ds_v1 = deeplake.load("./my_dataset", token="...", read_only=True)
ds_v1.checkout("commit_id_here")

# Branch for experimentation
ds.checkout("experimental-branch", create=True)
# ... add data ...
ds.commit("Experimental embedding update")
ds.checkout("main")  # back to main branch
```

---

## Activeloop Hub — Shared Datasets

```python
# Upload to Activeloop Hub (free for public datasets)
import deeplake

# Log in
# activeloop login (CLI) or token in env var

ds = deeplake.Dataset("hub://your_username/my_rag_dataset")
# This dataset is now accessible by your whole team

# Load someone else's public dataset
public_ds = deeplake.load("hub://activeloop/twitter-airline-sentiment-train")
print(f"Samples: {len(public_ds)}")
```

---

## Deep Lake vs Other Vector Stores

| Feature | Chroma | Pinecone | Deep Lake |
|---|---|---|---|
| **Multi-modal storage** | No | No | Yes (text + image + video) |
| **Versioning** | No | No | Yes (git-like) |
| **Cloud backends** | No | Managed only | S3, GCS, Azure, Hub |
| **ML training support** | No | No | Yes (PyTorch/TF dataloader) |
| **Streaming** | No | No | Yes |
| **Open-source** | Yes | No | Yes |
| **Primary use** | RAG | RAG | RAG + training data |
