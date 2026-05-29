# Indexing Strategies & ANN Algorithms

Understanding how vectors are indexed determines the trade-off between search speed, accuracy, and memory.

---

## The Scalability Problem

Brute-force (exact) nearest neighbor search:
- Compare query vector against every stored vector
- Time: **O(n × d)** where n = number of vectors, d = dimensions
- At 1M vectors × 1536 dims: ~1.5B operations per query — too slow

**Approximate Nearest Neighbor (ANN)** sacrifices a tiny bit of accuracy for massive speed gains.

---

## HNSW — Hierarchical Navigable Small World

The most popular ANN algorithm. Used by **Chroma, Qdrant, Weaviate, FAISS**.

### How it Works

HNSW builds a **multi-layer graph** of vectors:

```
Layer 2 (few nodes, long-range connections):
    A ─────────────────── E

Layer 1 (medium density):
    A ──── B ──── C ──── E
               │
    D ─────────┘

Layer 0 (all nodes, short-range):
    A─B─C─D─E─F─G─H─I─J  (all vectors connected to neighbors)
```

**Search algorithm:**
1. Enter at top layer (few nodes) — fast traversal
2. Greedily move toward query vector
3. Drop down to next layer, repeat until Layer 0
4. Return k nearest neighbors found at Layer 0

### HNSW Parameters

| Parameter | Meaning | Default | Effect |
|---|---|---|---|
| `M` | Max connections per node | 16 | Higher = better recall, more memory |
| `ef_construction` | Search width during build | 200 | Higher = better quality index, slower build |
| `ef` | Search width during query | 50 | Higher = better recall, slower query |

```python
# Setting HNSW params in Chroma
import chromadb

client = chromadb.PersistentClient("./db")
collection = client.create_collection(
    "my_docs",
    metadata={
        "hnsw:space": "cosine",       # distance metric
        "hnsw:M": 32,                  # more connections
        "hnsw:ef_construction": 200,   # build quality
        "hnsw:ef": 100,                # query recall
    }
)

# Setting HNSW params in Qdrant
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff

client.create_collection(
    "my_docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
)
```

---

## IVF — Inverted File Index

Cluster-based approach. Used heavily in **FAISS**.

### How it Works

1. **Training**: Cluster all vectors into `nlist` Voronoi cells (k-means)
2. **Indexing**: Assign each vector to its nearest cluster centroid
3. **Searching**: 
   - Find the `nprobe` nearest centroids to query
   - Search only the vectors in those clusters

```
Before search:
Query → find top-nprobe centroids (fast)
       → search only those clusters (small subset)

Trade-off: nprobe controls recall vs speed
```

```python
import faiss
import numpy as np

dim = 1536
n_clusters = 100    # number of Voronoi cells
n_probe = 10        # how many clusters to search at query time

quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)

# Must train before adding vectors
training_data = np.random.rand(10_000, dim).astype("float32")
index.train(training_data)
index.add(training_data)

# At query time
index.nprobe = n_probe  # search 10/100 clusters
query = np.random.rand(1, dim).astype("float32")
distances, indices = index.search(query, k=5)
```

### IVF Variants in FAISS

| Index | Storage | Speed | Accuracy |
|---|---|---|---|
| `IndexFlatL2` | Full vectors | Slowest (exact) | 100% recall |
| `IndexIVFFlat` | Full vectors in clusters | Fast | ~95%+ recall |
| `IndexIVFPQ` | Compressed (product quantization) | Fastest | ~90% recall |
| `IndexHNSWFlat` | Graph | Very fast | ~99% recall |

---

## Product Quantization (PQ) — Memory Compression

PQ compresses vectors to save RAM — crucial at billion-scale:

```python
# IVF + PQ: fast AND memory-efficient
m = 32          # number of subvectors (must divide dim evenly)
bits = 8        # bits per subvector code (256 centroids per subvector)

quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFPQ(quantizer, dim, n_clusters, m, bits)

# Memory: original = 1536 × 4 bytes = 6KB per vector
# With PQ:  32 × 1 byte = 32 bytes per vector  ← 187x compression!
```

---

## Scalar Quantization (SQ)

Simpler compression — quantize float32 to int8:

```python
index = faiss.IndexIVFScalarQuantizer(
    quantizer, dim, n_clusters,
    faiss.ScalarQuantizer.QT_8bit  # float32 → int8 (4x compression)
)
```

---

## Recall vs Speed Trade-off

```
RECALL (% of true nearest neighbors found)
100% │ FlatL2 (exact)
 99% │ HNSW (ef=200)
 97% │ HNSW (ef=100)
 95% │ IVF (nprobe=20)
 90% │ IVF+PQ
     └──────────────────────► QUERIES PER SECOND
       slow          fast
```

**Rule of thumb for production:**
- Recall > 95% is usually sufficient for RAG
- HNSW is the go-to algorithm for most use cases
- Use IVF+PQ when you have hundreds of millions of vectors and limited RAM

---

## Choosing an Index

```
< 100K vectors:     → Flat (exact search, always correct)
100K–10M vectors:   → HNSW (best recall/speed balance)
> 10M vectors:      → IVF or HNSW with tuning
RAM constrained:    → IVF+PQ (compressed storage)
GPU available:      → faiss-gpu (10–100x faster)
```

---

## Evaluating Your Index

```python
# Measure recall@k against ground truth
def recall_at_k(true_indices, predicted_indices, k):
    hits = sum(
        1 for true, pred in zip(true_indices, predicted_indices)
        if true in pred[:k]
    )
    return hits / len(true_indices)

# Benchmark query speed
import time

start = time.time()
for query in test_queries:
    index.search(query, k=5)
elapsed = time.time() - start

qps = len(test_queries) / elapsed
print(f"Queries per second: {qps:.1f}")
```
