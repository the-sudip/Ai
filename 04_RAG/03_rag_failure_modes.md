# RAG Failure Modes & Debugging

Understanding what goes wrong in RAG systems is just as important as knowing how to build them.

---

## The RAG Failure Chain

```
User Query
    ↓
[Retrieval Failure] ← Top-K docs don't contain the answer
    OR
[Context Processing Failure] ← Context too long, poorly formatted
    OR
[Generation Failure] ← LLM ignores context, hallucinates anyway
    ↓
Bad Answer
```

---

## Failure Mode 1: Poor Retrieval (Most Common)

**Symptoms**: The answer is in the documents but the model says "I don't know."

**Causes & Fixes:**

| Cause | Fix |
|---|---|
| Chunk too large | Reduce `chunk_size` |
| Chunk too small | Increase `chunk_size` — answer spans a full paragraph |
| K too small | Increase `k` (retrieve more docs) |
| Query phrasing mismatch | Use HyDE or Multi-Query retrieval |
| Wrong embedding model | Try a domain-specific embedding model |
| Document not indexed | Check loader and splitter |

**Diagnostic:**
```python
# Check what's being retrieved
query = "What is our refund policy?"
docs = retriever.invoke(query)
for i, doc in enumerate(docs):
    print(f"\n--- Doc {i+1} ---")
    print(f"Source: {doc.metadata}")
    print(doc.page_content[:300])
```

---

## Failure Mode 2: Lost in the Middle

**Symptom**: Answer exists in retrieved docs but LLM ignores it.

**Cause**: LLMs attend best to the **beginning and end** of their context. Information in the middle of a long context window is often missed.

**Fix**: Re-rank retrieved docs so the most relevant appear first and last (not in the middle).

```python
def reorder_docs(docs: list) -> list:
    """Place most relevant docs at beginning and end."""
    if len(docs) <= 2:
        return docs
    # Put top doc first, second-best last, rest in middle
    reordered = [docs[0]] + docs[2:] + [docs[1]]
    return reordered
```

Or use LangChain's `LongContextReorder`:
```python
from langchain_community.document_transformers.long_context_reorder import LongContextReorder

reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(retrieved_docs)
```

---

## Failure Mode 3: Context Too Long

**Symptom**: Token limit exceeded error, or very slow/expensive responses.

**Fix:**
```python
# 1. Reduce k
retriever = vs.as_retriever(search_kwargs={"k": 2})  # fewer docs

# 2. Use contextual compression to trim irrelevant text
from langchain.retrievers.document_compressors import LLMChainExtractor
compressor = LLMChainExtractor.from_llm(llm)

# 3. Set max_tokens_per_doc
def truncate_docs(docs, max_chars=500):
    return [Document(page_content=d.page_content[:max_chars], metadata=d.metadata) for d in docs]
```

---

## Failure Mode 4: Hallucination Despite Context

**Symptom**: The model gives an answer that contradicts the retrieved documents.

**Cause**: The model's parametric knowledge overrides the provided context, especially when the context is poorly formatted or the system prompt doesn't emphasize "use ONLY the context."

**Fix:**
```python
prompt = ChatPromptTemplate.from_template("""
IMPORTANT: You MUST answer using ONLY the information in the Context section below.
Do NOT use your general knowledge. If the answer is not in the context, say exactly:
"I cannot find this information in the provided documents."

Context:
{context}

Question: {question}

Answer (based only on context above):""")
```

---

## Failure Mode 5: Semantic Drift

**Symptom**: Query is about "pricing" but results are about "printing" (similar embedding distance).

**Fix**: Use hybrid search (BM25 + semantic) to catch exact keyword matches.

---

## Failure Mode 6: Stale Knowledge

**Symptom**: Vector store has outdated documents.

**Fix**: Implement a re-indexing pipeline:
```python
import schedule

def reindex():
    # 1. Load fresh documents
    new_docs = load_fresh_documents()
    # 2. Delete old collection
    vectorstore.delete_collection()
    # 3. Re-index
    vectorstore = Chroma.from_documents(new_docs, embeddings)

schedule.every().day.at("02:00").do(reindex)
```

---

## RAG Debugging Checklist

```
□ Are documents loading correctly? (check doc count and content)
□ Are chunks the right size? (print sample chunks)
□ Is the embedding model appropriate for the domain?
□ Are the right documents being retrieved? (log retriever output)
□ Is K large enough?
□ Is the context formatted clearly in the prompt?
□ Does the system prompt strongly direct the model to use context?
□ Is the model's temperature low (0–0.2) for factual retrieval?
□ Are retrieved docs in a good order (most relevant first)?
```
