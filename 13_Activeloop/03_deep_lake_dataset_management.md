# Deep Lake for ML Training & Dataset Management

Deep Lake bridges the gap between RAG pipelines and ML model training — the same dataset powers both.

---

## Dataset Management with Version Control

```python
import deeplake

# Create and populate a dataset
ds = deeplake.Dataset("hub://your_username/training_data")

ds.create_tensor("text", htype="text")
ds.create_tensor("label", htype="class_label", class_names=["positive", "negative", "neutral"])
ds.create_tensor("embedding", htype="embedding", dims=1536)

with ds:
    ds.text.append("This product is amazing!")
    ds.label.append(0)  # positive
    # ...add more samples

# Version control
ds.commit("Initial dataset — 1000 samples")

# Add more data
with ds:
    ds.text.extend(["New sample 1", "New sample 2"])
    ds.label.extend([1, 2])

ds.commit("Added 500 more samples — v2")

# View commit history
for commit in ds.commits:
    print(f"{commit.id[:8]} | {commit.message} | {commit.timestamp}")

# Rollback to previous version
ds.checkout("first_commit_id")
```

---

## Branching for Experiments

```python
import deeplake

ds = deeplake.load("hub://your_username/training_data")

# Create experiment branch
ds.checkout("experiment/new-labels", create=True)

with ds:
    # Re-label some samples
    ds.label[0] = 2

ds.commit("Updated labels with new taxonomy")

# Compare branches
ds.checkout("main")   # back to main
# Merge if experiment was successful
ds.merge("experiment/new-labels")
```

---

## PyTorch DataLoader Integration

Deep Lake streams directly into PyTorch training loops:

```python
import deeplake
import torch
from torch.utils.data import DataLoader

ds = deeplake.load("hub://activeloop/mnist-train")

# Create a PyTorch dataloader — streams data from cloud
dataloader = ds.pytorch(
    batch_size=32,
    shuffle=True,
    tensors=["images", "labels"],   # only load needed tensors
    num_workers=4,
    transform={
        "images": lambda x: torch.tensor(x / 255.0, dtype=torch.float32),
        "labels": lambda x: torch.tensor(x, dtype=torch.long),
    },
)

# Standard PyTorch training loop
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for batch in dataloader:
        images = batch["images"]
        labels = batch["labels"]
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## TensorFlow / Keras Integration

```python
import deeplake

ds = deeplake.load("hub://your_username/image_dataset")

# Create TF dataset
tf_dataset = ds.tensorflow()

# Use with Keras
import tensorflow as tf

model.fit(
    tf_dataset.batch(32).prefetch(tf.data.AUTOTUNE),
    epochs=10,
)
```

---

## Streaming Large Datasets

Deep Lake streams data — you never need to download the entire dataset:

```python
import deeplake

# Even a 100GB dataset streams in chunks
ds = deeplake.load("hub://activeloop/laion400m-subset")
print(f"Total samples: {len(ds)}")  # millions of samples

# Iterate in streaming mode (no full download needed)
for i, sample in enumerate(ds):
    image = sample["image"].numpy()
    caption = sample["caption"].text()
    # process...
    
    if i >= 1000:  # only process first 1000
        break

# Create a streaming subset
subset = ds[0:10000]   # virtual slice — still streamed
dataloader = subset.pytorch(batch_size=64, shuffle=True)
```

---

## Dataset Querying with TQL

```python
import deeplake

ds = deeplake.load("hub://your_username/documents")

# Query with Tensor Query Language
# Returns a DatasetView (not a copy — virtual filter)
view = ds.query("SELECT * WHERE metadata['topic'] == 'AI' AND metadata['year'] >= 2024")
print(f"Matching samples: {len(view)}")

# Complex filters
view = ds.query("""
    SELECT * 
    WHERE contains(text, 'neural network') 
    AND metadata['language'] == 'en'
    ORDER BY metadata['date'] DESC
    LIMIT 100
""")

# View is lazy — data only loaded when accessed
for sample in view:
    print(sample["text"].text()[:100])
```

---

## Connecting Deep Lake to LangSmith for Evaluation

```python
from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Build RAG chain backed by Deep Lake
embedder = OpenAIEmbeddings()
vectorstore = DeepLake(
    dataset_path="hub://your_username/eval_dataset",
    embedding=embedder,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_template("""Answer from context only.
Context: {context}
Question: {question}""")

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Evaluate with a test set stored in Deep Lake
import deeplake

eval_ds = deeplake.load("hub://your_username/eval_questions")

correct = 0
for sample in eval_ds:
    question = sample["question"].text()
    expected = sample["expected_answer"].text()
    
    predicted = rag_chain.invoke(question)
    
    # Simple exact match (use RAGAS for proper eval)
    if expected.lower() in predicted.lower():
        correct += 1

print(f"Accuracy: {correct / len(eval_ds):.2%}")
```

---

## Summary — When to Use Deep Lake

| Scenario | Recommendation |
|---|---|
| Simple RAG prototype | Chroma or FAISS |
| RAG + need dataset versioning | **Deep Lake** |
| Share datasets across team | **Deep Lake Hub** |
| Same data for RAG + model training | **Deep Lake** |
| Multi-modal data (images + text) | **Deep Lake** |
| Billion-scale with serverless ops | Pinecone |
| Max query performance (Rust) | Qdrant |
