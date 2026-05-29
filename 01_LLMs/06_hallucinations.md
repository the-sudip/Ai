# Hallucinations — What They Are & How to Reduce Them

---

## What is a Hallucination?

A hallucination is when an LLM generates a **confident, plausible-sounding response that is factually incorrect**. The model doesn't "know" it's wrong — it's generating statistically likely tokens given its training.

### Examples
- Fabricating citations or references that don't exist
- Stating incorrect historical facts with confidence
- Inventing function signatures or API details
- Making up names, dates, statistics

---

## Why Do Hallucinations Happen?

1. **Training objective**: The model is trained to predict likely next tokens, not to verify factual accuracy.
2. **Knowledge gaps**: If the training data didn't include a fact, the model fills in with plausible-sounding text.
3. **High temperature**: More randomness = more chance of drifting away from ground truth.
4. **Ambiguous prompts**: Vague questions let the model "guess" freely.
5. **Instruction-following pressure**: The model tries to give an answer even when it shouldn't.

---

## Mitigation Strategies

### 1. Use RAG — Ground Answers in Documents
```python
prompt = ChatPromptTemplate.from_template("""
You are a fact-based assistant. Answer ONLY based on the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
""")
```

### 2. Lower Temperature for Factual Tasks
```python
factual_llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

### 3. Chain-of-Thought Prompting
Forces explicit reasoning steps, making errors easier to catch.
```python
prompt = "Think step by step. First identify what is being asked, then reason carefully..."
```

### 4. Verification Chain
Have a second LLM call check the first answer.
```python
answer = answer_chain.invoke(question)
verified = verification_chain.invoke({
    "answer": answer,
    "context": retrieved_docs,
    "instruction": "Does this answer contradict any facts in the context? Reply YES or NO and explain."
})
```

### 5. Structured Output Constraints
Limit what the model can say by forcing a schema.
```python
class Answer(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    sources: list[str]

structured_llm = llm.with_structured_output(Answer)
```

### 6. Ask the Model to Admit Uncertainty
```python
system = """If you are not certain about a fact, explicitly say 
"I'm not certain about this" rather than guessing."""
```

### 7. Self-Consistency (Multiple Samples + Vote)
```python
answers = [chain.invoke(query) for _ in range(5)]  # 5 independent calls
from collections import Counter
final = Counter(answers).most_common(1)[0][0]
```

---

## Types of Hallucinations

| Type | Example |
|---|---|
| **Factual** | "The Eiffel Tower was built in 1850" (it was 1889) |
| **Fabricated reference** | Citing a paper that doesn't exist |
| **Contradictory** | Saying opposite things in the same response |
| **Instruction-ignoring** | Answering a question the user didn't ask |
| **Sycophantic** | Agreeing with a wrong claim from the user |

---

## Sycophancy
A special class of hallucination where the model agrees with the user even when wrong.

```python
# User: "Einstein invented the telephone, right?"
# Sycophantic response: "Yes, that's correct! Albert Einstein..."
# Correct response: "Actually, the telephone was invented by Alexander Graham Bell..."
```

**Mitigation**: "Do not agree with the user if they state incorrect facts. Always prioritize accuracy over agreement."
