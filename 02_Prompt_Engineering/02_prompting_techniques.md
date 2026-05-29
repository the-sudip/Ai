# Prompting Techniques

A collection of core prompting strategies ranked from simple to advanced.

---

## 1. Zero-Shot Prompting

Give the task directly with **no examples**. Relies entirely on the model's training.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")

# Sentiment classification — zero shot
response = llm.invoke([HumanMessage(
    "Classify the sentiment of this review as Positive, Negative, or Neutral:\n\n"
    "'The battery died after 2 hours. Extremely disappointed.'"
)])
# → "Negative"
```

**When to use:** Simple, well-defined tasks. Great starting point before adding complexity.

---

## 2. Few-Shot Prompting

Provide **2–10 input→output examples** before the actual query. The model learns the pattern from demonstrations.

```python
prompt = """Classify the sentiment as Positive, Negative, or Neutral.

Review: "Best phone I've ever owned!" → Positive
Review: "Stopped working after a week." → Negative
Review: "It's fine, nothing special." → Neutral
Review: "Amazing camera, but battery is poor." → ?"""

response = llm.invoke([HumanMessage(prompt)])
# → "Mixed" or "Negative" depending on model reasoning
```

**When to use:**
- Output format must be exact (specific labels, JSON structure)
- Task is nuanced and hard to describe in words
- Zero-shot gives inconsistent results

### Tips for good few-shot examples:
- Cover edge cases, not just easy cases
- Keep examples consistent in format
- Order matters — put hardest/most relevant examples last

---

## 3. Chain-of-Thought (CoT) Prompting

Add intermediate reasoning steps before the final answer. Dramatically improves performance on **math, logic, and multi-step reasoning**.

```python
# Without CoT (often wrong on hard problems)
prompt = "If I have 3 boxes with 12 eggs each, and I use 7 eggs for breakfast, how many eggs remain?"
# Model might answer: "29" (wrong)

# With CoT
prompt = """Solve step by step:
If I have 3 boxes with 12 eggs each, and I use 7 eggs for breakfast, how many eggs remain?

Step 1: Total eggs = 3 × 12 = 36
Step 2: Used eggs = 7
Step 3: Remaining = 36 - 7 = 29
Answer: 29 eggs.

Now solve: If I have 4 shelves with 15 books each, and I donate 22 books, how many remain?
"""
```

### Zero-Shot CoT
Just add "Let's think step by step." to the prompt — surprisingly effective!

```python
prompt = """Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 balls. How many tennis balls does he have now?
Let's think step by step."""
```

---

## 4. ReAct — Reason + Act

Used in **agents**. The model alternates between reasoning (Thought), calling a tool (Action), and reading results (Observation).

```
Thought: I need to find the current price of Bitcoin.
Action: web_search("Bitcoin current price USD")
Observation: Bitcoin is currently trading at $67,430.

Thought: I now have the answer.
Final Answer: Bitcoin is currently trading at approximately $67,430 USD.
```

```python
# In practice, the ReAct prompt is structured like this:
react_prompt = """You have access to the following tools:
{tools}

Use the following format:
Thought: What do I need to do?
Action: tool_name
Action Input: the input to the tool
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer.
Final Answer: [your answer]

Question: {input}
{agent_scratchpad}"""
```

---

## 5. Self-Consistency

Generate the same query **multiple times** with temperature > 0, then take a **majority vote**.

```python
from collections import Counter
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

question = "Is 9677 a prime number?"

# Generate 5 independent answers
answers = []
for _ in range(5):
    r = llm.invoke([HumanMessage(question)])
    answers.append(r.content.strip().lower())

# Majority vote
final = Counter(answers).most_common(1)[0][0]
print(final)
```

**Why it works:** Wrong answers tend to vary; correct answers converge. Improves accuracy on difficult reasoning tasks by ~10–20%.

---

## 6. Role Prompting

Assign a specific expert persona to improve domain-specific quality.

```python
system = "You are a senior security engineer with 15 years of experience in penetration testing and secure code review. You focus on OWASP Top 10 vulnerabilities."
```

---

## 7. Least-to-Most Prompting

Break a complex problem into simpler subproblems, solve each in sequence.

```python
# Step 1: Decompose
decompose_prompt = "Break this task into a list of simpler subtasks: {complex_task}"

# Step 2: Solve each subtask with its predecessor's answer
for subtask in subtasks:
    answer = chain.invoke({"subtask": subtask, "previous_results": results})
    results.append(answer)
```

---

## 8. Prompt Chaining

Connect multiple prompts where the output of one is the input to the next.

```python
# Chain: extract → analyze → summarize
extract_chain = extract_prompt | llm | StrOutputParser()
analyze_chain = analyze_prompt | llm | StrOutputParser()
summarize_chain = summarize_prompt | llm | StrOutputParser()

extracted = extract_chain.invoke({"text": raw_document})
analysis = analyze_chain.invoke({"extracted": extracted})
summary = summarize_chain.invoke({"analysis": analysis})
```

---

## Choosing the Right Technique

| Scenario | Technique |
|---|---|
| Simple factual questions | Zero-shot |
| Specific output format needed | Few-shot |
| Math, logic, multi-step reasoning | Chain-of-Thought |
| Needs to use external tools | ReAct (agents) |
| High accuracy critical, cost not a concern | Self-consistency |
| Complex multi-stage task | Prompt chaining / Least-to-most |
