# PromptTemplate in LangChain

LangChain's prompt templates make prompts **reusable, composable, and type-safe** by separating the static structure from dynamic values.

---

## Why Use PromptTemplates?

Without templates:
```python
# Bad: hard to reuse, prone to errors
question = "transformers"
prompt = f"You are an expert in AI. Explain {question} simply."
```

With templates:
```python
# Good: reusable, validated, composable
template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}."),
    ("human", "Explain {topic} simply."),
])
prompt = template.invoke({"domain": "AI", "topic": "transformers"})
```

---

## Types of Prompt Templates

### 1. `PromptTemplate` — Simple String Template
```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Translate '{text}' from {source_lang} to {target_lang}."
)

formatted = template.invoke({
    "text": "Good morning",
    "source_lang": "English",
    "target_lang": "French",
})
print(formatted.text)
# "Translate 'Good morning' from English to French."
```

### 2. `ChatPromptTemplate` — Multi-Role Chat Template (Recommended)
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Always respond in {language}."),
    ("human", "{question}"),
])

messages = template.invoke({
    "role": "senior Python developer",
    "language": "English",
    "question": "What is a generator?",
})
```

### 3. `from_messages` with Tuples vs Message Objects
```python
# Tuple format (shorthand)
ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{query}"),
])

# Explicit message objects (identical result)
from langchain_core.messages import SystemMessage, HumanMessage
ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="{query}"),
])
```

---

## `MessagesPlaceholder` — Inject Dynamic Message Lists

Used to insert a **variable-length list of messages** — most commonly for chat history.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),   # injects full chat history here
    ("human", "{question}"),
])

# At runtime:
messages = template.invoke({
    "history": [
        HumanMessage("What is Python?"),
        AIMessage("Python is a high-level programming language..."),
    ],
    "question": "What about Java?",
})
```

---

## Partial Templates — Pre-fill Some Variables

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}."),
    ("human", "{question}"),
])

# Pre-fill the domain — now only need to pass question
python_expert = template.partial(domain="Python programming")

result = python_expert.invoke({"question": "What is a decorator?"})
```

---

## Template Input Variables

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{question}"),
])

print(template.input_variables)  # ['role', 'question']
```

---

## Using Templates in LCEL Chains

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class {domain} expert."),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

answer = chain.invoke({
    "domain": "machine learning",
    "question": "What is gradient descent?",
})
print(answer)
```

---

## FewShotChatMessagePromptTemplate

Automatically formats a list of examples into few-shot messages.

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "fast", "output": "slow"},
]

example_template = ChatPromptTemplate.from_messages([
    ("human", "What is the antonym of {input}?"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_template,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a vocabulary expert."),
    few_shot_prompt,  # insert examples
    ("human", "What is the antonym of {word}?"),
])

chain = final_prompt | llm | StrOutputParser()
result = chain.invoke({"word": "beautiful"})
# → "ugly"
```
