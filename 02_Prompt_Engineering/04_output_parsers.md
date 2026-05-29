# Output Parsers

Output parsers transform the raw text response from an LLM into **structured Python objects** — strings, lists, JSON, Pydantic models, etc.

---

## Why Output Parsers?

An LLM always returns text. But your application needs structured data:

```python
# Raw LLM response:
"The recipe has 3 steps: 1. Boil water. 2. Add pasta. 3. Drain."

# What you actually need:
{"steps": ["Boil water", "Add pasta", "Drain"], "count": 3}
```

---

## 1. `StrOutputParser` — Most Common

Simply extracts `.content` from an `AIMessage`. Returns a plain string.

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "What is AI?"})
type(result)  # str
```

---

## 2. `JsonOutputParser` — Parse JSON

Forces the LLM to output valid JSON and parses it into a Python dict.

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Return only valid JSON."),
    ("human", "Give me info about Paris in JSON with keys: city, country, population"),
])

chain = prompt | llm | parser
result = chain.invoke({})
# {"city": "Paris", "country": "France", "population": 2161000}
type(result)  # dict
```

---

## 3. `PydanticOutputParser` — Validated Structured Output

Uses a Pydantic model to define and validate the exact output schema.

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    rating: int = Field(description="Rating from 1 to 10")
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")
    summary: str = Field(description="One-sentence summary")

parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic. {format_instructions}"),
    ("human", "Review the movie: {movie}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
review = chain.invoke({"movie": "Inception"})

print(review.title)    # "Inception"
print(review.rating)   # 9
print(review.pros)     # ["Mind-bending plot", "Great visuals", ...]
type(review)           # MovieReview (Pydantic model)
```

---

## 4. `.with_structured_output()` — Modern Recommended Approach

Directly tells the LLM to return structured output using its native function-calling mechanism (more reliable than prompting for JSON).

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's job")

# No need for format instructions in the prompt
structured_llm = llm.with_structured_output(Person)

result = structured_llm.invoke(
    "John Smith is a 35-year-old software engineer."
)
print(result.name)       # "John Smith"
print(result.age)        # 35
print(result.occupation) # "software engineer"
```

Also works with plain JSON schema:
```python
json_schema = {
    "title": "joke",
    "description": "A joke with setup and punchline",
    "type": "object",
    "properties": {
        "setup": {"type": "string"},
        "punchline": {"type": "string"},
    },
    "required": ["setup", "punchline"],
}
structured_llm = llm.with_structured_output(json_schema)
```

---

## 5. `CommaSeparatedListOutputParser`

Parses a comma-separated string into a Python list.

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
result = parser.parse("Python, JavaScript, Go, Rust, TypeScript")
# ["Python", "JavaScript", "Go", "Rust", "TypeScript"]
```

---

## 6. `EnumOutputParser`

Restricts output to a predefined set of values.

```python
from langchain.output_parsers import EnumOutputParser
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

parser = EnumOutputParser(enum=Sentiment)
result = parser.parse("positive")
# Sentiment.POSITIVE
```

---

## Comparison Table

| Parser | Output Type | Reliability | Use Case |
|---|---|---|---|
| `StrOutputParser` | `str` | ✅✅✅ | Any text response |
| `JsonOutputParser` | `dict` | ✅✅ | Flexible JSON |
| `PydanticOutputParser` | Pydantic model | ✅✅ | Validated schema |
| `.with_structured_output()` | Pydantic / dict | ✅✅✅ | Best structured output |
| `CommaSeparatedListOutputParser` | `list[str]` | ✅✅ | Simple lists |

---

## Handling Parser Errors

```python
from langchain.output_parsers import OutputFixingParser

base_parser = PydanticOutputParser(pydantic_object=MovieReview)

# If the LLM returns malformed output, this parser fixes it automatically
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
chain = prompt | llm | fixing_parser
```
