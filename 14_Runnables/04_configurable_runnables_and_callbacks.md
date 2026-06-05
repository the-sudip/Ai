# Configurable Runnables

LangChain lets you build chains where certain parameters can be swapped at runtime — without rebuilding the chain.

---

## `configurable_fields` — Runtime Parameter Overrides

Make specific fields of a Runnable configurable at call time:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Mark temperature and model as configurable
llm = ChatOpenAI(model="gpt-4o", temperature=0.7).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="Controls randomness. 0 = deterministic, 2 = very random.",
    ),
    model_name=ConfigurableField(
        id="llm_model",
        name="LLM Model",
        description="The OpenAI model to use.",
    ),
)

chain = ChatPromptTemplate.from_template("Write a poem about {topic}") | llm | StrOutputParser()

# Default settings
result = chain.invoke({"topic": "autumn"})

# Override at runtime
result = chain.invoke(
    {"topic": "autumn"},
    config={"configurable": {"llm_temperature": 1.5, "llm_model": "gpt-4o-mini"}},
)
```

---

## `configurable_alternatives` — Swap Entire Components

Swap out entire Runnables at runtime:

```python
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

llm = ChatOpenAI(model="gpt-4o").configurable_alternatives(
    which=ConfigurableField(
        id="llm_provider",
        name="LLM Provider",
        description="Which LLM provider to use",
    ),
    default_key="openai",
    anthropic=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    openai_mini=ChatOpenAI(model="gpt-4o-mini"),
)

chain = prompt | llm | parser

# Use default (OpenAI GPT-4o)
result = chain.invoke({"question": "What is LangChain?"})

# Switch to Anthropic at runtime
result = chain.invoke(
    {"question": "What is LangChain?"},
    config={"configurable": {"llm_provider": "anthropic"}},
)

# Use mini model
result = chain.invoke(
    {"question": "What is LangChain?"},
    config={"configurable": {"llm_provider": "openai_mini"}},
)
```

---

## `RunnableConfig` — Full Config Reference

```python
from langchain_core.runnables import RunnableConfig

config: RunnableConfig = {
    # ── Identification ──────────────────────────────────────────────
    "run_name": "my_rag_chain",          # shown in LangSmith traces
    "run_id": "uuid-here",              # custom run ID for correlation
    "tags": ["production", "rag"],      # filter traces in LangSmith
    "metadata": {                        # arbitrary key-value data
        "user_id": "user_123",
        "session_id": "sess_abc",
        "feature_flag": "v2",
    },
    
    # ── Execution control ────────────────────────────────────────────
    "max_concurrency": 10,              # max parallel calls in .batch()
    "recursion_limit": 25,             # for LangGraph graphs
    
    # ── Callbacks ───────────────────────────────────────────────────
    "callbacks": [my_callback_handler], # see below
    
    # ── Configurable values ──────────────────────────────────────────
    "configurable": {
        "llm_temperature": 0.5,         # overrides configurable_fields
        "thread_id": "chat_session_1",  # for LangGraph memory
    },
}

result = chain.invoke({"question": "..."}, config=config)
```

---

## Callbacks

Callbacks let you hook into every event in a chain's execution:

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any

class LoggingCallback(BaseCallbackHandler):
    """Logs every LLM call and result."""
    
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        print(f"\n🚀 LLM Start | Model: {serialized.get('name', 'unknown')}")
        for i, prompt in enumerate(prompts):
            print(f"  Prompt {i}: {prompt[:100]}...")
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        for gen in response.generations[0]:
            print(f"✅ LLM End | Output: {gen.text[:100]}...")
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        print(f"❌ LLM Error: {error}")
    
    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs) -> None:
        print(f"⛓️ Chain Start: {serialized.get('name', 'unknown')}")
    
    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        print(f"⛓️ Chain End")
    
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        print(f"🔧 Tool Start: {serialized.get('name')} | Input: {input_str[:80]}")
    
    def on_tool_end(self, output: Any, **kwargs) -> None:
        print(f"🔧 Tool End | Output: {str(output)[:80]}")

# Use the callback
logger = LoggingCallback()
result = chain.invoke({"question": "What is RAG?"}, config={"callbacks": [logger]})
```

### Token Usage Callback

```python
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"question": "Explain LangChain"})
    
print(f"Total tokens: {cb.total_tokens}")
print(f"Prompt tokens: {cb.prompt_tokens}")
print(f"Completion tokens: {cb.completion_tokens}")
print(f"Cost: ${cb.total_cost:.6f}")
```

---

## Binding Config to a Chain

```python
# Permanently attach config to a chain
production_chain = chain.with_config(
    run_name="prod_rag_chain",
    tags=["production", "v3.1"],
    metadata={"environment": "production"},
)

# Now every invocation inherits this config
result = production_chain.invoke({"question": "..."})
# Logged in LangSmith under "prod_rag_chain" with tags ["production", "v3.1"]

# Can still override per-call
result = production_chain.invoke(
    {"question": "..."},
    config={"metadata": {"user_id": "u_456"}},  # merged with base config
)
```

---

## Per-User Configuration Pattern

```python
from langchain_core.runnables import ConfigurableFieldSpec

# Build a chain configurable per user
llm = ChatOpenAI(model="gpt-4o").configurable_fields(
    temperature=ConfigurableFieldSpec(
        id="user_temperature",
        annotation=float,
        default=0.7,
        description="User's preferred creativity level",
    )
)

chain = system_prompt | llm | parser

def run_for_user(user_id: str, question: str, creativity: float = 0.7) -> str:
    return chain.invoke(
        {"question": question},
        config={
            "configurable": {"user_temperature": creativity},
            "metadata": {"user_id": user_id},
            "tags": [f"user:{user_id}"],
        }
    )

result = run_for_user("user_123", "Write a story", creativity=1.2)
```
