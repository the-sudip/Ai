# Defining & Using Tools

Tools are functions that give agents the ability to interact with the world beyond the LLM's parametric knowledge.

---

## What is a Tool?

A **tool** is a Python function wrapped with metadata (name, description, argument schema) so the LLM knows:
- **What it does** (description)
- **When to use it** (description)
- **What arguments it needs** (schema)
- **What it returns** (return type)

The quality of the **description** is critical — the LLM decides which tool to call based purely on the name and description.

---

## Method 1: `@tool` Decorator (Simplest)

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    Use this when the user asks about weather, temperature, forecast, or climate conditions.
    
    Args:
        city: The name of the city (e.g., 'London', 'Tokyo', 'New York')
    
    Returns:
        Current weather conditions as a string
    """
    weather_data = {
        "London": "18°C, partly cloudy",
        "Tokyo": "25°C, sunny",
        "New York": "12°C, rainy",
    }
    return weather_data.get(city, f"No weather data available for {city}")

# Tool metadata extracted automatically
print(get_weather.name)         # "get_weather"
print(get_weather.description)  # the docstring
print(get_weather.args)         # {"city": {"type": "string", "description": "..."}}

# Call it like a normal function
result = get_weather.invoke({"city": "London"})
print(result)  # "18°C, partly cloudy"
```

---

## Method 2: `@tool` with Pydantic Args (Type Safety)

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="The city name to get weather for")
    units: str = Field(default="celsius", description="Temperature units: 'celsius' or 'fahrenheit'")

@tool(args_schema=WeatherInput)
def get_weather_typed(city: str, units: str = "celsius") -> str:
    """Get current weather for a city with specified temperature units."""
    temp = 18  # mock
    if units == "fahrenheit":
        temp = temp * 9/5 + 32
    return f"{temp}°{'C' if units == 'celsius' else 'F'}, cloudy"
```

---

## Method 3: `StructuredTool` (Maximum Control)

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Max number of results to return (1-10)")

def web_search(query: str, max_results: int = 5) -> str:
    """Performs web search."""
    # Call actual search API here
    return f"Top {max_results} results for '{query}': [result1, result2, ...]"

search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description="Search the internet for current information about any topic. Use for recent events, facts, data.",
    args_schema=SearchInput,
    return_direct=False,  # True = bypass LLM and return result directly to user
)
```

---

## Method 4: `BaseTool` Class (Full Customization)

```python
from langchain_core.tools import BaseTool
from typing import Any

class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = "Query the company database for customer records, orders, or product information."
    
    # Custom attributes (must declare in the class)
    connection_string: str = ""
    
    def _run(self, query: str) -> str:
        """Execute a database query."""
        # Real DB query here
        return f"Query result: {query}"
    
    async def _arun(self, query: str) -> str:
        """Async version."""
        return self._run(query)

db_tool = DatabaseQueryTool(connection_string="postgresql://...")
```

---

## Binding Tools to an LLM

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")
tools = [get_weather, search_tool, db_tool]

# Bind tools — enables the LLM to call them
llm_with_tools = llm.bind_tools(tools)

response = llm_with_tools.invoke([HumanMessage("What's the weather in Tokyo?")])

# Check if LLM wants to call a tool
if response.tool_calls:
    print(response.tool_calls)
    # [{"name": "get_weather", "args": {"city": "Tokyo"}, "id": "call_abc123"}]
else:
    print(response.content)  # LLM answered without tools
```

---

## Executing Tool Calls

```python
from langchain_core.messages import ToolMessage

tool_map = {t.name: t for t in tools}

def execute_tools(response):
    """Execute all tool calls from an LLM response."""
    tool_messages = []
    for tool_call in response.tool_calls:
        tool = tool_map[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"],   # must match the tool_call id
        ))
    return tool_messages
```

---

## Tool Design Best Practices

```python
@tool
def good_tool_example(
    customer_id: str,
    include_history: bool = False
) -> str:
    """
    Look up customer information from the database.
    
    Use this tool when you need to find:
    - Customer contact details
    - Customer account status  
    - Customer order history (set include_history=True)
    
    Do NOT use for:
    - Financial transactions
    - Updating customer data
    
    Args:
        customer_id: The unique customer identifier (format: CUST-XXXXX)
        include_history: Whether to include order history (default: False)
    
    Returns:
        JSON string with customer data
    """
    ...
```

**Rules for good tools:**
1. **Specific name** — `get_customer_info` not `get_info`
2. **Clear description** — when to use AND when not to use
3. **Narrow scope** — one tool = one responsibility
4. **Error as string** — return errors as strings, don't raise exceptions
5. **Safe defaults** — optional args should have sensible defaults
6. **Validated input** — use Pydantic for complex argument validation

---

## Error Handling in Tools

```python
@tool
def safe_calculator(expression: str) -> str:
    """
    Evaluate a safe mathematical expression.
    Supports: +, -, *, /, **, sqrt, abs
    Example: '2 ** 10', 'abs(-5)', '(3 + 4) * 2'
    """
    import ast
    import math
    
    # Allowlist of safe operations only
    allowed_names = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
    }
    
    try:
        # Parse AST to ensure no unsafe operations
        tree = ast.parse(expression, mode='eval')
        result = eval(compile(tree, '<string>', 'eval'), {"__builtins__": {}}, allowed_names)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}. Please check the syntax."
    # ↑ Return errors as strings so the agent can understand and retry
```
