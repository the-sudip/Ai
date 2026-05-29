# Tool-Calling Agent (Modern Approach)

The **tool-calling agent** is the modern, recommended approach for building agents with LLMs that support native function/tool calling (GPT-4, Claude 3, Gemini 1.5, etc.).

---

## Why Tool-Calling Over ReAct?

| | ReAct (text-based) | Tool-Calling (native API) |
|---|---|---|
| Mechanism | Model writes `Action: tool_name\nAction Input: args` as text | Model returns structured `tool_calls` JSON |
| Parsing errors | Common (model can misformat) | Rare (structured output) |
| Argument types | Everything is strings | Proper types (int, bool, list) |
| Multiple calls | One at a time | Can call multiple tools in parallel |
| Reliability | ~85% | ~99% |

---

## Tool Calling Flow

```
1. LLM receives: messages + list of tool schemas
2. LLM decides: "I need to call get_weather(city='London')"
3. LLM returns: AIMessage with tool_calls=[{name, args, id}]
4. Application executes the tool
5. Result returned as ToolMessage(content=result, tool_call_id=id)
6. Messages history now includes AIMessage + ToolMessage
7. LLM is called again with updated history
8. LLM sees the result and decides: done or call another tool
```

---

## Creating a Tool-Calling Agent

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4o", temperature=0)

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {"London": "18°C, cloudy", "Tokyo": "25°C, sunny"}
    return weather_data.get(city, "Weather data not available")

@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    populations = {"London": "9.0 million", "Tokyo": "13.96 million"}
    return populations.get(city, "Data not available")

tools = [get_weather, get_population]

# Prompt must include MessagesPlaceholder for "agent_scratchpad"
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to weather and population data."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # REQUIRED for tool-calling agent
])

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create executor
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
)

result = executor.invoke({"input": "What's the weather and population of Tokyo?"})
print(result["output"])
```

---

## Parallel Tool Calling

Modern LLMs can call multiple tools simultaneously in a single response:

```python
result = executor.invoke({
    "input": "Compare the weather and population of London and Tokyo"
})

# LLM might call all 4 tools in parallel:
# get_weather("London"), get_weather("Tokyo"),
# get_population("London"), get_population("Tokyo")
# All at once → much faster than sequential calls
```

---

## Direct Tool Binding (Without AgentExecutor)

For more control, you can manage the loop manually:

```python
from langchain_core.messages import HumanMessage, ToolMessage

llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

messages = [HumanMessage("What is the weather in London?")]

while True:
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Check if done
    if not response.tool_calls:
        print("Final Answer:", response.content)
        break

    # Execute tools
    for tc in response.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tc["id"],
        ))
```

---

## Streaming with Tool-Calling Agent

```python
for chunk in executor.stream({"input": "Find the weather in Paris"}):
    if "output" in chunk:
        print(chunk["output"])
    if "steps" in chunk:
        for step in chunk["steps"]:
            print(f"Tool used: {step.action.tool}")
            print(f"Tool result: {step.observation}")
```

---

## Using `create_react_agent` from LangGraph (Recommended)

LangGraph's prebuilt `create_react_agent` is now the preferred way to create tool-calling agents:

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=tools,
    state_modifier="You are a helpful assistant.",  # system prompt
)

result = agent.invoke({
    "messages": [HumanMessage("What's the weather in Tokyo?")]
})
print(result["messages"][-1].content)
```

Benefits over LangChain's `AgentExecutor`:
- Built on LangGraph — natively supports memory, HITL, streaming
- More transparent and debuggable
- Extensible — add nodes to the graph
