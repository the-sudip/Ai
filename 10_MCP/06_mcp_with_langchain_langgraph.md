# MCP with LangChain and LangGraph

A complete guide to integrating MCP servers into LangChain/LangGraph applications.

---

## Setup

```bash
pip install langchain-mcp-adapters langchain-openai langgraph mcp
```

---

## `MultiServerMCPClient` — Core Integration

`MultiServerMCPClient` connects to one or more MCP servers and converts their tools into standard LangChain tools:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import asyncio

llm = ChatOpenAI(model="gpt-4o")

async def main():
    async with MultiServerMCPClient({
        # Local server via stdio
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/workspace"],
            "transport": "stdio",
        },
        # Remote server via SSE
        "company_tools": {
            "url": "https://tools.mycompany.com/mcp",
            "transport": "sse",
            "headers": {"Authorization": f"Bearer {os.environ['MCP_TOKEN']}"},
        },
    }) as client:
        tools = await client.get_tools()
        
        agent = create_react_agent(llm, tools)
        
        result = await agent.ainvoke({
            "messages": [HumanMessage("List the files in /tmp/workspace and summarize them")]
        })
        
        print(result["messages"][-1].content)

asyncio.run(main())
```

---

## Connecting a Single MCP Server

```python
from langchain_mcp_adapters.client import SingleServerMCPClient

async def single_server():
    async with SingleServerMCPClient(
        command="python",
        args=["my_mcp_server.py"],
        transport="stdio",
    ) as client:
        tools = await client.get_tools()
        print([t.name for t in tools])
        
        # Tools are standard LangChain BaseTool instances
        # You can call them directly too
        weather_tool = next(t for t in tools if t.name == "get_weather")
        result = await weather_tool.ainvoke({"city": "London"})
        print(result)
```

---

## Full LangGraph Agent with Multiple MCP Servers

```python
import asyncio
import os
from typing import TypedDict, Annotated
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Optional: internal LangChain tools to mix with MCP tools
@tool
def calculate_percentage(value: float, total: float) -> str:
    """Calculate what percentage value is of total."""
    if total == 0:
        return "Error: Cannot divide by zero"
    return f"{(value / total) * 100:.2f}%"

async def build_mcp_agent():
    """Build a full LangGraph agent with MCP tool support."""
    
    async with MultiServerMCPClient({
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]},
            "transport": "stdio",
        },
        "search": {
            "url": "http://localhost:8080/mcp",
            "transport": "sse",
        },
    }) as mcp_client:
        
        mcp_tools = await mcp_client.get_tools()
        all_tools = mcp_tools + [calculate_percentage]
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(all_tools)
        
        # Define state
        class AgentState(TypedDict):
            messages: Annotated[list[BaseMessage], add_messages]
        
        # Define nodes
        def call_model(state: AgentState) -> AgentState:
            system = SystemMessage("""You are a helpful research assistant.
Use tools to gather accurate information before answering.
Always cite your sources.""")
            response = llm_with_tools.invoke([system] + state["messages"])
            return {"messages": [response]}
        
        tool_node = ToolNode(all_tools)
        
        # Build graph
        graph = StateGraph(AgentState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        
        agent = graph.compile(checkpointer=MemorySaver())
        
        # Run the agent
        config = {"configurable": {"thread_id": "research_session_1"}}
        result = await agent.ainvoke(
            {"messages": [HumanMessage("What are the latest releases in the langchain-ai/langgraph GitHub repo?")]},
            config=config,
        )
        
        return result["messages"][-1].content

print(asyncio.run(build_mcp_agent()))
```

---

## Mixing MCP Tools with Native LangChain Tools

```python
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Native Python tools
@tool
def format_as_table(data: str) -> str:
    """Format data as a markdown table."""
    # formatting logic
    return f"| Column1 | Column2 |\n|---|---|\n| {data} |"

@tool  
def get_current_date() -> str:
    """Get today's date."""
    from datetime import date
    return str(date.today())

async def agent_with_mixed_tools():
    async with MultiServerMCPClient({
        "database": {"command": "python", "args": ["db_server.py"], "transport": "stdio"},
    }) as client:
        
        mcp_tools = await client.get_tools()
        
        # Combine MCP tools + native tools seamlessly
        all_tools = mcp_tools + [format_as_table, get_current_date]
        
        agent = create_react_agent(
            model=ChatOpenAI(model="gpt-4o"),
            tools=all_tools,
            state_modifier="You are a data analyst. Use database tools to query data, then format results clearly.",
        )
        
        return await agent.ainvoke({
            "messages": [HumanMessage("Show me today's sales report formatted as a table")]
        })
```

---

## Accessing MCP Resources from LangChain

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async def read_mcp_resource():
    async with MultiServerMCPClient({
        "files": {"command": "python", "args": ["file_server.py"], "transport": "stdio"},
    }) as client:
        # List available resources
        resources = await client.list_resources("files")
        print([r.uri for r in resources])
        
        # Read a specific resource
        content = await client.read_resource("files", "docs://readme")
        print(content)
```

---

## Error Handling in MCP Agents

```python
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph

# ToolNode automatically handles tool errors gracefully
# If a tool raises an exception, it returns a ToolMessage with the error text
# instead of crashing the agent

# Custom error handling:
class SafeToolNode(ToolNode):
    def handle_tool_error(self, error: Exception, tool_call_id: str) -> ToolMessage:
        return ToolMessage(
            content=f"Tool failed: {str(error)[:500]}. Please try a different approach.",
            tool_call_id=tool_call_id,
        )
```

---

## Key Points for Interviews

1. `MultiServerMCPClient` is the main adapter — connects multiple MCP servers
2. MCP tools become regular `BaseTool` instances — transparent to LangGraph
3. The `async with` context manager handles server lifecycle (start/stop)
4. You can mix MCP tools (`get_tools()`) with native `@tool` functions freely
5. `create_react_agent` works seamlessly with MCP tools
6. Resources are accessible via `client.read_resource(server_name, uri)`
