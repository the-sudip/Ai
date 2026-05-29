# Built-in Tools & Toolkits

LangChain provides a rich library of pre-built tools and toolkits so you don't have to build common capabilities from scratch.

---

## Individual Built-in Tools

### Web Search — DuckDuckGo (Free, No API Key)
```python
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults

# Simple — returns a string
search = DuckDuckGoSearchRun()
result = search.invoke("Latest Python version 2025")
print(result)  # "Python 3.13 was released..."

# Detailed — returns structured results with URLs
search_results = DuckDuckGoSearchResults(num_results=5)
result = search_results.invoke("LangGraph tutorial")
# Returns JSON with title, link, snippet for each result
```

### Wikipedia
```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=2000,
    )
)
result = wiki.invoke("Transformer neural network architecture")
```

### Python REPL — Run Code
```python
from langchain_experimental.tools import PythonREPLTool

python_repl = PythonREPLTool()
result = python_repl.invoke("import math; print(math.factorial(10))")
# "3628800"

# WARNING: Only use in sandboxed environments!
# This executes real Python code.
```

### Shell Tool — Run Shell Commands
```python
from langchain_community.tools import ShellTool

shell = ShellTool()
result = shell.invoke("ls -la | head -10")
# WARNING: Extremely dangerous in production! Sandboxed environments only.
```

### Tavily Search (Recommended for Production)
```python
from langchain_community.tools.tavily_search import TavilySearchResults
import os

os.environ["TAVILY_API_KEY"] = "your_key"

tavily = TavilySearchResults(
    max_results=5,
    include_answer=True,
    include_raw_content=False,
)
result = tavily.invoke("What are the latest developments in quantum computing?")
```

---

## File System Tools

```python
from langchain_community.tools import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    CopyFileTool,
    MoveFileTool,
)
from langchain_community.tools.file_management.toolkit import FileManagementToolkit

# Scoped to a specific directory for safety
toolkit = FileManagementToolkit(root_dir="./workspace/")
file_tools = toolkit.get_tools()

# Individual tools
read_tool = ReadFileTool(root_dir="./workspace/")
write_tool = WriteFileTool(root_dir="./workspace/")

result = read_tool.invoke({"file_path": "report.txt"})
write_tool.invoke({"file_path": "output.txt", "text": "Hello, World!"})
```

---

## Retriever Tool — RAG as a Tool

Convert any retriever into a tool for an agent to call:

```python
from langchain.tools.retriever import create_retriever_tool

# Wrap your vector store retriever as a tool
knowledge_base_tool = create_retriever_tool(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    name="search_company_knowledge_base",
    description="Search the company's internal knowledge base for policies, procedures, and FAQs. Use this for any questions about company-specific information.",
)

# Now agents can call this tool to do RAG
agent = create_react_agent(llm, [knowledge_base_tool, web_search])
```

---

## Toolkits — Grouped Tool Sets

Toolkits provide multiple related tools for a domain.

### SQL Database Toolkit
```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent

db = SQLDatabase.from_uri("sqlite:///sales.db")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Provides: sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="tool-calling",
)

result = agent.invoke({"input": "What were total sales by region in Q3 2024?"})
```

### CSV Agent
```python
from langchain_experimental.agents import create_csv_agent

agent = create_csv_agent(
    llm=llm,
    path="sales_data.csv",
    verbose=True,
    agent_type="tool-calling",
)

result = agent.invoke({"input": "What is the average order value by product category?"})
```

### Pandas DataFrame Agent
```python
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent

df = pd.read_csv("data.csv")
agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type="tool-calling")
result = agent.invoke({"input": "Show me the top 5 customers by revenue"})
```

---

## Custom Toolkit (Group Your Tools)

```python
from langchain_core.tools import BaseToolkit

class WeatherToolkit(BaseToolkit):
    """Toolkit for weather-related tools."""
    
    def get_tools(self):
        return [
            get_current_weather_tool,
            get_weather_forecast_tool,
            get_historical_weather_tool,
        ]

toolkit = WeatherToolkit()
tools = toolkit.get_tools()
agent = create_react_agent(llm, tools)
```

---

## Tool Selection Tips

| Use Case | Recommended Tool |
|---|---|
| Quick web search | `DuckDuckGoSearchRun` |
| Production web search | `TavilySearchResults` |
| Encyclopedia lookup | `WikipediaQueryRun` |
| Private document search | `create_retriever_tool` |
| Database queries | `SQLDatabaseToolkit` |
| CSV/DataFrame analysis | `create_pandas_dataframe_agent` |
| Code execution | `PythonREPLTool` (sandboxed!) |
| File operations | `FileManagementToolkit` (scoped!) |
