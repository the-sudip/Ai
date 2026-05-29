# ReAct Agent Pattern

**ReAct** (Reasoning + Acting) is the foundational prompting pattern for AI agents. The model alternates between **thinking** (Thought) and **doing** (Action + Observation) in a structured loop.

---

## The ReAct Format

```
Thought: [The model's internal reasoning about what to do]
Action: [The tool to call]
Action Input: [The arguments for the tool]
Observation: [The tool's output]
Thought: [Reasoning based on the observation]
... (repeats)
Thought: I now know the final answer.
Final Answer: [The answer to the user]
```

---

## Why ReAct Works

By **externalizing the reasoning process** into text, the model:
- Can plan before acting
- Can evaluate whether a tool result makes sense
- Can change strategy if a tool fails
- Produces an auditable reasoning trace

---

## ReAct Prompt Template

```python
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

# Pull the standard ReAct prompt from LangChain Hub
react_prompt = hub.pull("hwchase17/react")
print(react_prompt.template)
# Includes: "You have access to the following tools: {tools}
# Use the following format: Thought/Action/Action Input/Observation/...
# Begin! Question: {input} {agent_scratchpad}"

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [DuckDuckGoSearchRun()]

# Create the agent (just the decision-making part)
agent = create_react_agent(llm, tools, react_prompt)

# AgentExecutor runs the loop
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,         # prints every step
    max_iterations=8,
    handle_parsing_errors=True,
)

result = executor.invoke({"input": "Who is the current CEO of OpenAI and what is their background?"})
print(result["output"])
```

**Sample verbose output:**
```
> Entering new AgentExecutor chain...

Thought: I need to find who the current CEO of OpenAI is and their background.
Action: duckduckgo_search
Action Input: current CEO of OpenAI 2025
Observation: Sam Altman is the CEO of OpenAI. He was previously the president of Y Combinator...

Thought: I have enough information to answer.
Final Answer: The current CEO of OpenAI is Sam Altman. He previously served as president of Y Combinator...
```

---

## ReAct with Multiple Tools

```python
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use Python syntax."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"__name__": "__main__"})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

tools = [
    DuckDuckGoSearchRun(),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    calculator,
]

executor = AgentExecutor(
    agent=create_react_agent(llm, tools, react_prompt),
    tools=tools,
    verbose=True,
    max_iterations=10,
)

result = executor.invoke({
    "input": "What is the population of Tokyo, and what percentage is that of Japan's total population?"
})
```

The agent might:
1. Search for Tokyo population → 13.96 million
2. Search for Japan total population → 125.7 million
3. Use calculator: `(13.96 / 125.7) * 100` → 11.1%
4. Return: "Tokyo's population of ~13.96 million is approximately 11.1% of Japan's total population of ~125.7 million."

---

## ReAct vs Tool-Calling Agent

| | ReAct | Tool-Calling Agent |
|---|---|---|
| **Mechanism** | Generates `Thought/Action/Observation` text | Uses model's native function-calling API |
| **Model requirement** | Any LLM (even without function calling) | Must support tool/function calling |
| **Reliability** | Can have parsing errors | More reliable (structured output) |
| **Transparency** | Full reasoning trace visible | Less transparent scratchpad |
| **LangChain function** | `create_react_agent` | `create_tool_calling_agent` |

---

## Custom ReAct Prompt

```python
from langchain_core.prompts import ChatPromptTemplate

custom_react_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a financial research assistant. 
    
You have access to the following tools:
{tools}

Tool names: {tool_names}

ALWAYS follow this format:
Thought: [your reasoning]
Action: [tool name]
Action Input: [tool arguments]
Observation: [tool output — filled by system]
... (repeat as needed)
Thought: I now have enough information.
Final Answer: [your final answer]"""),
    ("human", "{input}\n\n{agent_scratchpad}"),
])

agent = create_react_agent(llm, tools, custom_react_prompt)
```
