# Agentic AI — Core Concepts

---

## What is Agentic AI?

**Agentic AI** refers to AI systems that act **autonomously over extended, multi-step tasks** — planning, executing, observing, adapting — with minimal human intervention.

The key shift: from AI that **responds** (chatbots) to AI that **acts** (agents that accomplish goals).

---

## Properties of Agentic Systems

| Property | Description | Example |
|---|---|---|
| **Autonomy** | Operates without step-by-step human guidance | "Research this topic and write a report" |
| **Goal-directedness** | Works toward a defined objective | Booking a flight within budget |
| **Reactivity** | Adapts when environment changes | Retries with different query when search fails |
| **Tool use** | Extends capabilities beyond the LLM | Web search, code execution, database |
| **Memory** | Maintains context across steps | Remembers user said "no Microsoft products" |
| **Multi-step planning** | Breaks complex goals into subtasks | Outline → research → write → edit |
| **Proactivity** | Takes initiative without being asked | Notifies user of errors found during review |

---

## Agentic AI vs Traditional AI

| | Traditional AI / Chatbot | Agentic AI |
|---|---|---|
| **Interaction** | Single Q&A turn | Multi-step autonomous execution |
| **Control flow** | Human-driven | Agent-driven |
| **Tool use** | None or fixed | Dynamic, autonomous |
| **Duration** | Seconds | Minutes to hours |
| **State** | Stateless | Persistent state across steps |
| **Examples** | ChatGPT free-form chat | Devin, AutoGPT, Manus, Claude Computer Use |

---

## Levels of Autonomy (Agency Spectrum)

```
Level 0: Pure Q&A
   "What is the capital of France?" → "Paris"

Level 1: Tool-Augmented
   Uses a single tool to answer one question

Level 2: Multi-Step Agent
   Plans and executes 3–10 steps autonomously

Level 3: Multi-Agent System
   Orchestrates multiple specialized agents

Level 4: Fully Autonomous
   Long-horizon tasks, self-directed, minimal HITL
```

---

## Real-World Examples

| System | Agentic Behavior |
|---|---|
| **GitHub Copilot Workspace** | Plans, edits, and commits code from a task description |
| **Devin (Cognition)** | End-to-end software engineering agent |
| **AutoGPT** | Open-source autonomous agent loop |
| **Claude Computer Use** | Controls a real computer UI |
| **LangGraph + tools** | Custom agentic workflows |
| **Manus** | General autonomous agent from China |

---

## Core Agentic Design Principles

### 1. Start with the simplest possible architecture
Don't build multi-agent systems when a single agent with 3 tools will do.

### 2. Make tools reliable before adding more
A chain is only as strong as its weakest tool.

### 3. Add memory only when needed
Memory adds complexity. Start stateless.

### 4. Add human-in-the-loop for irreversible actions
Never let an agent delete data or send emails without approval.

### 5. Observe before you trust
Always test agent behavior on edge cases before deploying.

---

## Agentic AI Stack

```
┌─────────────────────────────────────────┐
│          User / Application             │
├─────────────────────────────────────────┤
│     Orchestration (LangGraph / AutoGen) │
├─────────────┬───────────────────────────┤
│   Agent(s)  │  Memory / State           │
├─────────────┴───────────────────────────┤
│              Tools                      │
│  (Search, Code, DB, APIs, Files, Email) │
├─────────────────────────────────────────┤
│              LLM(s)                     │
│  (GPT-4o, Claude, Gemini, LLaMA)       │
└─────────────────────────────────────────┘
```
