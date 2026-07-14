# CampusAI Lite — Agentic University Information Assistant

> **Capstone Project 1** · Agentic AI · TechVista University

---

## Overview

**CampusAI Lite** is a fully agentic AI application that answers student queries about university programs, admissions, tuition, facilities, and more.

It is implemented using **seven AI frameworks** as required by the Capstone brief:

| Framework | Role | Part |
|---|---|---|
| **CrewAI** | Multi-agent crew: Planner → Information → Validation | A |
| **LangChain** | LCEL chain with OpenAI Functions agent | A |
| **LangGraph** | ReAct-style state graph (plan → retrieve → validate) | A |
| **PydanticAI** | Structured I/O validation for all agent inputs/outputs | A |
| **Docling** | University document (PDF/DOCX/MD) parsing tool | A |
| **AG2 (AutoGen)** | Group-chat multi-agent workflow + comparison with CrewAI | B |
| **BeeAI** | Two-agent ReAct proof-of-concept + comparison with CrewAI | B |
| **Gradio** | Interactive web UI with 4 tabs | A |

---

## Project Structure

```
campusai_lite/
├── main.py                        ← Entry point (UI / CLI / demo)
├── requirements.txt
├── .env.example                   ← Copy to .env and add OPENAI_API_KEY
│
├── agents/
│   ├── crewai_agents.py           ← Part A: CrewAI crew (Planner, Info, Validator)
│   ├── langchain_chain.py         ← Part A: LangChain LCEL chain
│   ├── pydantic_models.py         ← Part A: PydanticAI structured models
│   └── __init__.py
│
├── workflows/
│   ├── langgraph_workflow.py      ← Part A: LangGraph ReAct graph
│   └── __init__.py
│
├── tools/
│   ├── university_tool.py         ← Custom tool: university JSON knowledge base
│   ├── docling_tool.py            ← Custom tool: Docling document parser
│   └── __init__.py
│
├── ui/
│   ├── gradio_app.py              ← Gradio multi-tab UI
│   └── __init__.py
│
├── part_b/
│   ├── autogen_workflow.py        ← Part B: AG2/AutoGen workflow + comparison
│   ├── beeai_poc.py               ← Part B: BeeAI PoC + comparison
│   └── __init__.py
│
└── data/
    ├── university_info.json       ← University knowledge base
    └── sample_document.md         ← Course catalog (Docling parses this)
```

---

## Setup

### 1. Install dependencies

```bash
cd campusai_lite
pip install -r requirements.txt
```

For AG2 (Part B Option 1):
```bash
pip install pyautogen
```

For BeeAI (Part B Option 2):
```bash
pip install bee-agent-framework
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Run the Gradio UI

```bash
python main.py
# Open http://localhost:7860
```

---

## Usage

### Gradio UI (recommended)
```bash
python main.py
```
Four tabs:
1. **💬 Ask CampusAI** — Select a framework and ask any question
2. **⚖️ Compare Frameworks** — Side-by-side comparison of all Part A frameworks
3. **📋 Part B Comparison** — Framework comparison tables
4. **ℹ️ About** — Architecture diagram and setup info

### CLI (interactive)
```bash
python main.py --cli
```

### One-shot demo
```bash
python main.py --demo crewai
python main.py --demo langchain
python main.py --demo langgraph
python main.py --demo autogen
python main.py --demo beeai
python main.py --demo crewai --question "What are the tuition fees for international students?"
```

---

## Part A: Core Implementation

### Agent Architecture (CrewAI)

```
Student Question
       │
       ▼
┌─────────────────┐
│  PlannerAgent   │  → Decomposes question into sub-tasks
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  InformationAgent    │  → Uses tools to retrieve university data
│  • UniversityInfoTool│  → Queries local JSON knowledge base
│  • DoclingDocumentTool│ → Parses course catalog documents
└────────┬─────────────┘
         │
         ▼
┌─────────────────────┐
│  ValidationAgent    │  → Validates completeness, accuracy, clarity
└─────────────────────┘
         │
         ▼
   Final Answer
```

### LangGraph Workflow
```
[plan] → [retrieve] → [validate] → END
```
Each node is a pure function operating on a typed `CampusAIState`.

### PydanticAI Models
- `StudentQuery` — validated input
- `PlannerOutput` + `SubTask` — structured plan
- `InformationOutput` + `RetrievedInfo` — retrieved data
- `ValidationOutput` + `ValidationIssue` — validated final answer

---

## Part B: Framework Exploration

Both AG2 and BeeAI comparison tables are available in:
- `part_b/autogen_workflow.py` → `COMPARISON_TEXT`
- `part_b/beeai_poc.py` → `COMPARISON_TEXT`

And in the Gradio UI → **📋 Part B – Framework Comparison** tab.

---

## Sample Questions

- "What are the admission requirements for graduate programs?"
- "What is the tuition fee for international undergraduate students?"
- "When does the fall semester start?"
- "What research funding and industry partners does the university have?"
- "What courses are available in the Computer Science department?"
- "How can I contact the career center?"

---

## Notes

- The university data (`data/university_info.json`) is **fictional** (TechVista University).
  Replace it with real data to deploy for an actual institution.
- **BeeAI** gracefully falls back to a simulated two-agent workflow if `bee-agent-framework`
  is not installed, so the UI always works.
- **AutoGen** shows a helpful error message if `pyautogen` is not installed.
