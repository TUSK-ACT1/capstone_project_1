"""
campusai_lite/workflows/langgraph_workflow.py
Part A – LangGraph ReAct-style workflow for CampusAI Lite.

Graph nodes:
  plan    → Planner LLM call that classifies intent and selects topics.
  retrieve → Calls UniversityInfoTool and/or DoclingDocumentTool.
  validate → Checks completeness and builds final answer.
  respond → Terminal: returns the validated answer.
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from tools.university_tool import UniversityInfoTool
from tools.docling_tool import DoclingDocumentTool


# ── Graph state ───────────────────────────────────────────────────────────────
class CampusAIState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    plan: str
    retrieved_info: str
    final_answer: str
    iteration: int


# ── LLM + tools ──────────────────────────────────────────────────────────────
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("BOB_MODEL", "ibm/granite-3-3-8b-instruct"),
        temperature=0.3,
        api_key=os.getenv("BOB_API_KEY"),
        base_url=os.getenv("BOB_BASE_URL", "https://api.bam.res.ibm.com/v1"),
    )


_university_tool = UniversityInfoTool()
_docling_tool = DoclingDocumentTool()

TOPIC_MAP = {
    "admissions": ("admissions", None),
    "admission": ("admissions", None),
    "apply": ("admissions", None),
    "tuition": ("tuition", None),
    "fees": ("tuition", None),
    "cost": ("tuition", None),
    "department": ("departments", None),
    "program": ("departments", None),
    "course": ("departments", None),
    "calendar": ("calendar", None),
    "semester": ("calendar", None),
    "facilities": ("facilities", None),
    "campus": ("facilities", None),
    "research": ("research", None),
    "services": ("services", None),
    "contact": ("services", None),
    "general": ("general", None),
}


def _detect_topics(question: str) -> list[tuple[str, Optional[str]]]:
    """Simple keyword-based topic detection as a fallback."""
    q = question.lower()
    found = []
    seen = set()
    for kw, topic_pair in TOPIC_MAP.items():
        if kw in q and topic_pair[0] not in seen:
            found.append(topic_pair)
            seen.add(topic_pair[0])
    # sub-filter for graduate/undergraduate
    for i, (topic, _) in enumerate(found):
        if "graduate" in q or "grad" in q or "master" in q or "phd" in q:
            found[i] = (topic, "graduate")
        elif "undergraduate" in q or "bachelors" in q or "ug" in q:
            found[i] = (topic, "undergraduate")
    return found if found else [("general", None)]


# ── Node: plan ────────────────────────────────────────────────────────────────
def plan_node(state: CampusAIState) -> CampusAIState:
    llm = _get_llm()
    question = state["question"]

    response = llm.invoke([
        HumanMessage(content=(
            f"You are a university query planner. A student asks: '{question}'\n"
            "List the topics (as a JSON array) needed to answer it. "
            "Allowed topics: general, departments, admissions, tuition, facilities, "
            "calendar, services, research, Computer Science, Data Science, "
            "Artificial Intelligence, Business Administration.\n"
            "Also list sub_filters (graduate/undergraduate) if relevant.\n"
            'Output ONLY valid JSON: {"topics": ["topic1", ...], "sub_filters": {"topic1": "graduate"}}'
        ))
    ])

    plan_text = response.content
    state["plan"] = plan_text
    state["messages"] = state.get("messages", []) + [AIMessage(content=f"Plan: {plan_text}")]
    return state


# ── Node: retrieve ─────────────────────────────────────────────────────────────
def retrieve_node(state: CampusAIState) -> CampusAIState:
    question = state["question"]
    plan_text = state.get("plan", "")
    info_chunks: list[str] = []

    # Try to parse the LLM plan
    try:
        plan_data = json.loads(plan_text)
        topics: list[str] = plan_data.get("topics", [])
        sub_filters: dict[str, str] = plan_data.get("sub_filters", {})
    except (json.JSONDecodeError, TypeError):
        # Fallback: keyword detection
        detected = _detect_topics(question)
        topics = [t[0] for t in detected]
        sub_filters = {t[0]: t[1] for t in detected if t[1]}

    # Retrieve info for each topic
    for topic in topics:
        sf = sub_filters.get(topic)
        chunk = _university_tool._run(topic=topic, sub_filter=sf)
        info_chunks.append(f"[{topic}]\n{chunk}")

    # Check if course/document info needed
    if any(w in question.lower() for w in ("course", "cs 1", "cs 2", "ai 5", "ds 3", "mba")):
        doc_chunk = _docling_tool._run(
            file_path="sample_document.md",
            query=question[:80],
        )
        info_chunks.append(f"[course_catalog]\n{doc_chunk}")

    retrieved = "\n\n".join(info_chunks) if info_chunks else "No specific information found."
    state["retrieved_info"] = retrieved
    state["messages"] = state.get("messages", []) + [
        AIMessage(content=f"Retrieved information:\n{retrieved[:500]}…")
    ]
    return state


# ── Node: validate ─────────────────────────────────────────────────────────────
def validate_node(state: CampusAIState) -> CampusAIState:
    llm = _get_llm()
    question = state["question"]
    retrieved = state.get("retrieved_info", "")

    response = llm.invoke([
        HumanMessage(content=(
            f"You are CampusAI, TechVista University's AI assistant.\n\n"
            f"Student question: {question}\n\n"
            f"Retrieved university information:\n{retrieved}\n\n"
            "Write a clear, friendly, and complete answer to the student's question "
            "using only the retrieved information. End with a helpful next step or contact."
        ))
    ])

    final_answer = response.content
    state["final_answer"] = final_answer
    state["messages"] = state.get("messages", []) + [AIMessage(content=final_answer)]
    return state


# ── Edge condition ─────────────────────────────────────────────────────────────
def should_end(state: CampusAIState) -> str:
    """Always go to validate after retrieve, then end."""
    if state.get("final_answer"):
        return END
    return "validate"


# ── Build the graph ───────────────────────────────────────────────────────────
def build_langgraph() -> StateGraph:
    graph = StateGraph(CampusAIState)

    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "validate")
    graph.add_edge("validate", END)

    return graph.compile()


# ── Public entry point ────────────────────────────────────────────────────────
_compiled_graph = None


def run_langgraph(question: str) -> str:
    """Run the LangGraph workflow and return the final answer."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_langgraph()

    initial_state: CampusAIState = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "plan": "",
        "retrieved_info": "",
        "final_answer": "",
        "iteration": 0,
    }

    result = _compiled_graph.invoke(initial_state)
    return result.get("final_answer", "No answer generated.")


if __name__ == "__main__":
    sample = "What are the facilities available on campus and when does the fall semester start?"
    print(run_langgraph(sample))
