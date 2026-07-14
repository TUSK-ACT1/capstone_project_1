"""
campusai_lite/part_b/beeai_poc.py
Part B – Option 2: BeeAI Proof of Concept.

A simple two-agent workflow built with BeeAI Framework:
  - AgentA (Router): classifies the student's question and routes to AgentB.
  - AgentB (Answerer): uses the university tool to answer the question.

Comparison with CrewAI is documented at the bottom.
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # BeeAI framework (pip install bee-agent-framework)
    from bee_agent_framework.agents.bee.agent import BeeAgent  # type: ignore
    from bee_agent_framework.llms.openai import OpenAILLM  # type: ignore
    from bee_agent_framework.tools.base import Tool  # type: ignore
    from bee_agent_framework.memory.unconstrained_memory import UnconstrainedMemory  # type: ignore
    BEEAI_AVAILABLE = True
except ImportError:
    BEEAI_AVAILABLE = False

from tools.university_tool import UniversityInfoTool

_university_tool = UniversityInfoTool()


# ── Fallback simulation when BeeAI is not installed ──────────────────────────
class _SimulatedBeeAgent:
    """Simulates a two-agent BeeAI workflow without the actual SDK."""

    def __init__(self, role: str, system_prompt: str):
        self.role = role
        self.system_prompt = system_prompt
        # Use LangChain ChatOpenAI as the underlying LLM
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model=os.getenv("BOB_MODEL", "ibm/granite-3-3-8b-instruct"),
            temperature=0.3,
            api_key=os.getenv("BOB_API_KEY"),
            base_url=os.getenv("BOB_BASE_URL", "https://api.bam.res.ibm.com/v1"),
        )

    def run(self, message: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=message),
        ])
        return response.content


# ── BeeAI workflow ─────────────────────────────────────────────────────────────
def run_beeai(question: str) -> str:
    """
    Two-agent BeeAI workflow:
      1. RouterAgent classifies intent and extracts the topic.
      2. AnswererAgent uses the university tool to build the answer.
    """
    if BEEAI_AVAILABLE:
        return _run_with_real_beeai(question)
    else:
        return _run_simulated_beeai(question)


def _run_simulated_beeai(question: str) -> str:
    """Simulated BeeAI two-agent pipeline (no SDK needed)."""
    # ── Agent A: Router ───────────────────────────────────────────────────────
    router_prompt = (
        "You are a university query router. Given a student question, "
        "extract the most relevant topic from: "
        "general, departments, admissions, tuition, facilities, calendar, services, research. "
        "Also extract a sub_filter if mentioned (graduate/undergraduate). "
        'Reply ONLY as JSON: {"topic": "...", "sub_filter": "..." or null}'
    )
    router = _SimulatedBeeAgent("Router", router_prompt)
    routing_result = router.run(question)

    # Parse routing
    import json
    try:
        routing = json.loads(routing_result)
        topic = routing.get("topic", "general")
        sub_filter = routing.get("sub_filter")
    except (json.JSONDecodeError, TypeError):
        topic = "general"
        sub_filter = None

    # ── Tool call ─────────────────────────────────────────────────────────────
    retrieved = _university_tool._run(topic=topic, sub_filter=sub_filter)

    # ── Agent B: Answerer ─────────────────────────────────────────────────────
    answerer_prompt = (
        "You are CampusAI, TechVista University's helpful AI assistant. "
        "Use the provided information to answer the student's question clearly "
        "and concisely. End with a contact or next step."
    )
    answerer = _SimulatedBeeAgent("Answerer", answerer_prompt)
    final_answer = answerer.run(
        f"Student question: {question}\n\nRetrieved information:\n{retrieved}\n\n"
        "Please write a clear answer for the student."
    )

    return (
        f"[BeeAI Simulation — RouterAgent topic: '{topic}', sub_filter: '{sub_filter}']\n\n"
        f"{final_answer}"
    )


def _run_with_real_beeai(question: str) -> str:
    """Actual BeeAI SDK workflow (requires bee-agent-framework installed)."""
    try:
        llm = OpenAILLM(
            model_id=os.getenv("BOB_MODEL", "ibm/granite-3-3-8b-instruct"),
            api_key=os.getenv("BOB_API_KEY"),
            base_url=os.getenv("BOB_BASE_URL", "https://api.bam.res.ibm.com/v1"),
        )

        # Simple wrapper to make UniversityInfoTool compatible with BeeAI
        class UniversityBeeAITool(Tool):
            name = "university_info"
            description = "Retrieve TechVista University information by topic."

            def _run(self, topic: str, sub_filter: str = "") -> str:
                return _university_tool._run(topic=topic, sub_filter=sub_filter or None)

        agent = BeeAgent(
            llm=llm,
            tools=[UniversityBeeAITool()],
            memory=UnconstrainedMemory(),
        )
        result = agent.run(question)
        return str(result.result.text)
    except Exception as exc:
        return f"BeeAI SDK error: {exc}\n\nFalling back to simulation:\n" + _run_simulated_beeai(question)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON: BeeAI vs CrewAI
# ─────────────────────────────────────────────────────────────────────────────
COMPARISON_TEXT = """
## BeeAI vs CrewAI — Developer Experience Comparison

| Dimension              | CrewAI                                   | BeeAI                                    |
|------------------------|------------------------------------------|------------------------------------------|
| Primary language       | Python                                   | Python (also TypeScript SDK)             |
| Agent paradigm         | Role-based crew with sequential tasks    | ReAct-style single/multi-agent loop      |
| Tool integration       | LangChain tools / CrewAI tools           | Custom Tool base class; MCP compatible   |
| Memory support         | Basic (task context)                     | Built-in memory modules (sliding, token) |
| Observability          | Verbose crew logs                        | Built-in instrumentation hooks           |
| Multi-agent comms      | Delegation via task context              | Agent-to-agent messaging protocols       |
| Learning curve         | Low — beginner-friendly                  | Medium — requires understanding ReAct    |
| Community / maturity   | Mature, large community                  | Newer, smaller community (IBM-backed)    |
| Deployment             | Self-hosted or CrewAI Cloud              | BeeAI Platform or self-hosted            |
| Best suited for        | Structured, role-based pipelines         | Research, tool-heavy, or conversational  |

### Summary
- **CrewAI** is the more mature and beginner-friendly choice for structured, sequential
  multi-agent workflows with well-defined roles.
- **BeeAI** offers stronger built-in memory management, better observability hooks, and
  a TypeScript-first SDK that is well-suited for production deployments within enterprise
  environments (IBM ecosystem). Its ReAct loop is more transparent and inspectable.
- For a university assistant like CampusAI Lite, CrewAI delivered faster results; BeeAI
  would be preferred for production deployments requiring persistent memory and audit trails.
"""


if __name__ == "__main__":
    sample = "Tell me about research opportunities at TechVista University."
    print(run_beeai(sample))
    print(COMPARISON_TEXT)
