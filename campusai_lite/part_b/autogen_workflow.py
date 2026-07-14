"""
campusai_lite/part_b/autogen_workflow.py
Part B – Option 1: AG2 (AutoGen) multi-agent workflow for CampusAI Lite.

Comparison with CrewAI is documented at the bottom of this file.

Agents:
  - UserProxyAgent   : represents the student / drives conversation
  - PlannerAgent     : decomposes the question
  - InfoAgent        : retrieves university info using function calls
  - ValidatorAgent   : checks and finalises the answer
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import autogen  # type: ignore
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

from tools.university_tool import UniversityInfoTool
from tools.docling_tool import DoclingDocumentTool

_university_tool = UniversityInfoTool()
_docling_tool = DoclingDocumentTool()


# ── Registered callable functions (AutoGen function-calling interface) ────────
def university_info(topic: str, sub_filter: str = "") -> str:
    """Retrieve structured university information by topic."""
    return _university_tool._run(topic=topic, sub_filter=sub_filter or None)


def parse_document(file_path: str, query: str = "") -> str:
    """Parse a university document and return its text content."""
    return _docling_tool._run(file_path=file_path, query=query or None)


# ── LLM config ────────────────────────────────────────────────────────────────
def _llm_config() -> dict:
    return {
        "config_list": [
            {
                "model": os.getenv("BOB_MODEL", "ibm/granite-3-3-8b-instruct"),
                "api_key": os.getenv("BOB_API_KEY"),
                "base_url": os.getenv("BOB_BASE_URL", "https://api.bam.res.ibm.com/v1"),
            }
        ],
        "temperature": 0.3,
        "functions": [
            {
                "name": "university_info",
                "description": (
                    "Look up university information. Topics: general, departments, "
                    "admissions, tuition, facilities, calendar, services, research, "
                    "or a department name like 'Computer Science'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "The topic to look up."},
                        "sub_filter": {
                            "type": "string",
                            "description": "Optional sub-filter like 'graduate' or 'undergraduate'.",
                        },
                    },
                    "required": ["topic"],
                },
            },
            {
                "name": "parse_document",
                "description": "Parse a university document (PDF, DOCX, Markdown) with Docling.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path or filename."},
                        "query": {"type": "string", "description": "Keyword filter."},
                    },
                    "required": ["file_path"],
                },
            },
        ],
    }


# ── Build AG2 group chat ──────────────────────────────────────────────────────
def build_autogen_chat(question: str) -> str:
    """Build and run an AutoGen multi-agent group chat for the question."""
    if not AUTOGEN_AVAILABLE:
        return (
            "AutoGen (pyautogen) is not installed.\n"
            "Install with: pip install pyautogen\n"
            "Then retry your question."
        )

    llm_cfg = _llm_config()

    # ── Define agents ────────────────────────────────────────────────────────
    user_proxy = autogen.UserProxyAgent(
        name="StudentProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda msg: "FINAL ANSWER:" in msg.get("content", ""),
        code_execution_config=False,
        function_map={
            "university_info": university_info,
            "parse_document": parse_document,
        },
    )

    planner = autogen.AssistantAgent(
        name="PlannerAgent",
        system_message=(
            "You are a university query planner. When you receive a student question, "
            "analyse it and output a JSON plan with a list of topics to retrieve. "
            "Then pass control to InfoAgent with your plan."
        ),
        llm_config=llm_cfg,
    )

    info_agent = autogen.AssistantAgent(
        name="InfoAgent",
        system_message=(
            "You are a university information specialist. Use the 'university_info' "
            "function to retrieve data for each topic in the planner's list. "
            "If course info is needed, also call 'parse_document' with 'sample_document.md'. "
            "After retrieving all data, compile a detailed response and pass it to ValidationAgent."
        ),
        llm_config=llm_cfg,
    )

    validator = autogen.AssistantAgent(
        name="ValidationAgent",
        system_message=(
            "You are a response quality validator. Review the compiled information and "
            "ensure it fully answers the student's original question. "
            "Produce the final polished answer and prefix it with 'FINAL ANSWER:' "
            "to signal completion."
        ),
        llm_config=llm_cfg,
    )

    # ── Group chat ────────────────────────────────────────────────────────────
    group_chat = autogen.GroupChat(
        agents=[user_proxy, planner, info_agent, validator],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin",
    )

    manager = autogen.GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_cfg,
    )

    # Capture output
    final_answer = ""
    try:
        user_proxy.initiate_chat(
            manager,
            message=f"Student question: {question}",
        )
        # Extract the last FINAL ANSWER from messages
        for msg in reversed(group_chat.messages):
            content = msg.get("content", "")
            if "FINAL ANSWER:" in content:
                final_answer = content.split("FINAL ANSWER:", 1)[-1].strip()
                break
        if not final_answer:
            final_answer = group_chat.messages[-1].get("content", "No answer generated.")
    except Exception as exc:
        final_answer = f"AutoGen error: {exc}"

    return final_answer


def run_autogen(question: str) -> str:
    """Public entry point for the AG2 workflow."""
    return build_autogen_chat(question)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON: AutoGen (AG2) vs CrewAI
# ─────────────────────────────────────────────────────────────────────────────
COMPARISON_TEXT = """
## AG2 (AutoGen) vs CrewAI — Comparison

| Dimension              | CrewAI                                   | AG2 / AutoGen                            |
|------------------------|------------------------------------------|------------------------------------------|
| Abstraction level      | High — opinionated roles & tasks         | Medium — flexible conversation protocol  |
| Agent definition       | Role + Goal + Backstory + Tools          | SystemMessage + function_map             |
| Workflow type          | Sequential or hierarchical tasks         | Group chat with round-robin or custom    |
| Tool integration       | Native CrewAI tools or LangChain tools   | Function-calling via OpenAI schema       |
| Human-in-the-loop      | Limited; mostly automated                | First-class; UserProxyAgent design       |
| Verbosity / debugging  | Verbose agent logs built-in              | Conversation logs; harder to inspect     |
| Code execution         | Not default                              | Supports sandboxed code execution        |
| Learning curve         | Easier for beginners                     | Steeper; requires understanding GroupChat|
| Ecosystem              | CrewAI Hub, CrewAI tools                 | AutoGen Studio, AutoBuild                |
| Best suited for        | Structured business workflows            | Research / conversational agent systems  |

### Summary
- **CrewAI** is ideal when you have a well-defined, sequential workflow with clear roles and tasks
  (like the planner → retriever → validator pipeline). It is more beginner-friendly and readable.
- **AG2 / AutoGen** shines in open-ended conversations, code-generation tasks, and scenarios
  requiring human-in-the-loop or dynamic agent selection. Its group-chat model is more flexible
  but requires more configuration.
- For CampusAI Lite, CrewAI provided a cleaner developer experience; AutoGen would be preferred
  if we needed students to interactively provide feedback during the conversation.
"""


if __name__ == "__main__":
    sample = "What are the admission deadlines and tuition fees for graduate programs?"
    print(run_autogen(sample))
    print(COMPARISON_TEXT)
