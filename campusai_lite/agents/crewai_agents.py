"""
campusai_lite/agents/crewai_agents.py
Part A – CrewAI multi-agent system for CampusAI Lite.

Agents:
  1. PlannerAgent     – decomposes the student's question into sub-tasks
  2. InformationAgent – retrieves university data using tools
  3. ValidationAgent  – validates the answer for accuracy and completeness

NOTE: Requires crewai and crewai-tools. If not installed the module degrades
gracefully: CREWAI_AVAILABLE is set to False and run_crewai() returns a
helpful installation message instead of raising an ImportError.
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool as crewai_tool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

from langchain_openai import ChatOpenAI

# ── Import our custom tools ───────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.university_tool import UniversityInfoTool
from tools.docling_tool import DoclingDocumentTool


# ── LLM setup ────────────────────────────────────────────────────────────────
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("BOB_MODEL", "ibm/granite-3-3-8b-instruct"),
        temperature=0.3,
        api_key=os.getenv("BOB_API_KEY"),
        base_url=os.getenv("BOB_BASE_URL", "https://api.bam.res.ibm.com/v1"),
    )


# ── Wrap LangChain tools as CrewAI-compatible tools ──────────────────────────
_university_tool = UniversityInfoTool()
_docling_tool = DoclingDocumentTool()


if CREWAI_AVAILABLE:
    @crewai_tool("UniversityInfoLookup")
    def university_info_lookup(topic: str, sub_filter: str = "") -> str:
        """Look up structured university information by topic.
        Topics: general, departments, admissions, tuition, facilities,
        calendar, services, research, or a specific department name."""
        return _university_tool._run(topic=topic, sub_filter=sub_filter or None)

    @crewai_tool("ParseUniversityDocument")
    def parse_university_document(file_path: str, query: str = "") -> str:
        """Parse a university document (PDF, DOCX, Markdown) with Docling.
        Returns the text content, filtered by an optional keyword query."""
        return _docling_tool._run(file_path=file_path, query=query or None)
else:
    university_info_lookup = None  # type: ignore[assignment]
    parse_university_document = None  # type: ignore[assignment]


# ── Agent definitions ─────────────────────────────────────────────────────────
def create_planner_agent(llm: ChatOpenAI):  # type: ignore[return]
    return Agent(
        role="University Query Planner",
        goal=(
            "Analyse the student's question and break it down into clear sub-tasks "
            "that the Information Agent can execute. Identify which topics (admissions, "
            "tuition, departments, calendar, etc.) need to be queried."
        ),
        backstory=(
            "You are an experienced academic advisor who understands university processes "
            "deeply. You excel at understanding complex student questions and decomposing "
            "them into actionable information-retrieval tasks."
        ),
        verbose=True,
        allow_delegation=True,
        llm=llm,
        tools=[],
    )


def create_information_agent(llm: ChatOpenAI) -> Agent:
    return Agent(
        role="University Information Retrieval Specialist",
        goal=(
            "Use the available tools to retrieve accurate, up-to-date university "
            "information that answers the student's question completely."
        ),
        backstory=(
            "You are a university information officer with access to the official "
            "TechVista University knowledge base and document repository. You are "
            "thorough, accurate, and present information clearly."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[university_info_lookup, parse_university_document] if CREWAI_AVAILABLE else [],
    )


def create_validation_agent(llm: ChatOpenAI) -> Agent:
    return Agent(
        role="Response Quality Validator",
        goal=(
            "Review the retrieved information for completeness, accuracy, and clarity. "
            "Ensure the response directly addresses the student's original question "
            "and flag any gaps or inconsistencies."
        ),
        backstory=(
            "You are a senior academic quality assurance officer. Your job is to "
            "ensure that student-facing information is accurate, clear, complete, "
            "and appropriately caveated where needed."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[],
    )


# ── Task definitions ──────────────────────────────────────────────────────────
def create_tasks(
    question: str,
    planner: Agent,
    info_agent: Agent,
    validator: Agent,
) -> list[Task]:
    plan_task = Task(
        description=(
            f"A student has asked: '{question}'\n\n"
            "Analyse this question and produce a clear plan listing:\n"
            "1. The main information topics needed (e.g., admissions, tuition, departments).\n"
            "2. Any specific sub-filters (e.g., undergraduate vs graduate).\n"
            "3. Whether any university documents need to be parsed.\n"
            "Output a numbered action plan."
        ),
        expected_output="A numbered action plan with topics and sub-filters to query.",
        agent=planner,
    )

    retrieve_task = Task(
        description=(
            f"Using the action plan created for the question: '{question}'\n"
            "Execute each step using the available tools:\n"
            "- Use 'UniversityInfoLookup' to retrieve structured data.\n"
            "- Use 'ParseUniversityDocument' with file 'sample_document.md' for course details.\n"
            "Compile all retrieved information into a comprehensive, well-structured response."
        ),
        expected_output="A comprehensive, well-structured answer to the student's question.",
        agent=info_agent,
        context=[plan_task],
    )

    validate_task = Task(
        description=(
            f"Review the response drafted for the student's question: '{question}'\n"
            "Check:\n"
            "1. Does it fully answer the question?\n"
            "2. Is it factually consistent with TechVista University data?\n"
            "3. Is it clear and free of jargon?\n"
            "4. Are contact details / next steps included where appropriate?\n"
            "Produce the final, polished answer ready to be shown to the student."
        ),
        expected_output=(
            "A final, polished, student-ready answer that has been validated "
            "for accuracy, completeness, and clarity."
        ),
        agent=validator,
        context=[retrieve_task],
    )

    return [plan_task, retrieve_task, validate_task]


# ── Main crew builder ─────────────────────────────────────────────────────────
def build_campusai_crew(question: str):
    if not CREWAI_AVAILABLE:
        raise ImportError("CrewAI is not installed. Run: pip install crewai crewai-tools")
    llm = _get_llm()
    planner = create_planner_agent(llm)
    info_agent = create_information_agent(llm)
    validator = create_validation_agent(llm)
    tasks = create_tasks(question, planner, info_agent, validator)

    return Crew(
        agents=[planner, info_agent, validator],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


def run_crewai(question: str) -> str:
    """Public entry-point: run the CrewAI pipeline and return the final answer."""
    if not CREWAI_AVAILABLE:
        return (
            "⚠️ CrewAI is not installed on this Python version yet.\n"
            "Install with: pip install crewai crewai-tools\n"
            "Then restart the application."
        )
    crew = build_campusai_crew(question)
    result = crew.kickoff()
    # CrewAI ≥0.60 returns a CrewOutput object; coerce to str for consistency.
    return str(result)


if __name__ == "__main__":
    sample = "What are the admission requirements for the Computer Science graduate program?"
    print(run_crewai(sample))
