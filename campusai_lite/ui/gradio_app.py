"""
campusai_lite/ui/gradio_app.py
Part A – Gradio UI for CampusAI Lite.

Tabs:
  1. Ask CampusAI    – choose framework and submit question
  2. Compare Frameworks – side-by-side responses from all three frameworks
  3. Framework Comparison Notes – static comparison tables from Part B
  4. About           – project info
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Import runners (lazy to avoid import errors on missing packages) ──────────
def _run_crewai(question: str) -> str:
    try:
        from agents.crewai_agents import run_crewai
        return run_crewai(question)
    except Exception as exc:
        return f"❌ CrewAI error: {exc}"


def _run_langchain(question: str) -> str:
    try:
        from agents.langchain_chain import run_langchain
        return run_langchain(question)
    except Exception as exc:
        return f"❌ LangChain error: {exc}"


def _run_langgraph(question: str) -> str:
    try:
        from workflows.langgraph_workflow import run_langgraph
        return run_langgraph(question)
    except Exception as exc:
        return f"❌ LangGraph error: {exc}"


def _run_autogen(question: str) -> str:
    try:
        from part_b.autogen_workflow import run_autogen
        return run_autogen(question)
    except Exception as exc:
        return f"❌ AutoGen error: {exc}"


def _run_beeai(question: str) -> str:
    try:
        from part_b.beeai_poc import run_beeai
        return run_beeai(question)
    except Exception as exc:
        return f"❌ BeeAI error: {exc}"


# ── Single-framework runner ───────────────────────────────────────────────────
FRAMEWORK_RUNNERS = {
    "🤖 CrewAI (Part A)": _run_crewai,
    "🔗 LangChain (Part A)": _run_langchain,
    "📊 LangGraph (Part A)": _run_langgraph,
    "⚡ AG2 / AutoGen (Part B)": _run_autogen,
    "🐝 BeeAI (Part B PoC)": _run_beeai,
}


def handle_single_query(question: str, framework: str) -> tuple[str, str]:
    """Handler for the Ask tab."""
    if not question.strip():
        return "⚠️ Please enter a question.", ""
    if not os.getenv("BOB_API_KEY"):
        return (
            "⚠️ BOB_API_KEY not set.\n"
            "Create a .env file (see .env.example) and restart the app.",
            "",
        )
    runner = FRAMEWORK_RUNNERS[framework]
    answer = runner(question)
    return answer, f"✅ Answered using **{framework}**"


def handle_compare(question: str, compare_all: bool) -> tuple[str, str, str]:
    """Handler for the Compare tab."""
    if not question.strip():
        return "⚠️ Enter a question.", "⚠️ Enter a question.", "⚠️ Enter a question."
    if not os.getenv("BOB_API_KEY"):
        err = "⚠️ BOB_API_KEY not set."
        return err, err, err

    crewai_ans = _run_crewai(question)
    langchain_ans = _run_langchain(question)
    langgraph_ans = _run_langgraph(question)
    return crewai_ans, langchain_ans, langgraph_ans


def get_comparison_notes() -> str:
    from part_b.autogen_workflow import COMPARISON_TEXT as ag2_cmp
    from part_b.beeai_poc import COMPARISON_TEXT as bee_cmp
    return ag2_cmp + "\n\n---\n" + bee_cmp


# ── Sample questions ──────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "What are the admission requirements for graduate programs?",
    "What is the tuition fee for international undergraduate students?",
    "What departments does the university offer?",
    "When does the fall semester start and what are the important dates?",
    "What research opportunities are available and who are the industry partners?",
    "What courses are available in the Computer Science department?",
    "How can I contact the career center and what services do they offer?",
    "What on-campus facilities are available for students?",
]


# ── Build the Gradio interface ─────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="CampusAI Lite – Agentic University Assistant",
    ) as demo:

        gr.Markdown(
            """
# 🎓 CampusAI Lite
### Agentic University Information Assistant — TechVista University

> Powered by **CrewAI · LangChain · LangGraph · AG2/AutoGen · BeeAI · PydanticAI · Docling**
            """
        )

        with gr.Tabs():
            # ── Tab 1: Ask ───────────────────────────────────────────────────
            with gr.TabItem("💬 Ask CampusAI"):
                gr.Markdown("Ask any question about TechVista University and choose which AI framework to use.")

                with gr.Row():
                    with gr.Column(scale=3):
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What are the admission requirements for the graduate CS program?",
                            lines=3,
                        )
                    with gr.Column(scale=1):
                        framework_choice = gr.Dropdown(
                            label="Framework",
                            choices=list(FRAMEWORK_RUNNERS.keys()),
                            value="🔗 LangChain (Part A)",
                        )

                with gr.Row():
                    submit_btn = gr.Button("🚀 Ask", variant="primary", scale=1)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

                status_box = gr.Markdown("")
                answer_box = gr.Markdown(label="Answer")

                gr.Markdown("**💡 Try these sample questions:**")
                with gr.Row():
                    for i, sq in enumerate(SAMPLE_QUESTIONS[:4]):
                        gr.Button(sq[:60] + "…", size="sm").click(
                            fn=lambda q=sq: (q,),
                            outputs=[question_input],
                        )
                with gr.Row():
                    for i, sq in enumerate(SAMPLE_QUESTIONS[4:]):
                        gr.Button(sq[:60] + "…", size="sm").click(
                            fn=lambda q=sq: (q,),
                            outputs=[question_input],
                        )

                submit_btn.click(
                    fn=handle_single_query,
                    inputs=[question_input, framework_choice],
                    outputs=[answer_box, status_box],
                )
                clear_btn.click(
                    fn=lambda: ("", "", ""),
                    outputs=[question_input, answer_box, status_box],
                )

            # ── Tab 2: Compare ───────────────────────────────────────────────
            with gr.TabItem("⚖️ Compare Frameworks"):
                gr.Markdown(
                    "Compare how **CrewAI**, **LangChain**, and **LangGraph** "
                    "answer the same question side by side."
                )

                compare_q = gr.Textbox(
                    label="Question",
                    placeholder="Enter a question to compare all three Part A frameworks…",
                    lines=2,
                )
                compare_btn = gr.Button("🔄 Compare All Three", variant="primary")

                with gr.Row():
                    crewai_out = gr.Markdown(label="🤖 CrewAI Answer")
                    langchain_out = gr.Markdown(label="🔗 LangChain Answer")
                    langgraph_out = gr.Markdown(label="📊 LangGraph Answer")

                compare_btn.click(
                    fn=handle_compare,
                    inputs=[compare_q, gr.Checkbox(value=True, visible=False)],
                    outputs=[crewai_out, langchain_out, langgraph_out],
                )

            # ── Tab 3: Framework Comparison Notes (Part B) ───────────────────
            with gr.TabItem("📋 Part B – Framework Comparison"):
                gr.Markdown(
                    "## Part B: Framework Exploration Notes\n"
                    "Comparison tables between **AG2 (AutoGen)** and **BeeAI** vs **CrewAI**."
                )
                notes_box = gr.Markdown()
                load_btn = gr.Button("Load Comparison Notes")
                load_btn.click(fn=get_comparison_notes, outputs=[notes_box])
                # Load on render
                demo.load(fn=get_comparison_notes, outputs=[notes_box])

            # ── Tab 4: About ─────────────────────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown(
                    """
## About CampusAI Lite

**CampusAI Lite** is a Capstone Project demonstrating an **Agentic AI** system for
answering university-related queries.

### Architecture
```
Student Question
      │
      ▼
 ┌─────────────────────────────────────┐
 │        PydanticAI Validation        │  ← Structured I/O validation
 └──────────────────┬──────────────────┘
                    │
      ┌─────────────┼─────────────────┐
      ▼             ▼                 ▼
 CrewAI        LangChain          LangGraph
 (3 agents)    (LCEL chain)       (ReAct graph)
      │             │                 │
      └─────────────┼─────────────────┘
                    │
      ┌─────────────┴─────────────────┐
      │         Custom Tools          │
      │  UniversityInfoTool           │  ← Local JSON knowledge base
      │  DoclingDocumentTool          │  ← PDF/DOCX parsing with Docling
      └───────────────────────────────┘
```

### Frameworks Used
| Framework   | Role                                        | Part  |
|-------------|---------------------------------------------|-------|
| CrewAI      | Multi-agent crew (Planner, Info, Validator) | A     |
| LangChain   | LCEL chain + OpenAI Functions agent         | A     |
| LangGraph   | ReAct-style state graph workflow            | A     |
| PydanticAI  | Structured input/output validation models   | A     |
| Docling     | University document parsing tool            | A     |
| AG2/AutoGen | Group-chat multi-agent workflow             | B     |
| BeeAI       | Two-agent ReAct proof of concept            | B     |
| Gradio      | Interactive web UI                          | A     |

### University: TechVista University (Fictional)
This demo uses a fictional university database. Replace `data/university_info.json`
and `data/sample_document.md` with real documents to deploy for a real institution.

### Setup
```bash
cd campusai_lite
pip install -r requirements.txt
cp .env.example .env   # Add your BOB_API_KEY
python main.py
```
                    """
                )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────
def launch(share: bool = False, port: int = 7860):
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    launch()
