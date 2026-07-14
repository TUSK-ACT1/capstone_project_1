"""
campusai_lite/main.py
Main entry point for CampusAI Lite.

Usage:
    python main.py                     # Launch Gradio UI (default)
    python main.py --cli               # CLI mode (interactive)
    python main.py --demo crewai       # Run a one-shot demo question
    python main.py --demo langgraph
    python main.py --demo langchain
    python main.py --demo autogen
    python main.py --demo beeai
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Ensure package is on path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DEMO_QUESTION = "What are the admission requirements and tuition fees for the graduate Computer Science program?"


# ── CLI helpers ───────────────────────────────────────────────────────────────
def _check_api_key():
    if not os.getenv("BOB_API_KEY"):
        print(
            "❌  BOB_API_KEY is not set.\n"
            "    Copy .env.example → .env and add your key, then re-run.\n"
        )
        sys.exit(1)


def run_demo(framework: str, question: str = DEMO_QUESTION):
    _check_api_key()
    print(f"\n{'='*60}")
    print(f"  CampusAI Lite — {framework.upper()} Demo")
    print(f"{'='*60}")
    print(f"Question: {question}\n")

    if framework == "crewai":
        from agents.crewai_agents import run_crewai
        answer = run_crewai(question)
    elif framework == "langchain":
        from agents.langchain_chain import run_langchain
        answer = run_langchain(question)
    elif framework == "langgraph":
        from workflows.langgraph_workflow import run_langgraph
        answer = run_langgraph(question)
    elif framework == "autogen":
        from part_b.autogen_workflow import run_autogen, COMPARISON_TEXT
        answer = run_autogen(question)
        print("\n" + COMPARISON_TEXT)
    elif framework == "beeai":
        from part_b.beeai_poc import run_beeai, COMPARISON_TEXT
        answer = run_beeai(question)
        print("\n" + COMPARISON_TEXT)
    else:
        print(f"Unknown framework '{framework}'. Choose: crewai, langchain, langgraph, autogen, beeai")
        return

    print(f"\n{'─'*60}")
    print("Answer:")
    print(answer)
    print(f"{'─'*60}\n")


def run_cli():
    _check_api_key()
    print("\n🎓 CampusAI Lite — Interactive CLI Mode")
    print("Type 'exit' to quit. Type 'switch <framework>' to change framework.")
    print("Frameworks: crewai, langchain, langgraph, autogen, beeai\n")

    current_framework = "langchain"
    print(f"Using: {current_framework} (default)\n")

    while True:
        try:
            question = input("📚 Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if question.lower().startswith("switch "):
            fw = question.split(" ", 1)[1].strip()
            current_framework = fw
            print(f"✅ Switched to: {current_framework}\n")
            continue

        run_demo(current_framework, question)


def run_ui(port: int = 7860, share: bool = False):
    from ui.gradio_app import launch
    print(f"\n🎓 CampusAI Lite — Launching Gradio UI on port {port}…")
    print("    Open your browser at: http://localhost:{}\n".format(port))
    launch(share=share, port=port)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CampusAI Lite – Agentic University Information Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cli", action="store_true", help="Run in interactive CLI mode.")
    parser.add_argument(
        "--demo",
        metavar="FRAMEWORK",
        help="Run a one-shot demo with the specified framework.",
    )
    parser.add_argument(
        "--question",
        metavar="QUESTION",
        default=DEMO_QUESTION,
        help="Question to use in demo mode.",
    )
    parser.add_argument("--port", type=int, default=7860, help="Gradio UI port (default: 7860).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")

    args = parser.parse_args()

    if args.demo:
        run_demo(args.demo, args.question)
    elif args.cli:
        run_cli()
    else:
        run_ui(port=args.port, share=args.share)
