"""
campusai_lite/tools/docling_tool.py
Custom LangChain tool that parses university documents (PDF, DOCX, MD, HTML)
using Docling and returns structured text for the agents.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class DoclingParseInput(BaseModel):
    """Input schema for the document parsing tool."""
    file_path: str = Field(..., description="Absolute or relative path to the document to parse.")
    query: Optional[str] = Field(
        None,
        description="Optional keyword or phrase to filter relevant sections from the parsed document.",
    )


class DoclingDocumentTool(BaseTool):
    """
    Parses university documents (PDF, DOCX, HTML, MD) with Docling and
    returns their text content, optionally filtered by a keyword query.
    """

    name: str = "parse_university_document"
    description: str = (
        "Parse a university document (PDF, DOCX, Markdown, or HTML) and return its text. "
        "Provide a file path and an optional keyword query to filter sections."
    )
    args_schema: type[BaseModel] = DoclingParseInput

    def _run(self, file_path: str, query: Optional[str] = None) -> str:  # type: ignore[override]
        path = Path(file_path)
        if not path.exists():
            # Try relative to the data folder
            alt = Path(__file__).parent.parent / "data" / file_path
            if alt.exists():
                path = alt
            else:
                return f"File not found: {file_path}"

        suffix = path.suffix.lower()

        # ── Docling for PDF / DOCX / HTML ──────────────────────────────────
        if suffix in (".pdf", ".docx", ".html", ".htm"):
            try:
                from docling.document_converter import DocumentConverter  # type: ignore

                converter = DocumentConverter()
                result = converter.convert(str(path))
                text = result.document.export_to_markdown()
            except ImportError:
                return (
                    "Docling is not installed. Install it with: pip install docling\n"
                    f"Cannot parse {path.name}."
                )
            except Exception as exc:
                return f"Docling failed to parse '{path.name}': {exc}"

        # ── Plain text / Markdown – read directly ──────────────────────────
        elif suffix in (".md", ".txt", ".rst"):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception as exc:
                return f"Failed to read '{path.name}': {exc}"
        else:
            return f"Unsupported file type '{suffix}'. Supported: PDF, DOCX, HTML, MD, TXT."

        # ── Optional keyword filter ─────────────────────────────────────────
        if query:
            lines = text.splitlines()
            query_lower = query.lower()
            relevant = [
                line for line in lines if query_lower in line.lower()
            ]
            if relevant:
                return f"Sections matching '{query}' in {path.name}:\n\n" + "\n".join(relevant)
            return f"No sections matching '{query}' found in {path.name}.\n\nFull content:\n{text[:3000]}"

        return f"Parsed content of {path.name}:\n\n{text[:5000]}"

    async def _arun(self, file_path: str, query: Optional[str] = None) -> str:  # type: ignore[override]
        return self._run(file_path, query)
