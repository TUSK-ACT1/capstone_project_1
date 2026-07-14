"""
campusai_lite/agents/pydantic_models.py
PydanticAI – structured validation models for agent inputs and outputs.
Used across all agents to ensure type-safe, validated I/O.
"""

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


# ── Input ─────────────────────────────────────────────────────────────────────
class StudentQuery(BaseModel):
    """Validated student query input."""
    question: str = Field(..., min_length=5, max_length=1000, description="The student's question.")
    context: Optional[str] = Field(None, description="Optional background context.")
    preferred_detail_level: Literal["brief", "detailed", "comprehensive"] = Field(
        "detailed", description="Desired depth of the answer."
    )

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be blank.")
        return v.strip()


# ── Planner output ────────────────────────────────────────────────────────────
class SubTask(BaseModel):
    """A single information-retrieval sub-task."""
    topic: str = Field(..., description="Topic to query (e.g. 'admissions', 'tuition').")
    sub_filter: Optional[str] = Field(None, description="Secondary filter (e.g. 'graduate').")
    document_to_parse: Optional[str] = Field(None, description="Document filename if parsing needed.")


class PlannerOutput(BaseModel):
    """Structured output from the Planner agent."""
    original_question: str
    sub_tasks: List[SubTask] = Field(..., min_length=1)
    notes: Optional[str] = Field(None, description="Any additional notes for the Information agent.")


# ── Information retrieval output ──────────────────────────────────────────────
class RetrievedInfo(BaseModel):
    """A single retrieved information snippet."""
    topic: str
    content: str
    source: Literal["university_db", "document", "llm_inference"] = "university_db"


class InformationOutput(BaseModel):
    """Structured output from the Information agent."""
    question: str
    retrieved_items: List[RetrievedInfo]
    combined_answer: str = Field(..., description="Combined prose answer from all retrieved info.")


# ── Validation output ─────────────────────────────────────────────────────────
class ValidationIssue(BaseModel):
    """A single validation issue found."""
    severity: Literal["error", "warning", "suggestion"]
    description: str


class ValidationOutput(BaseModel):
    """Structured output from the Validation agent."""
    original_question: str
    is_complete: bool = Field(..., description="Does the answer fully address the question?")
    is_accurate: bool = Field(..., description="Is the information factually correct?")
    issues: List[ValidationIssue] = Field(default_factory=list)
    final_answer: str = Field(..., description="The validated, polished final answer.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1.")


# ── PydanticAI agent helper ───────────────────────────────────────────────────
def validate_student_query(raw_question: str, detail_level: str = "detailed") -> StudentQuery:
    """Validate and parse a raw student question into a StudentQuery model."""
    return StudentQuery(
        question=raw_question,
        preferred_detail_level=detail_level,  # type: ignore[arg-type]
    )


def build_validation_output(
    question: str,
    final_answer: str,
    issues: Optional[List[ValidationIssue]] = None,
    confidence: float = 0.9,
) -> ValidationOutput:
    """Convenience builder for ValidationOutput."""
    has_issues = bool(issues and any(i.severity == "error" for i in issues))
    return ValidationOutput(
        original_question=question,
        is_complete=not has_issues,
        is_accurate=not has_issues,
        issues=issues or [],
        final_answer=final_answer,
        confidence_score=confidence,
    )
