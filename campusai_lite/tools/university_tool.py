"""
campusai_lite/tools/university_tool.py
Custom LangChain tool that looks up university information from a local JSON store.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# ── Data load ────────────────────────────────────────────────────────────────
_DATA_PATH = Path(__file__).parent.parent / "data" / "university_info.json"

def _load_data() -> dict:
    with open(_DATA_PATH, "r") as f:
        return json.load(f)


# ── Input schema (PydanticAI-style structured I/O) ───────────────────────────
class UniversityQueryInput(BaseModel):
    """Structured input for the UniversityInfoTool."""
    topic: str = Field(
        ...,
        description=(
            "The university topic to look up. "
            "Accepted values: 'general', 'departments', 'admissions', "
            "'tuition', 'facilities', 'calendar', 'services', 'research', "
            "or a specific department name like 'Computer Science'."
        ),
    )
    sub_filter: Optional[str] = Field(
        None,
        description="Optional secondary filter, e.g. 'undergraduate' or 'graduate'.",
    )


class UniversityInfoTool(BaseTool):
    """
    Custom tool: looks up structured university information.
    Returns a human-readable summary for the specified topic.
    """

    name: str = "university_info"
    description: str = (
        "Retrieve detailed information about TechVista University. "
        "Topics: general, departments, admissions, tuition, facilities, "
        "calendar, services, research, or a specific department name."
    )
    args_schema: type[BaseModel] = UniversityQueryInput

    def _run(self, topic: str, sub_filter: Optional[str] = None) -> str:  # type: ignore[override]
        data = _load_data()
        topic_lower = topic.lower().strip()

        # ── General university info ──
        if topic_lower in ("general", "overview", "about"):
            u = data["university"]
            return (
                f"**{u['name']}** (Est. {u['established']})\n"
                f"Location: {u['location']}\n"
                f"Website: {u['website']}\n"
                f"Contact: {u['contact']}"
            )

        # ── Departments ──
        if topic_lower in ("departments", "programs", "courses"):
            depts = data["departments"]
            if sub_filter:
                dept = next(
                    (d for d in depts if sub_filter.lower() in d["name"].lower()), None
                )
                if dept:
                    programs = ", ".join(dept["programs"])
                    credits = json.dumps(dept["credits_required"])
                    return (
                        f"**{dept['name']}**\n"
                        f"Head: {dept['head']}\n"
                        f"Programs: {programs}\n"
                        f"Credits required: {credits}"
                    )
                return f"No department found matching '{sub_filter}'."
            lines = [
                f"- {d['name']} (Head: {d['head']}, Programs: {', '.join(d['programs'])})"
                for d in depts
            ]
            return "Available Departments:\n" + "\n".join(lines)

        # ── Admissions ──
        if topic_lower in ("admissions", "admission", "apply", "application"):
            adm = data["admissions"]
            level = sub_filter.lower() if sub_filter else None
            if level in ("undergraduate", "ug", "bachelors"):
                a = adm["undergraduate"]
            elif level in ("graduate", "grad", "masters", "phd"):
                a = adm["graduate"]
            else:
                # Return both
                return (
                    "**Undergraduate Admissions**\n"
                    f"Opens: {adm['undergraduate']['open_date']}  |  Deadline: {adm['undergraduate']['deadline']}\n"
                    f"Min GPA: {adm['undergraduate']['min_gpa']}\n"
                    f"Requirements: {', '.join(adm['undergraduate']['requirements'])}\n\n"
                    "**Graduate Admissions**\n"
                    f"Opens: {adm['graduate']['open_date']}  |  Deadline: {adm['graduate']['deadline']}\n"
                    f"Min GPA: {adm['graduate']['min_gpa']}\n"
                    f"Requirements: {', '.join(adm['graduate']['requirements'])}"
                )
            return (
                f"Opens: {a['open_date']}  |  Deadline: {a['deadline']}\n"
                f"Min GPA: {a['min_gpa']}\n"
                f"Requirements: {', '.join(a['requirements'])}"
            )

        # ── Tuition ──
        if topic_lower in ("tuition", "fees", "cost", "financial"):
            t = data["tuition"]
            return (
                "**Tuition (Annual)**\n"
                f"Undergraduate — In-state: ${t['undergraduate']['in_state']:,}  "
                f"Out-of-state: ${t['undergraduate']['out_of_state']:,}  "
                f"International: ${t['undergraduate']['international']:,}\n"
                f"Graduate — In-state: ${t['graduate']['in_state']:,}  "
                f"Out-of-state: ${t['graduate']['out_of_state']:,}  "
                f"International: ${t['graduate']['international']:,}\n"
                f"Financial Aid: {t['financial_aid']}"
            )

        # ── Facilities ──
        if topic_lower in ("facilities", "campus", "buildings"):
            facilities = data["facilities"]
            return "Campus Facilities:\n" + "\n".join(f"  • {f}" for f in facilities)

        # ── Academic calendar ──
        if topic_lower in ("calendar", "schedule", "semester", "dates"):
            cal = data["academic_calendar"]
            holidays = ", ".join(cal["holidays"])
            return (
                f"Fall Semester: {cal['fall_semester']['start']} – {cal['fall_semester']['end']}\n"
                f"Spring Semester: {cal['spring_semester']['start']} – {cal['spring_semester']['end']}\n"
                f"Summer Session: {cal['summer_session']['start']} – {cal['summer_session']['end']}\n"
                f"Holidays: {holidays}"
            )

        # ── Student services ──
        if topic_lower in ("services", "student services", "support", "contact"):
            svcs = data["student_services"]
            lines = [f"  • {k.replace('_', ' ').title()}: {v}" for k, v in svcs.items()]
            return "Student Services:\n" + "\n".join(lines)

        # ── Research ──
        if topic_lower in ("research", "publications", "funding"):
            r = data["research"]
            partners = ", ".join(r["industry_partners"])
            return (
                f"Annual Research Funding: {r['annual_funding']}\n"
                f"Active Projects: {r['active_projects']}\n"
                f"2024 Publications: {r['publication_count_2024']}\n"
                f"Industry Partners: {partners}"
            )

        # ── Specific department name ──
        for dept in data["departments"]:
            if topic_lower in dept["name"].lower():
                programs = ", ".join(dept["programs"])
                credits = json.dumps(dept["credits_required"])
                return (
                    f"**{dept['name']}**\n"
                    f"Head: {dept['head']}\n"
                    f"Programs: {programs}\n"
                    f"Credits required: {credits}"
                )

        return (
            f"No information found for topic '{topic}'. "
            "Try: general, departments, admissions, tuition, facilities, "
            "calendar, services, research, or a department name."
        )

    async def _arun(self, topic: str, sub_filter: Optional[str] = None) -> str:  # type: ignore[override]
        return self._run(topic, sub_filter)
