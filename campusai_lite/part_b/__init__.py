"""campusai_lite/part_b/__init__.py"""
from .autogen_workflow import run_autogen, COMPARISON_TEXT as AUTOGEN_VS_CREWAI
from .beeai_poc import run_beeai, COMPARISON_TEXT as BEEAI_VS_CREWAI

__all__ = ["run_autogen", "run_beeai", "AUTOGEN_VS_CREWAI", "BEEAI_VS_CREWAI"]
