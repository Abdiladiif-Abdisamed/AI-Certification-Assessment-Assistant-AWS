"""Study-plan API.

Public entry point for building a personalized study plan. The implementation lives
in ``ai.engine.generate_study_plan`` (deterministic day-by-day plan, with optional
OpenAI wording polish); this module re-exports it.
"""

from ..ai.engine import generate_study_plan

__all__ = ["generate_study_plan"]

