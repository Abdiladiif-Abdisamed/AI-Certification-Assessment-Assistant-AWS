"""Personalized recommendation API.

Public entry point for turning weak areas into targeted study recommendations. The
implementation lives in ``ai.engine.generate_recommendations`` (deterministic draft
with optional OpenAI wording polish); this module re-exports it.
"""

from ..ai.engine import generate_recommendations

__all__ = ["generate_recommendations"]

