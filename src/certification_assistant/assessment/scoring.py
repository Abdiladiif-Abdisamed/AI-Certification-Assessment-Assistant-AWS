"""Deterministic scoring API.

Public entry point for scoring an assessment. The actual (LLM-free) implementation
lives in ``evaluation.evaluate_assessment``; this module re-exports it so callers
import from a stable, intention-revealing path.
"""

from .evaluation import evaluate_assessment

__all__ = ["evaluate_assessment"]

