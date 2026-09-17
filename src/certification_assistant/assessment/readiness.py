"""Deterministic readiness API.

Public entry point for readiness classification (Ready / Nearly Ready / Needs
Improvement). The threshold logic lives in ``evaluation.calculate_readiness``;
this module re-exports it.
"""

from .evaluation import calculate_readiness

__all__ = ["calculate_readiness"]

