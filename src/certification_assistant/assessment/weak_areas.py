"""Deterministic weak-area API.

Public entry point for detecting weak topics/domains from a scored assessment. The
implementation lives in ``evaluation.detect_weak_areas``; this module re-exports it.
"""

from .evaluation import detect_weak_areas

__all__ = ["detect_weak_areas"]

