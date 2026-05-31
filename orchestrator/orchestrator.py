# orchestrator/orchestrator.py
"""
Canonical pipeline entry point — delegates to orchestrator/graph.py.

The LangGraph StateGraph in graph.py is the single source of truth.
This module exists so that main.py and any legacy callers continue to work
without change: `from orchestrator.orchestrator import run_pipeline`.
"""

from orchestrator.graph import run_pipeline, build_graph
from orchestrator.pre_check_tool import run_pre_check

# Verifier always runs in the current architecture (no sampling).
# Exposed here so legacy test patches targeting this module still find the attribute.
_VERIFIER_SAMPLE_RATE: float = 1.0

__all__ = ["run_pipeline", "build_graph", "run_pre_check", "_VERIFIER_SAMPLE_RATE"]
