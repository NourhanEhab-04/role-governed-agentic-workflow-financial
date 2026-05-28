# orchestrator/orchestrator.py
"""
Canonical pipeline entry point — delegates to orchestrator/graph.py.

The LangGraph StateGraph in graph.py is the single source of truth.
This module exists so that main.py and any legacy callers continue to work
without change: `from orchestrator.orchestrator import run_pipeline`.
"""

from orchestrator.graph import run_pipeline, build_graph

__all__ = ["run_pipeline", "build_graph"]
