"""
save_graph.py
=============
Renders the MiFID II pipeline graph as PNG using mermaid.ink.

Usage:
  python save_graph.py
"""

import base64
import httpx
from config.llm_config import get_model_client, get_verifier_client, get_disclosure_client
from orchestrator.graph import build_graph

graph = build_graph(get_model_client(), get_verifier_client(), get_disclosure_client())

mermaid_source = graph.get_graph().draw_mermaid()

print("Mermaid source:\n")
print(mermaid_source)

encoded = base64.urlsafe_b64encode(mermaid_source.encode()).decode()
url = f"https://mermaid.ink/img/{encoded}"

print("\nFetching PNG from mermaid.ink...")
response = httpx.get(url, timeout=30)
response.raise_for_status()

with open("pipeline_graph.png", "wb") as f:
    f.write(response.content)

print("Saved: pipeline_graph.png")
