# api.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

from orchestrator.orchestrator import run_pipeline
from config.llm_config import get_model_client


# ── Request schema ─────────────────────────────────────────────────────────────

class AssessmentRequest(BaseModel):
    client_input: str
    product_input: str


# ── App + CORS ─────────────────────────────────────────────────────────────────

app = FastAPI(title="MiFID II Suitability API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def serialize_state(state: dict) -> dict:
    """
    Recursively convert any non-JSON-serializable values in state.
    Handles: dataclasses, Enum members, objects with __dict__.
    """
    import dataclasses
    from enum import Enum

    if isinstance(state, dict):
        return {k: serialize_state(v) for k, v in state.items()}
    if isinstance(state, list):
        return [serialize_state(i) for i in state]
    if isinstance(state, Enum):
        return state.value
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        # Convert dataclass to dict, rename pass_ → pass_ stays as-is
        # (frontend reads pass_ directly)
        return serialize_state(dataclasses.asdict(state))
    return state


# ── Endpoint ───────────────────────────────────────────────────────────────────

@app.post("/assess")
async def assess(req: AssessmentRequest):
    try:
        state, _audit_log = await run_pipeline(
            client_input=req.client_input,
            product_input=req.product_input,
            model_client=get_model_client(),
        )
        return serialize_state(state)

    except Exception as e:
        traceback.print_exc()          # full trace in your terminal
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check (useful for quick curl tests) ─────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}