# api.py

import asyncio
import json
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from orchestrator.graph import run_pipeline
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
        return serialize_state(dataclasses.asdict(state))
    return state


_STRIP = {"client_input", "product_input"}


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/assess")
async def assess(req: AssessmentRequest):
    try:
        state, _audit_log = await run_pipeline(
            client_input=req.client_input,
            product_input=req.product_input,
            model_client=get_model_client(),
        )
        filtered = {k: v for k, v in state.items() if k not in _STRIP}
        return serialize_state(filtered)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assess/stream")
async def assess_stream(req: AssessmentRequest):
    """
    SSE endpoint — streams pipeline state after each major stage.

    Events are newline-delimited JSON:
      data: {"type": "<stage>", "state": {...}}\n\n

    Terminated with:
      data: [DONE]\n\n
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def on_event(event_type: str, state_snapshot: dict):
        filtered = {k: v for k, v in state_snapshot.items() if k not in _STRIP}
        payload = {"type": event_type, "state": serialize_state(filtered)}
        await queue.put(payload)

    async def pipeline_task():
        try:
            await run_pipeline(
                client_input=req.client_input,
                product_input=req.product_input,
                model_client=get_model_client(),
                on_event=on_event,
            )
        except Exception as e:
            traceback.print_exc()
            await queue.put({"type": "error", "detail": str(e)})
        finally:
            await queue.put(None)  # sentinel — signals stream end

    async def event_generator():
        task = asyncio.create_task(pipeline_task())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}
