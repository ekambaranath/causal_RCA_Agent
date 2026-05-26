"""FastAPI server — exposes the Causal RCA pipeline via HTTP."""

import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.chains.graph import run_pipeline
from src.tools.memory_tools import store_incident
from src.utils.logging import get_logger, setup_logging

setup_logging()
log = get_logger("api")

app = FastAPI(title="Causal RCA Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response models ─────────────────────────────────────────────────

class IncidentRequest(BaseModel):
    description: str = Field(..., min_length=5, description="Plain-English incident description")
    service:     str = Field(default="unknown", description="Primary affected service")
    error_rate:  float = Field(default=0.0, ge=0, le=100)


class PipelineResponse(BaseModel):
    incident_id:   str
    root_cause:    str
    confidence:    float
    intervention:  str
    ttr_minutes:   int
    interventions: list
    escalate:      bool
    retries_used:  int
    memory_used:   bool
    steps:         list
    elapsed_s:     float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "causal-rca-agent"}


@app.post("/analyze", response_model=PipelineResponse)
async def analyze_incident(req: IncidentRequest):
    incident = {
        "id":          str(uuid.uuid4())[:8],
        "description": req.description,
        "service":     req.service,
        "error_rate":  req.error_rate,
    }

    log.info("api_analyze", incident_id=incident["id"], description=req.description[:60])
    start = time.time()

    try:
        state = run_pipeline(incident)
    except Exception as e:
        log.error("pipeline_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    verdict       = state.get("verdict") or {}
    interventions = state.get("interventions") or []
    elapsed       = round(time.time() - start, 2)

    # Persist resolved incident to memory
    if verdict.get("confidence", 0) >= 0.70 and not verdict.get("escalate"):
        try:
            store_incident(incident, state.get("evidence") or {}, verdict)
        except Exception as e:
            log.warning("memory_store_failed", error=str(e))

    log.info(
        "api_complete",
        incident_id=incident["id"],
        elapsed_s=elapsed,
        confidence=verdict.get("confidence"),
        escalate=verdict.get("escalate"),
    )

    return PipelineResponse(
        incident_id=incident["id"],
        root_cause=verdict.get("root_cause", "Unknown"),
        confidence=verdict.get("confidence", 0.0),
        intervention=verdict.get("intervention", "Manual investigation required"),
        ttr_minutes=verdict.get("ttr_minutes", 0),
        interventions=interventions,
        escalate=verdict.get("escalate", False),
        retries_used=verdict.get("retries_used", 0),
        memory_used=verdict.get("memory_used", False),
        steps=state.get("steps", []),
        elapsed_s=elapsed,
    )
