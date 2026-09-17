"""
backend/app.py
==============
FastAPI application entry point for the Salesforce AI Integration API.

Endpoints:
  GET  /health            — Health check
  POST /salesforce-chat   — Main conversational ReAct agent
  POST /format-notes      — Direct LoRA notes formatter
  GET  /dashboard/*       — Serves static frontend assets
"""

import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict

from .config import cfg
from .orchestrator import format_notes_with_lora, run_query

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# Configure once at the application entry point.
# All modules use logging.getLogger(__name__) and inherit this root config.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Salesforce AI & Notes Formatter API",
    description=(
        "Enterprise API integrating a local Gemma-2B LoRA notes formatter with "
        "Salesforce REST and the Model Context Protocol (MCP)."
    ),
    version="1.0.0",
)

# CORS — restricted to configured origins (set CORS_ALLOWED_ORIGINS in .env)
allowed_origins = cfg.cors_allowed_origins()
logger.info("CORS allowed origins: %s", allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Serve static frontend at /dashboard if the directory exists
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(_frontend_dir), html=True),
        name="frontend",
    )
    logger.info("Frontend dashboard mounted from %s", _frontend_dir)
else:
    logger.warning("Frontend directory not found at %s; /dashboard will not be available.", _frontend_dir)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str


class FormatNotesRequest(BaseModel):
    raw_notes: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Returns service health status and basic configuration info."""
    return {
        "status": "online",
        "service": "Salesforce AI Integration API",
        "lora_adapter": "salesforce_notes_adapter (Gemma-2B-IT)",
        "salesforce_instance": cfg.salesforce_instance_url() or "Not configured",
    }


@app.post("/salesforce-chat")
def salesforce_chat(req: ChatRequest) -> Dict[str, Any]:
    """
    Main conversational endpoint.
    Routes the prompt through the Grounded ReAct Agent loop and returns
    a structured response with an executive answer and full tool audit trace.
    """
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    logger.info("Chat request received | prompt=%r", req.prompt[:120])
    try:
        return run_query(req.prompt.strip())
    except Exception as exc:  # noqa: BLE001
        logger.error("Unhandled error in salesforce_chat: %s", exc, exc_info=True)
        return {"type": "error", "error": str(exc)}


@app.post("/format-notes")
def format_notes_endpoint(req: FormatNotesRequest) -> Dict[str, Any]:
    """
    Direct endpoint for formatting unstructured CRM notes via the fine-tuned LoRA model.
    Returns the raw input alongside the formatted structured output.
    """
    if not req.raw_notes or not req.raw_notes.strip():
        raise HTTPException(status_code=400, detail="raw_notes cannot be empty.")

    logger.info("Format-notes request received | length=%d chars", len(req.raw_notes))
    try:
        formatted = format_notes_with_lora(req.raw_notes.strip())
        return {
            "status": "success",
            "raw_notes": req.raw_notes,
            "formatted_notes": formatted,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("LoRA formatting failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Formatting failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
