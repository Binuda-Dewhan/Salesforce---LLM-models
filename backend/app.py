import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional

from .orchestrator import run_query, format_notes_with_lora

app = FastAPI(
    title="Salesforce AI & Notes Formatter API",
    description="Enterprise API integrating local Gemma-2B LoRA notes formatter with Salesforce REST and MCP Protocol.",
    version="1.0.0"
)

# Enable CORS for frontend web demo clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static frontend assets at /dashboard if directory exists
from fastapi.staticfiles import StaticFiles
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/dashboard", StaticFiles(directory=frontend_dir, html=True), name="frontend")


class ChatRequest(BaseModel):
    prompt: str


class FormatNotesRequest(BaseModel):
    raw_notes: str


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "Salesforce AI Integration API",
        "lora_adapter": "salesforce_notes_adapter (Gemma-2B-IT)",
        "salesforce_instance": os.getenv("SF_INSTANCE_URL", "Configured")
    }


@app.post("/salesforce-chat")
def salesforce_chat(req: ChatRequest) -> Dict[str, Any]:
    """
    Main conversational endpoint:
    Intelligently routes requests to Salesforce search, note creation, or local LoRA formatter.
    """
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    try:
        return run_query(req.prompt.strip())
    except Exception as e:
        return {
            "type": "error",
            "error": str(e)
        }


@app.post("/format-notes")
def format_notes_endpoint(req: FormatNotesRequest) -> Dict[str, Any]:
    """
    Direct endpoint for formatting unstructured CRM notes using the fine-tuned LoRA model.
    """
    if not req.raw_notes or not req.raw_notes.strip():
        raise HTTPException(status_code=400, detail="raw_notes cannot be empty.")
    try:
        formatted = format_notes_with_lora(req.raw_notes.strip())
        return {
            "status": "success",
            "raw_notes": req.raw_notes,
            "formatted_notes": formatted
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Formatting failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
