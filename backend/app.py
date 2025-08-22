# backend/app.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from .orchestrator import run_query
# from .mcp_agent import run_agent_query  # <-- add

app = FastAPI(title="Salesforce Chat API")

class ChatRequest(BaseModel):
    prompt: str

@app.post("/salesforce-chat")
def salesforce_chat(req: ChatRequest):
    return run_query(req.prompt)

# # New: MCP Agent endpoint (Task 4 demo)
# @app.post("/mcp-agent-chat")
# def mcp_agent_chat(req: ChatRequest):
#     return run_agent_query(req.prompt)

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
