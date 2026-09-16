"""
Model Context Protocol (MCP) Server
===================================
A standalone JSON-RPC 2.0 microservice exposing Salesforce CRM operations
and LoRA Notes Formatting as standard MCP Tools.

Protocol: Model Context Protocol over HTTP (JSON-RPC 2.0)
Default Port: 5000
Endpoint: POST /rpc
"""

import os
import uvicorn
from fastapi import FastAPI, Request
from typing import Dict, Any

from .salesforce_client import SalesforceClient
from .orchestrator import format_notes_with_lora
from .tools import TOOLS

app = FastAPI(
    title="Salesforce MCP Server",
    description="JSON-RPC 2.0 Model Context Protocol Server for Salesforce CRM Tools",
    version="1.0.0"
)

sf_client = SalesforceClient()

TOOL_HANDLERS = {
    "searchOpportunities": lambda a: sf_client.search_opportunities(a["opportunityName"]),
    "getOpportunity":      lambda a: sf_client.get_opportunity(a["opportunityId"]),
    "searchAccounts":      lambda a: sf_client.search_accounts(a["accountName"]),
    "createNotes":         lambda a: sf_client.create_note(
        a["opportunityId"], a["newNoteToAdd"], a.get("title", "Meeting Notes")
    ),
    "getLatestNotes":      lambda a: sf_client.get_latest_notes(a["opportunityId"], a.get("limit", 3)),
    "formatNotes":         lambda a: format_notes_with_lora(a["rawText"]),
    "executeSOQL":         lambda a: sf_client.execute_soql(a["query"]),
}


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "online",
        "protocol": "Model Context Protocol (JSON-RPC 2.0)",
        "available_tools": list(TOOL_HANDLERS.keys())
    }


@app.get("/tools")
def list_tools():
    """Returns the MCP Tool Manifest."""
    return {"tools": TOOLS}


@app.post("/rpc")
async def handle_rpc(request: Request) -> Dict[str, Any]:
    """
    Main MCP JSON-RPC 2.0 Request Dispatcher:
    Accepts: {"jsonrpc": "2.0", "method": "tools/toolName", "params": {...}, "id": 1}
    Returns: {"jsonrpc": "2.0", "result": {...}, "id": 1}
    """
    body = await request.json()
    req_id = body.get("id", 1)
    method = body.get("method", "")
    params = body.get("params", {})

    # Extract tool name from method (e.g. "tools/searchOpportunities" -> "searchOpportunities")
    tool_name = method.replace("tools/", "").strip()

    if tool_name not in TOOL_HANDLERS:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32601,
                "message": f"Method '{method}' not found. Available: {list(TOOL_HANDLERS.keys())}"
            },
            "id": req_id
        }

    try:
        handler = TOOL_HANDLERS[tool_name]
        result = handler(params)
        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": req_id
        }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32000,
                "message": f"Execution error in '{tool_name}': {str(e)}"
            },
            "id": req_id
        }


if __name__ == "__main__":
    uvicorn.run("backend.mcp_server:app", host="0.0.0.0", port=5000, reload=True)
