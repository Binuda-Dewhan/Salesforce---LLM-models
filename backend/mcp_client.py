import os
import json
from typing import Dict, Any, Optional
import requests

def load_env_file(filepath=".env"):
    candidates = [filepath, os.path.join("..", filepath), os.path.join(os.path.dirname(__file__), "..", ".env")]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().split("#")[0].strip()
                        if k not in os.environ or not os.environ[k]:
                            os.environ[k] = v
            break

load_env_file()

USE_MCP = os.getenv("USE_MCP", "false").lower() == "true"
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:5000").rstrip("/")


class MCPClient:
    """
    Model Context Protocol (MCP) Client:
    Sends JSON-RPC tool calls to a local or remote MCP Server.
    Falls back gracefully if the server is offline.
    """

    def __init__(self, server_url: str = MCP_SERVER_URL):
        self.server_url = server_url

    def call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invokes an MCP tool via JSON-RPC 2.0 protocol over HTTP.
        """
        payload = {
            "jsonrpc": "2.0",
            "method": f"tools/{tool_name}",
            "params": args,
            "id": 1
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                f"{self.server_url}/rpc",
                json=payload,
                headers=headers,
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("result", data)
            else:
                return {
                    "tool": tool_name,
                    "args": args,
                    "via": "mcp_fallback",
                    "status_code": response.status_code,
                    "raw": response.text
                }
        except requests.exceptions.RequestException as e:
            # Graceful fallback indicator
            return {
                "tool": tool_name,
                "args": args,
                "via": "mcp_client",
                "ok": True,
                "server_reachable": False,
                "notice": f"MCP server at {self.server_url} is not currently running. Mock acknowledgement returned."
            }
