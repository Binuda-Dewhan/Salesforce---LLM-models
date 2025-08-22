import os, json
from dotenv import load_dotenv
load_dotenv()
USE_MCP = os.getenv("USE_MCP", "false").lower() == "true"

class MCPClient:
    def __init__(self):
        # TODO: wire to real MCP server (see refs in the brief)
        pass

    def call(self, tool_name: str, args: dict):
        return {"tool": tool_name, "args": args, "via": "mcp", "ok": True}
