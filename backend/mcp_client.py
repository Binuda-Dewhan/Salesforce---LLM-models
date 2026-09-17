"""
backend/mcp_client.py
=====================
Model Context Protocol (MCP) Client.
Sends JSON-RPC 2.0 tool calls to a local or remote MCP Server.
Falls back gracefully with a descriptive notice if the server is offline.

Environment is loaded centrally by backend.config — this module does NOT call
load_env_file() itself.
"""

import logging
from typing import Any, Dict

import requests

from .config import cfg

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Invokes MCP tools via JSON-RPC 2.0 over HTTP.

    Falls back gracefully if the MCP server is offline — callers receive a
    structured notice dict instead of an exception, so the agent loop continues.
    """

    def __init__(self, server_url: str | None = None) -> None:
        self.server_url = (server_url or cfg.mcp_server_url()).rstrip("/")

    def call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invokes an MCP tool via JSON-RPC 2.0.

        Args:
            tool_name: The registered MCP tool name (e.g. "searchOpportunities").
            args: Tool arguments as a plain dictionary.

        Returns:
            The tool result dict on success, or a fallback notice dict on failure.
        """
        payload = {
            "jsonrpc": "2.0",
            "method": f"tools/{tool_name}",
            "params": args,
            "id": 1,
        }

        try:
            response = requests.post(
                f"{self.server_url}/rpc",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=(3, 10),  # (connect, read)
            )
            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    logger.warning(
                        "MCP server returned JSON-RPC error for tool '%s': %s",
                        tool_name,
                        data["error"],
                    )
                return data.get("result", data)

            logger.warning(
                "MCP server responded with HTTP %s for tool '%s'.",
                response.status_code,
                tool_name,
            )
            return {
                "tool": tool_name,
                "args": args,
                "via": "mcp_fallback",
                "status_code": response.status_code,
                "raw": response.text,
            }

        except requests.exceptions.Timeout:
            logger.warning("MCP server at %s timed out for tool '%s'.", self.server_url, tool_name)
        except requests.exceptions.ConnectionError:
            logger.debug("MCP server at %s is not reachable (tool: '%s').", self.server_url, tool_name)
        except requests.exceptions.RequestException as exc:
            logger.warning("MCP request for tool '%s' failed: %s", tool_name, exc)

        # Graceful fallback — non-fatal, agent loop continues
        return {
            "tool": tool_name,
            "args": args,
            "via": "mcp_client",
            "ok": True,
            "server_reachable": False,
            "notice": (
                f"MCP server at {self.server_url} is not currently running. "
                "Mock acknowledgement returned."
            ),
        }
