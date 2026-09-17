"""
Salesforce AI Integration Backend Package
==========================================
Public API surface for the backend package.
"""

from .config import cfg
from .orchestrator import format_notes_with_lora, run_query
from .salesforce_client import SalesforceClient
from .llm_client import chat_with_tools
from .mcp_client import MCPClient

__all__ = [
    "cfg",
    "run_query",
    "format_notes_with_lora",
    "SalesforceClient",
    "chat_with_tools",
    "MCPClient",
]
