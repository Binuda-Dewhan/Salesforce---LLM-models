"""
backend/mistral_client.py
=========================
Backward-compatibility shim — redirects imports to backend.llm_client.

This module exists solely to avoid breaking any external scripts or
notebooks that may import directly from `backend.mistral_client`.
All functionality now lives in `backend.llm_client`.

NOTE: Do not add new logic here. Import from `backend.llm_client` directly.
"""

from .llm_client import (  # noqa: F401  (re-exported for backward compatibility)
    MockFunction,
    MockMessage,
    MockToolCall,
    call_gemini_api,
    call_mistral_api,
    chat_with_tools,
    fallback_intent_matcher,
)

__all__ = [
    "chat_with_tools",
    "call_gemini_api",
    "call_mistral_api",
    "fallback_intent_matcher",
    "MockMessage",
    "MockFunction",
    "MockToolCall",
]
