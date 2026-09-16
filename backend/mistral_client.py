"""
Legacy module alias for backward-compatibility.
Redirects directly to backend.llm_client.
"""

from .llm_client import (
    chat_with_tools,
    call_gemini_api,
    call_mistral_api,
    fallback_intent_matcher,
    MockMessage,
    MockFunction,
    MockToolCall,
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
