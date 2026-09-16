"""
LLM Agent Client & Multi-Provider Tool Router
=============================================
Provides intelligent multi-provider tool calling and routing across:
1. Google Gemini API (gemini-2.0-flash, gemini-1.5-flash) via OpenAI-compatible endpoint (Free Tier)
2. Mistral API (mistral-small-latest) via mistralai SDK
3. Local Rule-Based Deterministic Fallback Matcher (100% Offline Resilience)
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple, Optional
import requests


def load_env_file(filepath: str = ".env") -> None:
    """Loads environment variables from .env file if not already present."""
    candidates = [
        filepath,
        os.path.join("..", filepath),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
    ]
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


class MockMessage:
    """Mock message wrapper representing LLM response content."""

    def __init__(self, content: str = ""):
        self.content = content

    def __str__(self):
        return self.content


class MockFunction:
    """Mock function container for tool name and arguments."""

    def __init__(self, name: str, arguments: Any):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    """Mock tool call container matching standard function calling interface."""

    def __init__(self, name: str, arguments: Any):
        self.function = MockFunction(name, arguments)


def fallback_intent_matcher(user_text: str) -> Optional[List[MockToolCall]]:
    """
    Deterministic regex-based intent routing when external LLM APIs are offline or unconfigured.
    """
    text_lower = user_text.lower()

    # 1. Format notes intent
    if any(
        k in text_lower
        for k in ["prepare a note", "format note", "take notes", "meeting notes:"]
    ) or (len(user_text.splitlines()) > 1 and not text_lower.startswith("create a note on")):
        clean_text = re.sub(
            r"^(prepare a note for me\s*|format\s*(this|the)?\s*notes?:?\s*)",
            "",
            user_text,
            flags=re.IGNORECASE,
        ).strip()
        return [MockToolCall("formatNotes", {"rawText": clean_text})]

    # 2. Create note intent
    if "create a note" in text_lower or "create note" in text_lower:
        opp_id_match = re.search(r"006[a-zA-Z0-9]{12,15}", user_text)
        opp_id = opp_id_match.group(0) if opp_id_match else "006gL00000A4GZtQAN"
        body_match = re.search(
            r"saying:?\s*(.*)", user_text, flags=re.IGNORECASE | re.DOTALL
        )
        note_body = body_match.group(1).strip() if body_match else user_text
        return [
            MockToolCall(
                "createNotes",
                {
                    "opportunityId": opp_id,
                    "newNoteToAdd": note_body,
                    "title": "Meeting Notes",
                },
            )
        ]

    # 3. Get latest notes
    if (
        "latest notes" in text_lower
        or "show notes" in text_lower
        or "get notes" in text_lower
    ):
        opp_id_match = re.search(r"006[a-zA-Z0-9]{12,15}", user_text)
        opp_id = opp_id_match.group(0) if opp_id_match else ""
        return [MockToolCall("getLatestNotes", {"opportunityId": opp_id, "limit": 3})]

    # 4. Get opportunity stage / details
    if "stage of opportunity" in text_lower or "opportunity stage" in text_lower:
        opp_id_match = re.search(r"006[a-zA-Z0-9]{12,15}", user_text)
        opp_id = opp_id_match.group(0) if opp_id_match else ""
        return [MockToolCall("getOpportunity", {"opportunityId": opp_id})]

    # 5. Search opportunity by name
    if "opportunity" in text_lower and any(
        w in text_lower for w in ["find", "search", "named", "called", "lookup"]
    ):
        name_match = re.search(
            r"(?:called|named|opportunity)\s+([A-Za-z0-9\s\-]+?)(?:\.|$|\?|in salesforce)",
            user_text,
            flags=re.IGNORECASE,
        )
        opp_name = name_match.group(1).strip() if name_match else user_text
        return [MockToolCall("searchOpportunities", {"opportunityName": opp_name})]

    # 6. Search accounts
    if "find account" in text_lower or "search account" in text_lower:
        name_match = re.search(
            r"(?:called|named|account)\s+([A-Za-z0-9\s\-]+?)(?:\.|$|\?)",
            user_text,
            flags=re.IGNORECASE,
        )
        acc_name = name_match.group(1).strip() if name_match else user_text
        return [MockToolCall("searchAccounts", {"accountName": acc_name})]

    return None


def call_gemini_api(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    api_key: str,
) -> Optional[Tuple[Any, Optional[List[Any]]]]:
    """
    Calls Google Gemini API using the OpenAI-compatible REST endpoint with function calling.
    Cascades from gemini-2.0-flash to gemini-1.5-flash on the Google AI Studio Free Tier.
    """
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Verified Google AI Studio Models in Fallback Priority Order
    # Each model has its own separate quota bucket on the free tier.
    models_to_try = [
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
    ]

    # Prepare system prompt to ensure consistent CRM tool selection
    system_prompt = (
        "You are an expert AI Salesforce CRM Assistant. You have access to tools for Salesforce CRM, SOQL queries, and note formatting. "
        "When the user asks for counts, aggregates, lists, rankings, custom filtering, or any general Salesforce data query, call executeSOQL. "
        "When the user asks to search opportunities or accounts, get opportunity details or stage, create notes, "
        "fetch latest notes, or prepare/format meeting notes and transcripts, call the corresponding function tool "
        "(executeSOQL, searchOpportunities, getOpportunity, createNotes, getLatestNotes, searchAccounts, formatNotes)."
    )

    formatted_messages = []
    if not any(m.get("role") == "system" for m in messages):
        formatted_messages.append({"role": "system", "content": system_prompt})
    formatted_messages.extend(messages)

    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "messages": formatted_messages,
            "tools": tools,
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=20)
            if r.status_code == 200:
                data = r.json()
                choice = data["choices"][0]["message"]
                content = choice.get("content", "")
                raw_tool_calls = choice.get("tool_calls", [])

                tool_calls = []
                for tc in raw_tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name")
                    args = fn.get("arguments", {})
                    tool_calls.append(MockToolCall(name, args))

                return MockMessage(content), (tool_calls if tool_calls else None)
        except Exception:
            continue

    return None


def call_mistral_api(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    api_key: str,
) -> Optional[Tuple[Any, Optional[List[Any]]]]:
    """Calls Mistral API using mistralai SDK with Function Calling."""
    try:
        from mistralai.client import Mistral

        client = Mistral(api_key=api_key)
        resp = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        choice = resp.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None)
        return choice, tool_calls
    except Exception:
        return None


def chat_with_tools(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
) -> Tuple[Any, Optional[List[Any]]]:
    """
    Multi-Provider LLM Agent Router:
    1. Tries Google Gemini API (if GEMINI_API_KEY is configured).
    2. Tries Mistral API (if MISTRAL_API_KEY is configured).
    3. Falls back to deterministic intent matcher if APIs are unreachable.
    """
    load_env_file()

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        result = call_gemini_api(messages, tools, gemini_key)
        if result is not None:
            return result

    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if mistral_key:
        result = call_mistral_api(messages, tools, mistral_key)
        if result is not None:
            return result

    # Fallback to local deterministic intent routing
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    tool_calls = fallback_intent_matcher(last_user_msg)
    if tool_calls:
        return MockMessage("Executing requested tool..."), tool_calls

    return MockMessage(f"Received query: '{last_user_msg}'. No specific tool matched."), None
