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


CRM_AGENT_SYSTEM_PROMPT = (
    "You are an enterprise Salesforce CRM Executive Assistant.\n"
    "Your role is exclusively to help users query, analyze, inspect, and update their Salesforce CRM data "
    "(Opportunities, Accounts, Contacts, Notes), and to format meeting notes using the fine-tuned LoRA model.\n\n"
    "STRICT DOMAIN BOUNDARIES & SCOPE GUARDRAILS:\n"
    "1. You operate EXCLUSIVELY within the domain of Salesforce CRM, sales pipelines, accounts, opportunities, CRM notes, and sales operations.\n"
    "2. If the user asks ANY general knowledge, trivia, coding riddles, recipes, history, geography, creative writing, or non-CRM questions "
    "(e.g., 'What is the capital of France?', 'How do I bake bread?', 'Tell me a joke', 'Who won the World Cup?', 'Write a Python script'):\n"
    "   - You MUST POLITELY DECLINE the request.\n"
    "   - Formulate your response as: 'I am specialized exclusively as your Salesforce CRM Executive Assistant. I cannot assist with general knowledge or off-topic inquiries, but I am ready to help you analyze your Salesforce pipeline, query opportunities, inspect accounts, or format meeting notes.'\n"
    "   - Do NOT answer the off-topic question. Do NOT invoke any tools.\n\n"
    "TOOL CALLING INSTRUCTIONS:\n"
    "1. For questions requiring Salesforce data (counts, sums, lists, stages, rankings, custom filtering), call `executeSOQL`.\n"
    "2. When the user asks a compound question with multiple data requirements (e.g., 'How many opportunities are in Salesforce? and what are the valuable ones?'), "
    "   you MUST generate ALL necessary tool calls (e.g. one executeSOQL for COUNT() and one executeSOQL for the top opportunities by Amount).\n"
    "3. For formatting unstructured meeting notes or call transcripts, call `formatNotes`.\n"
    "4. For creating/attaching notes to opportunities, call `createNotes`.\n"
    "5. For fetching latest notes, call `getLatestNotes`.\n"
    "6. For finding specific accounts or opportunities by name, call `searchAccounts` or `searchOpportunities`."
)

SYNTHESIZER_SYSTEM_PROMPT = (
    "You are an enterprise Salesforce CRM Executive Assistant.\n"
    "Your task is to synthesize a professional, executive-ready, natural language response to the user's inquiry based strictly on the real observations returned by Salesforce tools.\n\n"
    "GROUNDING & SYNTHESIS RULES:\n"
    "1. STRICT TRUTHFULNESS: Only state facts, numbers, names, and metrics that appear in the Tool Execution Observations. Never hallucinate or invent records.\n"
    "2. EXECUTIVE TONE & BUSINESS INSIGHTS: Provide clear business context, highlight key figures (e.g., total counts, highest value deals, deal stages, account names), and explain what the numbers mean.\n"
    "3. CONTEXT-ADAPTIVE FORMATTING:\n"
    "   - Address ALL parts of the user's question (e.g. if they asked both for total count AND top valuable deals, answer both clearly).\n"
    "   - Use bold text for key figures, metrics, and opportunity names.\n"
    "   - Use clean bullet points for readability when listing multiple items.\n"
    "   - Do NOT force ASCII or raw markdown tables unless the user explicitly requested a table.\n"
    "   - Keep it concise, executive, and actionable.\n"
    "4. If a tool returned no records or an error, explain politely and suggest what query or search might work."
)


def call_gemini_api(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    api_key: str,
) -> Optional[Tuple[Any, Optional[List[Any]]]]:
    """
    Calls Google Gemini API using OpenAI-compatible REST endpoint with function calling.
    Cascades across Google AI Studio models with separate quota buckets.
    """
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    models_to_try = [
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
    ]

    formatted_messages = []
    if not any(m.get("role") == "system" for m in messages):
        formatted_messages.append({"role": "system", "content": CRM_AGENT_SYSTEM_PROMPT})
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

        formatted_messages = []
        if not any(m.get("role") == "system" for m in messages):
            formatted_messages.append({"role": "system", "content": CRM_AGENT_SYSTEM_PROMPT})
        formatted_messages.extend(messages)

        client = Mistral(api_key=api_key)
        resp = client.chat.complete(
            model="mistral-small-latest",
            messages=formatted_messages,
            tools=tools,
            tool_choice="auto",
        )
        choice = resp.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None)
        return choice, tool_calls
    except Exception:
        return None


def generate_llm_text(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    Direct LLM text completion (without function calling) for answer synthesis.
    Cascades through Gemini models, then Mistral API.
    """
    load_env_file()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        headers = {
            "Authorization": f"Bearer {gemini_key}",
            "Content-Type": "application/json",
        }
        models_to_try = [
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-2.5-flash",
        ]
        for model_name in models_to_try:
            try:
                r = requests.post(
                    url,
                    headers=headers,
                    json={"model": model_name, "messages": messages},
                    timeout=25,
                )
                if r.status_code == 200:
                    data = r.json()
                    content = data["choices"][0]["message"].get("content", "").strip()
                    if content:
                        return content
            except Exception:
                continue

    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if mistral_key:
        try:
            from mistralai.client import Mistral

            client = Mistral(api_key=mistral_key)
            resp = client.chat.complete(
                model="mistral-small-latest",
                messages=messages,
            )
            content = getattr(resp.choices[0].message, "content", "").strip()
            if content:
                return content
        except Exception:
            pass

    return None


def fallback_synthesizer(user_prompt: str, tool_executions: List[Dict[str, Any]]) -> str:
    """
    Intelligent deterministic synthesizer when cloud LLMs are offline or unreachable.
    Generates clean Markdown summaries from the tool observations.
    """
    parts = []
    has_count = False
    has_records = False

    for item in tool_executions:
        tool = item.get("tool")
        res = item.get("result", {})

        if tool == "executeSOQL":
            if isinstance(res, dict) and "totalSize" in res and (not res.get("records") or len(res["records"]) == 0):
                parts.append(f"There are currently **{res['totalSize']} total opportunities** recorded in your Salesforce CRM.")
                has_count = True
            elif isinstance(res, dict) and res.get("records"):
                records = res["records"]
                has_records = True
                lines = ["Here are the top opportunities from your pipeline:"]
                for i, r in enumerate(records[:5], 1):
                    name = r.get("Name", "Unnamed")
                    stage = r.get("StageName", "N/A")
                    amount = r.get("Amount")
                    amt_str = f"${float(amount):,.2f}" if amount is not None else "Amount not set"
                    lines.append(f"{i}. **{name}** — {amt_str} | Stage: `{stage}`")
                parts.append("\n".join(lines))

        elif tool in ["searchOpportunities", "searchAccounts", "getOpportunity"]:
            records = res.get("records", []) if isinstance(res, dict) else []
            if records:
                lines = [f"Found **{len(records)}** matching record(s) in Salesforce:"]
                for r in records[:5]:
                    name = r.get("Name", "N/A")
                    id_val = r.get("Id", "")
                    stage = r.get("StageName")
                    extra = f" | Stage: `{stage}`" if stage else ""
                    lines.append(f"- **{name}** (`{id_val}`){extra}")
                parts.append("\n".join(lines))
            else:
                parts.append("No matching records found in Salesforce.")

        elif tool == "createNotes":
            args = item.get("args", {})
            parts.append(f"Successfully attached note to Opportunity **{args.get('opportunityId')}** in Salesforce.")

    if parts:
        return "\n\n".join(parts)

    return "Salesforce query executed successfully. See execution details below."


def synthesize_grounded_answer(user_prompt: str, tool_executions: List[Dict[str, Any]]) -> str:
    """
    Synthesizes an executive, contextual, natural-language response strictly grounded
    in the live Salesforce observations.
    """
    if not tool_executions:
        return "No Salesforce actions were required."

    # Direct passthrough for formatNotes LoRA output
    if len(tool_executions) == 1 and tool_executions[0].get("tool") == "formatNotes":
        return str(tool_executions[0].get("result", ""))

    # Prepare structured observation context
    obs_text = ""
    for i, item in enumerate(tool_executions, 1):
        tool = item.get("tool", "unknown")
        args = item.get("args", {})
        res = item.get("result", {})
        obs_text += f"\n--- Observation {i} ({tool}) ---\nParameters: {json.dumps(args)}\nResult: {json.dumps(res, default=str)}\n"

    synthesis_messages = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User Inquiry: {user_prompt}\n\n"
                f"Salesforce Real Observations:\n{obs_text}\n\n"
                f"Now synthesize an executive, grounded, helpful response strictly based on the above observations:"
            ),
        },
    ]

    response = generate_llm_text(synthesis_messages)
    if response and response.strip():
        return response.strip()

    # Resilient fallback synthesis
    return fallback_synthesizer(user_prompt, tool_executions)


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

    # Offline domain guardrail check
    off_topic_indicators = [
        "capital of", "recipe", "who is the president", "tell me a joke",
        "write a poem", "how to bake", "sort an array", "weather in"
    ]
    if any(k in last_user_msg.lower() for k in off_topic_indicators):
        return (
            MockMessage(
                "I am specialized exclusively as your Salesforce CRM Executive Assistant. "
                "I cannot assist with general knowledge or off-topic inquiries, but I am ready "
                "to help you analyze your Salesforce pipeline, query opportunities, inspect accounts, "
                "or format meeting notes."
            ),
            None,
        )

    tool_calls = fallback_intent_matcher(last_user_msg)
    if tool_calls:
        return MockMessage("Executing requested tool..."), tool_calls

    return (
        MockMessage(
            f"I am your Salesforce CRM Executive Assistant. How can I assist you with your pipeline, "
            f"opportunities, accounts, or meeting notes today?"
        ),
        None,
    )

