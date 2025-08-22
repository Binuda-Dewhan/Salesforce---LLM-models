import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

from .tools import TOOLS
from .mistral_client import chat_with_tools
from .salesforce_client import SalesforceClient
from .mcp_client import MCPClient, USE_MCP
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

BASE_MODEL = "./lora_salesforce_notes"   # 👈 use your saved folder, not HF repo

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

pipe_lora = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")


def format_notes_with_lora(raw_text: str) -> str:
    prompt = f"Meeting Notes:\n{raw_text}\n\nFormatted Notes:\n"
    output = pipe_lora(prompt, max_new_tokens=200, eos_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    return output

load_dotenv()

sf = SalesforceClient()
mcp = MCPClient()

LOCAL_TOOL_MAP = {
    "searchOpportunities": lambda a: sf.search_opportunities(a["opportunityName"]),
    "getOpportunity":      lambda a: sf.get_opportunity(a["opportunityId"]),
    "searchAccounts":      lambda a: sf.search_accounts(a["accountName"]),
    "createNotes":         lambda a: sf.create_note(
        a["opportunityId"], a["newNoteToAdd"], a.get("title", "Meeting Notes")
    ),
    "getLatestNotes":      lambda a: sf.get_latest_notes(a["opportunityId"], a.get("limit", 3)),
    "formatNotes":         lambda a: format_notes_with_lora(a["rawText"]),
}


def parse_args(raw_args) -> Dict[str, Any]:
    """
    Ensure tool arguments are always a dict.
    Mistral sometimes returns JSON string instead of dict.
    """
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except Exception:
            return {"_raw": raw_args}  # fallback for debugging
    return {"_raw": str(raw_args)}


def run_query(user_prompt: str) -> Dict[str, Any]:
    messages = [{"role": "user", "content": user_prompt}]
    model_msg, tool_calls = chat_with_tools(messages, TOOLS)

    if not tool_calls:
        # No tool call, return model's direct answer
        return {"type": "answer", "message": model_msg.content}

    # Only handle the first tool call for Task 2 demo
    tc = tool_calls[0]
    tool_name = getattr(tc.function, "name", None)
    raw_args = getattr(tc.function, "arguments", {})

    args = parse_args(raw_args)

    if tool_name in LOCAL_TOOL_MAP:
        try:
            # Special case: for Task 2, demonstrate MCP routing for "getLatestNotes"
            if USE_MCP and tool_name == "getLatestNotes":
                mcp_ack = mcp.call(tool_name, args)  # bridge call
                # You can log or include mcp_ack in the response if required

            result = LOCAL_TOOL_MAP[tool_name](args)  # real Salesforce call
            return {
                "type": "tool_result",
                "tool": tool_name,
                "args": args,
                "result": result
            }
        except Exception as e:
            return {
                "type": "error",
                "tool": tool_name,
                "args": args,
                "error": str(e)
            }

    if USE_MCP:
        try:
            res = mcp.call(tool_name, args)
            return {
                "type": "tool_result_mcp",
                "tool": tool_name,
                "args": args,
                "result": res
            }
        except Exception as e:
            return {
                "type": "error",
                "tool": tool_name,
                "args": args,
                "error": f"MCP error: {str(e)}"
            }

    return {
        "type": "error",
        "error": f"Unknown tool '{tool_name}' and MCP disabled.",
        "args": args
    }
