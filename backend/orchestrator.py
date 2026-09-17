"""
backend/orchestrator.py
=======================
ReAct Agent Orchestrator — the core intelligence loop for every user query.

Flow:
  1. Evaluate user prompt via the configured LLM (guardrails + tool planning).
  2. Execute all planned tool calls (Salesforce REST, MCP, or local LoRA model).
  3. Synthesize a grounded executive answer from the tool observations.
  4. Return a unified payload containing the answer and full audit trace.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import cfg
from .llm_client import chat_with_tools, synthesize_grounded_answer
from .mcp_client import MCPClient
from .salesforce_client import SalesforceClient
from .tools import TOOLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA Model Manager — Singleton, lazy-loaded on first use
# ---------------------------------------------------------------------------

class LoRAModelManager:
    """
    Manages lazy-loading and inference for the fine-tuned Salesforce LoRA adapter.

    The model is loaded exactly once on first call to `format_notes()` and held
    in memory for the lifetime of the process.
    """

    _instance: Optional["LoRAModelManager"] = None
    _model: Any = None
    _tokenizer: Any = None

    _SYSTEM_PROMPT = (
        "You are an expert Salesforce CRM Assistant. Your task is to take raw, messy meeting "
        "notes and format them strictly according to the company's best practices.\n"
        "You must extract and organize the information into the following sections exactly:\n"
        "Summary:\n"
        "Key Pain Points:\n"
        "Action Items:\n"
        "Next Steps:\n"
        "Date/Time of Interaction:\n"
    )

    _BASE_MODEL_ID = "google/gemma-2b-it"

    @classmethod
    def get_instance(cls) -> "LoRAModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _resolve_adapter_path(self) -> Optional[Path]:
        """
        Resolves the LoRA adapter directory using a deterministic lookup strategy:
          1. LORA_ADAPTER_PATH env var (explicit override — highest priority)
          2. Known paths relative to this file's location (absolute, reliable)
        """
        # 1. Env var override
        env_override = cfg.lora_adapter_path()
        if env_override:
            p = Path(env_override)
            if p.exists() and (p / "adapter_config.json").exists():
                return p
            logger.warning("LORA_ADAPTER_PATH set to %s but adapter_config.json not found.", p)

        # 2. Deterministic absolute paths relative to this file
        repo_root = Path(__file__).resolve().parent.parent
        candidates = [
            repo_root / "Notebook" / "salesforce_notes_adapter",
            repo_root / "salesforce_notes_adapter",
        ]
        for candidate in candidates:
            if candidate.exists() and (
                (candidate / "adapter_model.safetensors").exists()
                or (candidate / "adapter_config.json").exists()
            ):
                return candidate

        return None

    def _load_model(self) -> None:
        """Lazy-loads the base model and attaches the LoRA adapter if found."""
        if self._model is not None:
            return  # Already loaded

        adapter_path = self._resolve_adapter_path()

        tokenizer_source = str(adapter_path) if adapter_path else self._BASE_MODEL_ID
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self._BASE_MODEL_ID,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if adapter_path:
            self._model = PeftModel.from_pretrained(base_model, str(adapter_path))
            logger.info("Loaded fine-tuned LoRA adapter from: %s", adapter_path)
        else:
            self._model = base_model
            logger.warning(
                "LoRA adapter directory not found. Running with base model '%s'. "
                "Set LORA_ADAPTER_PATH in your .env to point to the adapter directory.",
                self._BASE_MODEL_ID,
            )

        self._model.eval()
        self._model.config.use_cache = True

    def format_notes(self, raw_text: str) -> str:
        """Formats unstructured meeting notes into the standard 5-section enterprise CRM schema."""
        self._load_model()

        prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{self._SYSTEM_PROMPT}\n\n{raw_text}"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=[
                    self._tokenizer.eos_token_id,
                    self._tokenizer.convert_tokens_to_ids("<end_of_turn>"),
                ],
                pad_token_id=self._tokenizer.eos_token_id,
            )

        input_length = inputs.input_ids.shape[1]
        response = self._tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        ).strip()

        # Clean repetitive placeholder loops that can occur at context tail
        response = re.sub(
            r"(\n?•\s*\[Insert current date/time\][^\n]*)+",
            "\n• [Insert current date/time]",
            response,
        )
        return response


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_model_manager = LoRAModelManager.get_instance()
_sf = SalesforceClient()
_mcp = MCPClient()


def format_notes_with_lora(raw_text: str) -> str:
    """Public helper: formats raw meeting notes using the LoRA model."""
    return _model_manager.format_notes(raw_text)


# ---------------------------------------------------------------------------
# Local Tool Dispatch Map
# ---------------------------------------------------------------------------

LOCAL_TOOL_MAP = {
    "searchOpportunities": lambda a: _sf.search_opportunities(a["opportunityName"]),
    "getOpportunity":      lambda a: _sf.get_opportunity(a["opportunityId"]),
    "searchAccounts":      lambda a: _sf.search_accounts(a["accountName"]),
    "createNotes":         lambda a: _sf.create_note(
        a["opportunityId"], a["newNoteToAdd"], a.get("title", "Meeting Notes")
    ),
    "getLatestNotes":      lambda a: _sf.get_latest_notes(a["opportunityId"], a.get("limit", 3)),
    "formatNotes":         lambda a: format_notes_with_lora(a["rawText"]),
    "executeSOQL":         lambda a: _sf.execute_soql(a["query"]),
}


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def _parse_args(raw_args: Any) -> Dict[str, Any]:
    """Ensures tool arguments are always returned as a Python dictionary."""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse tool arguments as JSON: %s | raw=%r", exc, raw_args)
            return {"_raw": raw_args}
    return {"_raw": str(raw_args)}


# ---------------------------------------------------------------------------
# Main ReAct Agent Entry Point
# ---------------------------------------------------------------------------

def run_query(user_prompt: str) -> Dict[str, Any]:
    """
    Main entry point for conversational agent queries (Grounded ReAct Loop).

    Steps:
      1. Evaluates the user prompt against CRM Executive system prompt + domain guardrails.
      2. Identifies required tool(s) — supports parallel / multi-tool execution.
      3. Executes each tool against Salesforce REST, MCP microservice, or local LoRA model.
      4. Gathers all observations and runs a grounded answer synthesis pass.
      5. Returns a unified payload with the executive answer and an audit trace.

    Returns:
        {
            "type": "agent_response",
            "answer": str,
            "is_refusal": bool,
            "tool_executions": List[Dict]
        }
    """
    messages = [{"role": "user", "content": user_prompt}]
    model_msg, tool_calls = chat_with_tools(messages, TOOLS)

    # --- Stage 1: No tool calls → direct conversational/guardrail response ---
    if not tool_calls:
        content = getattr(model_msg, "content", str(model_msg)).strip()
        is_refusal = any(
            phrase in content.lower()
            for phrase in [
                "specialized exclusively",
                "cannot assist with general",
                "salesforce crm executive assistant",
                "off-topic inquiries",
                "unrelated to salesforce",
            ]
        )
        return {
            "type": "agent_response",
            "answer": content,
            "is_refusal": is_refusal,
            "tool_executions": [],
        }

    # --- Stage 2: Execute all planned tool calls ---
    use_mcp = cfg.use_mcp()
    tool_executions: List[Dict[str, Any]] = []

    for tc in tool_calls:
        tool_name = getattr(tc.function, "name", None)
        raw_args = getattr(tc.function, "arguments", {})
        args = _parse_args(raw_args)

        logger.info("Executing tool: %s | args: %s", tool_name, args)

        try:
            if tool_name in LOCAL_TOOL_MAP:
                result = LOCAL_TOOL_MAP[tool_name](args)
                # Optionally log to MCP for audit/observability (result still comes from local)
                if use_mcp:
                    try:
                        _mcp.call(tool_name, args)
                    except Exception as mcp_exc:  # noqa: BLE001
                        logger.debug("MCP audit call for '%s' failed (non-critical): %s", tool_name, mcp_exc)
            elif use_mcp:
                result = _mcp.call(tool_name, args)
            else:
                result = {"error": f"Unknown tool '{tool_name}' and MCP is disabled."}
        except Exception as exc:  # noqa: BLE001
            logger.error("Tool '%s' execution failed: %s", tool_name, exc, exc_info=True)
            result = {"error": str(exc)}

        tool_executions.append({"tool": tool_name, "args": args, "result": result})

    # --- Stage 3: Grounded Answer Synthesis ---
    try:
        synthesized_answer = synthesize_grounded_answer(user_prompt, tool_executions)
    except Exception as exc:  # noqa: BLE001
        logger.error("Answer synthesis failed: %s", exc, exc_info=True)
        synthesized_answer = (
            f"Executed {len(tool_executions)} Salesforce operation(s) successfully. "
            "Please review the execution details below."
        )

    return {
        "type": "agent_response",
        "answer": synthesized_answer,
        "is_refusal": False,
        "tool_executions": tool_executions,
    }
