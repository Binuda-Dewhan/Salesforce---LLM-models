import os
import re
import json
import torch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from .tools import TOOLS
from .llm_client import chat_with_tools
from .salesforce_client import SalesforceClient
from .mcp_client import MCPClient, USE_MCP

# Pure-Python env loader
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

# -------------------------------------------------------------
# Singleton Fine-Tuned LoRA Model Manager
# -------------------------------------------------------------
class LoRAModelManager:
    """Manages lazy-loading and inference for our fine-tuned Salesforce LoRA adapter."""
    _instance = None
    _model = None
    _tokenizer = None

    SYSTEM_PROMPT = """You are an expert Salesforce CRM Assistant. Your task is to take raw, messy meeting notes and format them strictly according to the company's best practices.
You must extract and organize the information into the following sections exactly:
Summary:
Key Pain Points:
Action Items:
Next Steps:
Date/Time of Interaction:
"""

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        if self._model is not None:
            return

        # Find the adapter path
        candidates = [
            "Notebook/salesforce_notes_adapter",
            "salesforce_notes_adapter",
            os.path.join(os.path.dirname(__file__), "..", "Notebook", "salesforce_notes_adapter"),
            os.path.join(os.path.dirname(__file__), "..", "salesforce_notes_adapter")
        ]
        adapter_path = None
        for path in candidates:
            if os.path.exists(path) and (
                os.path.exists(os.path.join(path, "adapter_model.safetensors")) or
                os.path.exists(os.path.join(path, "adapter_config.json"))
            ):
                adapter_path = path
                break

        base_model_id = "google/gemma-2b-it"

        # 4-bit Quantization Config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # Load Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            adapter_path if adapter_path else base_model_id
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Attach LoRA Adapter
        if adapter_path:
            self._model = PeftModel.from_pretrained(base_model, adapter_path)
            print(f"[OK] Loaded fine-tuned LoRA adapter from: {adapter_path}")
        else:
            self._model = base_model
            print("Notice: LoRA adapter directory not found. Running with base model.")

        self._model.eval()
        self._model.config.use_cache = True

    def format_notes(self, raw_text: str) -> str:
        """Formats unstructured notes into standard 5-section enterprise CRM schema."""
        self._load_model()
        
        prompt = self._tokenizer.apply_chat_template([
            {"role": "user", "content": f"{self.SYSTEM_PROMPT}\n\n{raw_text}"}
        ], tokenize=False, add_generation_prompt=True)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=[self._tokenizer.eos_token_id, self._tokenizer.convert_tokens_to_ids("<end_of_turn>")],
                pad_token_id=self._tokenizer.eos_token_id
            )

        input_length = inputs.input_ids.shape[1]
        response = self._tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        # Clean repetitive placeholder loops if model reaches tail of context
        response = re.sub(r'(\n?•\s*\[Insert current date/time\][^\n]*)+', '\n• [Insert current date/time]', response)
        return response


# Global Instances
model_manager = LoRAModelManager.get_instance()
sf = SalesforceClient()
mcp = MCPClient()


def format_notes_with_lora(raw_text: str) -> str:
    """Helper function exposed to tools and orchestrator."""
    return model_manager.format_notes(raw_text)


LOCAL_TOOL_MAP = {
    "searchOpportunities": lambda a: sf.search_opportunities(a["opportunityName"]),
    "getOpportunity":      lambda a: sf.get_opportunity(a["opportunityId"]),
    "searchAccounts":      lambda a: sf.search_accounts(a["accountName"]),
    "createNotes":         lambda a: sf.create_note(
        a["opportunityId"], a["newNoteToAdd"], a.get("title", "Meeting Notes")
    ),
    "getLatestNotes":      lambda a: sf.get_latest_notes(a["opportunityId"], a.get("limit", 3)),
    "formatNotes":         lambda a: format_notes_with_lora(a["rawText"]),
    "executeSOQL":         lambda a: sf.execute_soql(a["query"]),
}


def parse_args(raw_args) -> Dict[str, Any]:
    """Ensures tool arguments are parsed reliably as a Python dictionary."""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except Exception:
            return {"_raw": raw_args}
    return {"_raw": str(raw_args)}


def run_query(user_prompt: str) -> Dict[str, Any]:
    """
    Main entry point for conversational agent queries:
    1. Sends prompt to LLM planner (Mistral or fallback).
    2. Identifies required tool.
    3. Executes tool (Salesforce REST, MCP, or Local LoRA Formatter).
    4. Returns structured execution payload.
    """
    messages = [{"role": "user", "content": user_prompt}]
    model_msg, tool_calls = chat_with_tools(messages, TOOLS)

    if not tool_calls:
        return {"type": "answer", "message": getattr(model_msg, "content", str(model_msg))}

    tc = tool_calls[0]
    tool_name = getattr(tc.function, "name", None)
    raw_args = getattr(tc.function, "arguments", {})
    args = parse_args(raw_args)

    if tool_name in LOCAL_TOOL_MAP:
        try:
            # Check if MCP routing is active for getLatestNotes
            if USE_MCP and tool_name == "getLatestNotes":
                mcp.call(tool_name, args)

            result = LOCAL_TOOL_MAP[tool_name](args)
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
        "error": f"Unknown tool '{tool_name}' and MCP is disabled.",
        "args": args
    }
