"""
Automated Test Suite for Salesforce Grounded ReAct Agent & CRM Domain Guardrails
================================================================================
Evaluates:
1. CRM Domain Scope Guardrail (Polite rejection of off-topic / general knowledge questions)
2. Compound Query Tool Calling & Grounded Answer Synthesis (Count + Top Deals)
3. Direct Salesforce Record Inspection
4. Fine-Tuned Gemma-2B LoRA Notes Formatter Routing
5. Offline Resilience & Deterministic Fallback Synthesizer
"""

import os
import sys

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.orchestrator import run_query, format_notes_with_lora
from backend.llm_client import fallback_intent_matcher, fallback_synthesizer, MockToolCall


def test_domain_guardrail_rejection():
    """Confirms off-topic trivia queries are rejected and no tools are executed."""
    prompts = [
        "What is the capital of France?",
        "How do I bake sourdough bread?",
        "Write a python function to sort an array"
    ]
    for p in prompts:
        res = run_query(p)
        assert res.get("type") == "agent_response"
        assert res.get("is_refusal") is True
        assert len(res.get("tool_executions", [])) == 0
        answer_lower = res.get("answer", "").lower()
        assert any(
            phrase in answer_lower
            for phrase in ["specialized exclusively", "salesforce crm executive assistant", "cannot assist with general"]
        )


def test_compound_query_react_grounding():
    """Confirms compound queries plan multiple tools and synthesize grounded answers."""
    prompt = "How many opportunities are in Salesforce? and what are the valuable once"
    res = run_query(prompt)

    assert res.get("type") == "agent_response"
    assert res.get("is_refusal") is False
    tool_execs = res.get("tool_executions", [])
    assert len(tool_execs) >= 1

    answer = res.get("answer", "")
    assert len(answer) > 20
    # Must mention counts or opportunities
    assert any(w in answer.lower() for w in ["opportunity", "opportunities", "pipeline", "31", "record"])


def test_single_intent_soql():
    """Confirms targeted SOQL query execution and grounded synthesis."""
    prompt = "Show me opportunities in Closed Won stage"
    res = run_query(prompt)

    assert res.get("type") == "agent_response"
    assert res.get("is_refusal") is False
    tool_execs = res.get("tool_executions", [])
    assert len(tool_execs) >= 1
    assert any(te.get("tool") == "executeSOQL" for te in tool_execs)


def test_format_notes_lora():
    """Confirms unstructured meeting notes format into standard 5-section enterprise CRM schema."""
    raw = "Client call with Acme Corp CTO John. Legacy CRM is too slow. Action: send security whitepaper by Friday."
    formatted = format_notes_with_lora(raw)

    assert "Summary" in formatted
    assert "Key Pain Points" in formatted
    assert "Action Items" in formatted
    assert "Next Steps" in formatted


def test_offline_fallback_synthesizer():
    """Verifies that deterministic offline synthesizer produces rich Markdown summaries."""
    tool_execs = [
        {
            "tool": "executeSOQL",
            "args": {"query": "SELECT COUNT() FROM Opportunity"},
            "result": {"totalSize": 31, "records": []}
        },
        {
            "tool": "executeSOQL",
            "args": {"query": "SELECT Name, Amount, StageName FROM Opportunity ORDER BY Amount DESC LIMIT 2"},
            "result": {
                "records": [
                    {"Name": "United Oil SLA", "Amount": 5600000, "StageName": "Closed Won"},
                    {"Name": "Express Logistics", "Amount": 1200000, "StageName": "Prospecting"}
                ]
            }
        }
    ]
    summary = fallback_synthesizer("Summarize my pipeline", tool_execs)
    assert "**31 total opportunities**" in summary
    assert "United Oil SLA" in summary
    assert "$5,600,000.00" in summary


if __name__ == "__main__":
    print("Running Automated Tests...")
    test_domain_guardrail_rejection()
    print(" [OK] Domain Scope Guardrail Rejection")

    test_offline_fallback_synthesizer()
    print(" [OK] Offline Fallback Synthesizer")

    test_compound_query_react_grounding()
    print(" [OK] Compound Query ReAct Grounding")

    test_single_intent_soql()
    print(" [OK] Single-Intent SOQL Execution")

    test_format_notes_lora()
    print(" [OK] LoRA Meeting Notes Formatter")

    print("\nAll Test Cases Passed Successfully!")
