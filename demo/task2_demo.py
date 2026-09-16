import os
import sys

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure backend package can be resolved
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.orchestrator import run_query

if __name__ == "__main__":
    print("==================================================")
    print("[DEMO] Salesforce AI & LoRA Orchestrator Demo Suite")
    print("==================================================")

    queries = [
        "Find the opportunity called Dickenson Mobile Generators.",
        "What is the stage of opportunity 006gL00000A4GZtQAN?",
        "Create a note on 006gL00000A4GZtQAN saying: Discussed new features for the Acme Corp Opportunity.",
        "prepare a note for me RetailAxis - Merchandising Lead Sarah Gold: Struggling with inventory sync. Wants cloud migration plan. Will send existing inventory reports."
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n[{i}] USER QUERY: {q}")
        print(">>> EXECUTING ORCHESTRATOR PIPELINE...")
        out = run_query(q)
        print(">>> SYSTEM RESPONSE:")
        print(out)
        print("-" * 60)
