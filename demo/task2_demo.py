
from backend.orchestrator import run_query

if __name__ == "__main__":
    # Sample queries to test
    queries = [
        "Find the opportunity called Dickenson Mobile Generators.",
        "What is the stage of opportunity 006gL00000A4GZtQAN?",
        "Create a note on 006gL00000A4GZtQAN saying: Discussed new features for the Acme Corp Opportunity.",
        "prepare a note for me RetailAxis – Merchandising Lead Sarah Gold: Struggling with inventory sync. ○ Wants cloud migration plan. ○ Will send existing inventory reports. "


        
    ]
    for q in queries:
        print("\n---")
        print("USER:", q)
        out = run_query(q)
        print("OUTPUT:", out)
