import os
import sys

# Ensure backend package can be imported if script is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.salesforce_client import SalesforceClient
from backend.orchestrator import format_notes_with_lora

def main():
    print("=== Salesforce Batch Note Ingestion Engine ===\n")
    
    # Initialize Salesforce client with dynamic .env authentication
    sf_client = SalesforceClient()

    # Raw notes to format and sync for each opportunity
    notes = [
        {
            "opportunity_id": "006gL00000A4GZtQAN",  # Acme Corp
            "opportunity_name": "Acme Corp",
            "raw_note": """Explored current infrastructure; using legacy CRM.
Interested in reducing manual data entry by 40%.
Identified budget stage and decision timeline (Q4)."""
        },
        {
            "opportunity_id": "006gL00000A4GZuQAN",  # United Oil Office Portable Generators
            "opportunity_name": "United Oil Office Portable Generators",
            "raw_note": """Has 5 vendors; wants vendor comparison.
Needs scalability to 5,000 users.
Agreed to send basic requirements doc."""
        },
        {
            "opportunity_id": "006gL00000A4GZvQAN",  # Express Logistics Standby Generator
            "opportunity_name": "Express Logistics Standby Generator",
            "raw_note": """Seeking AI-driven analytics.
IT already has BI tools—curious about overlay.
Agreed to a follow-up demo with BI team."""
        }
    ]

    for item in notes:
        opp_id = item["opportunity_id"]
        opp_name = item["opportunity_name"]
        raw = item["raw_note"]

        print(f"\n--- Processing Opportunity: {opp_name} ({opp_id}) ---")
        print("Raw Note:\n", raw)

        # 1. Format raw note with our fine-tuned LoRA model
        print("Formatting note via Local Gemma-2B LoRA Adapter...")
        formatted_note = format_notes_with_lora(raw)
        print("Formatted Note:\n", formatted_note)

        # 2. Sync note to Salesforce
        try:
            print("Syncing note to Salesforce Cloud...")
            response = sf_client.create_note(opp_id, formatted_note, title=f"Meeting Notes - {opp_name}")
            print(f"✓ Note successfully created in Salesforce: {response}")
        except Exception as e:
            print(f"✗ Failed to create note in Salesforce: {e}")

if __name__ == "__main__":
    main()
