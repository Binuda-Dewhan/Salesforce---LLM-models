"""
backend/add_notes.py
====================
Standalone batch note ingestion utility script.

PURPOSE:
  This is a one-shot CLI script — NOT part of the API server or agent loop.
  It formats a predefined set of raw meeting notes using the local LoRA model
  and then syncs the formatted notes to Salesforce as structured Note records.

USAGE:
  Run from the repository root:
    python -m backend.add_notes

  Or directly:
    cd salesforce-integration
    python backend/add_notes.py

NOTE: Opportunity IDs in this file are sample/demo records. Update them to
      match your own Salesforce org's Opportunity IDs before running.
"""

import logging
import os
import sys

# Allow running as a standalone script from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.orchestrator import format_notes_with_lora
from backend.salesforce_client import SalesforceClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo notes — update opportunity_id values to match your org
# ---------------------------------------------------------------------------
_DEMO_NOTES = [
    {
        "opportunity_id": "006gL00000A4GZtQAN",  # Example: Acme Corp
        "opportunity_name": "Acme Corp",
        "raw_note": (
            "Explored current infrastructure; using legacy CRM.\n"
            "Interested in reducing manual data entry by 40%.\n"
            "Identified budget stage and decision timeline (Q4)."
        ),
    },
    {
        "opportunity_id": "006gL00000A4GZuQAN",  # Example: United Oil Office Portable Generators
        "opportunity_name": "United Oil Office Portable Generators",
        "raw_note": (
            "Has 5 vendors; wants vendor comparison.\n"
            "Needs scalability to 5,000 users.\n"
            "Agreed to send basic requirements doc."
        ),
    },
    {
        "opportunity_id": "006gL00000A4GZvQAN",  # Example: Express Logistics Standby Generator
        "opportunity_name": "Express Logistics Standby Generator",
        "raw_note": (
            "Seeking AI-driven analytics.\n"
            "IT already has BI tools — curious about overlay.\n"
            "Agreed to a follow-up demo with BI team."
        ),
    },
]


def main() -> None:
    logger.info("=== Salesforce Batch Note Ingestion Engine ===")
    sf_client = SalesforceClient()

    for item in _DEMO_NOTES:
        opp_id = item["opportunity_id"]
        opp_name = item["opportunity_name"]
        raw = item["raw_note"]

        logger.info("Processing: %s (%s)", opp_name, opp_id)
        logger.debug("Raw note:\n%s", raw)

        # 1. Format raw note with the fine-tuned LoRA model
        logger.info("Formatting via Local Gemma-2B LoRA Adapter...")
        formatted_note = format_notes_with_lora(raw)
        logger.info("Formatted note:\n%s", formatted_note)

        # 2. Sync formatted note to Salesforce
        try:
            response = sf_client.create_note(
                opp_id, formatted_note, title=f"Meeting Notes — {opp_name}"
            )
            logger.info("Note created in Salesforce: %s", response)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to create note for '%s': %s", opp_name, exc)


if __name__ == "__main__":
    main()
