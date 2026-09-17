"""
backend/tools.py
================
OpenAI/Gemini-compatible function-calling tool definitions for the CRM Agent.

All tools are defined in the standard JSON Schema format expected by the
Gemini and Mistral function-calling APIs.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "searchOpportunities",
            "description": "Find Salesforce Opportunities by partial or full name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "opportunityName": {
                        "type": "string",
                        "description": "Partial or full name of the opportunity to search for.",
                    }
                },
                "required": ["opportunityName"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getOpportunity",
            "description": "Fetch a single Salesforce Opportunity record by its Id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "opportunityId": {
                        "type": "string",
                        "description": "Salesforce Opportunity Id (15 or 18-character).",
                    }
                },
                "required": ["opportunityId"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "searchAccounts",
            "description": "Find Salesforce Accounts by partial or full name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "accountName": {
                        "type": "string",
                        "description": "Partial or full account name to search for.",
                    }
                },
                "required": ["accountName"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getLatestNotes",
            "description": (
                "Retrieve the most recent Salesforce Notes attached to a specific Opportunity. "
                "Use when the user asks for 'latest notes', 'show notes', or 'get notes'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "opportunityId": {
                        "type": "string",
                        "description": "Salesforce Opportunity Id.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of notes to return.",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["opportunityId"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "formatNotes",
            "description": (
                "Format raw, messy meeting transcripts, call notes, or unstructured text into "
                "a standard structured 5-section Salesforce CRM note using the fine-tuned LoRA model. "
                "ALWAYS call this tool when the user provides meeting notes, transcripts, or asks to "
                "prepare or format notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rawText": {
                        "type": "string",
                        "description": "The raw unstructured notes or meeting transcript to format.",
                    }
                },
                "required": ["rawText"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "createNotes",
            "description": "Create a Note (Title + Body) and attach it to a Salesforce Opportunity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "opportunityId": {
                        "type": "string",
                        "description": "Salesforce Opportunity Id to attach the note to.",
                    },
                    "newNoteToAdd": {
                        "type": "string",
                        "description": "The note body content.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Note title.",
                        "default": "Meeting Notes",
                    },
                },
                "required": ["opportunityId", "newNoteToAdd"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "executeSOQL",
            "description": (
                "Execute any read-only Salesforce Object Query Language (SOQL) SELECT query "
                "to fetch, count, aggregate, filter, or inspect CRM records "
                "(Opportunities, Accounts, Contacts, Notes). "
                "Use when the user asks for counts, aggregates, rankings, lists, or custom filters. "
                "Examples: "
                "'SELECT COUNT() FROM Opportunity', "
                "'SELECT Name, Amount, StageName FROM Opportunity ORDER BY Amount DESC LIMIT 5'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A valid SOQL query starting with SELECT.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]
