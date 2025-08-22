TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "searchOpportunities",
            "description": "Find opportunities by (partial) name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "opportunityName": {"type": "string", "description": "Partial or full name of the opportunity."}
                },
                "required": ["opportunityName"],
                "additionalProperties": False
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "getLatestNotes",
            "description": "Get the latest Salesforce Notes for an Opportunity. Use this when the user asks for 'latest notes' or 'show notes'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "opportunityId": {"type": "string", "description": "Salesforce Opportunity Id"},
                    "limit": {"type": "integer", "default": 3, "minimum": 1, "maximum": 20}
                },
                "required": ["opportunityId"],
                "additionalProperties": False
            }
        }
    }


    ,
    {
        "type": "function",
        "function": {
            "name": "getOpportunity",
            "description": "Fetch a single opportunity by Id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "opportunityId": {"type": "string", "description": "Salesforce Opportunity Id (18-char)."}
                },
                "required": ["opportunityId"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "searchAccounts",
            "description": "Find accounts by (partial) name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "accountName": {"type": "string", "description": "Partial or full account name."}
                },
                "required": ["accountName"],
                "additionalProperties": False
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "formatNotes",
            "description": "Format raw meeting transcript into structured Salesforce notes",
            "parameters": {
            "type": "object",
            "properties": {
                "rawText": {"type": "string"}
            },
            "required": ["rawText"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "createNotes",
            "description": "Create a Note (Title+Body) attached to an Opportunity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "opportunityId": {"type": "string"},
                    "newNoteToAdd": {"type": "string"},
                    "title": {"type": "string", "default": "Meeting Notes"}
                },
                "required": ["opportunityId", "newNoteToAdd"],
                "additionalProperties": False
            }
        }
    }
]
