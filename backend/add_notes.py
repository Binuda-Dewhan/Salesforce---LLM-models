from salesforce_client import SalesforceClient

# Replace with your Salesforce instance URL and access token
SF_INSTANCE_URL = "https://orgfarm-c00ad172f3-dev-ed.develop.my.salesforce.com"  # Your Salesforce URL
SF_ACCESS_TOKEN = "00DgL000008h5hN!AQEAQA4JsdTCkImja83Wvvk_rVt7hNqwpj5hh11WM2siOIYAiSNpDE1hKV1x8rflvqbWGYsykZd.1CtdOuC4luM1Fqnm2.KQ"  # Your Salesforce Access Token

# Initialize Salesforce client
sf_client = SalesforceClient(SF_INSTANCE_URL, SF_ACCESS_TOKEN)

# List of notes to add to Salesforce for each opportunity
notes = [
    {
        "opportunity_id": "006gL00000A4GZtQAN",  # Acme Corp
        "note": """Explored current infrastructure; using legacy CRM.
Interested in reducing manual data entry by 40%.
Identified budget stage and decision timeline (Q4)."""
    },
    {
        "opportunity_id": "006gL00000A4GZuQAN",  # United Oil Office Portable Generators
        "note": """Has 5 vendors; wants vendor comparison.
Needs scalability to 5,000 users.
Agreed to send basic requirements doc."""
    },
    {
        "opportunity_id": "006gL00000A4GZvQAN",  # Express Logistics Standby Generator
        "note": """Seeking AI-driven analytics.
IT already has BI tools—curious about overlay.
Agreed to a follow-up demo with BI team."""
    },
    {
        "opportunity_id": "006gL00000A4GZwQAN",  # GenePoint Standby Generator
        "note": """Must comply with HIPAA.
Current system 8 years old, unsupported.
Recommended compliance checklist call for next week."""
    },
    {
        "opportunity_id": "006gL00000A4GZxQAN",  # Grand Hotels Kitchen Generator
        "note": """Freight company needs route optimization.
Interested in mobile access.
Will share fleet size and current app stack."""
    },
    {
        "opportunity_id": "006gL00000A4GZyQAN",  # United Oil Refinery Generators
        "note": """Eager for fraud detection module.
Wants ROI figures.
Qualified—moves to needs analysis."""
    },
    {
        "opportunity_id": "006gL00000A4GZzQAN",  # United Oil SLA
        "note": """Looking for LMS integration.
10,000 student base.
Wants to see onboarding demo."""
    },
    {
        "opportunity_id": "006gL00000A4Ga0QAF",  # Grand Hotels Guest Portable Generators
        "note": """Struggling with inventory sync.
Wants cloud migration plan.
Will send existing inventory reports."""
    },
    {
        "opportunity_id": "006gL00000A4Ga1QAF",  # Edge Emergency Generator
        "note": """Needs validation for clinical data.
FDA compliance focus.
Offered a pharmaceutical use-case whitepaper."""
    },
    {
        "opportunity_id": "006gL00000A4Ga2QAF",  # University of AZ Portable Generators
        "note": """Exploring IoT dashboard analytics.
20 sensors installed.
Wants to see IoT integration capabilities."""
    }
]

# Add the notes to Salesforce
for note in notes:
    try:
        # Call the create_note method from SalesforceClient to create the note
        response = sf_client.create_note(note["opportunity_id"], note["note"])
        print(f"Note created for Opportunity ID {note['opportunity_id']}: {response}")
    except Exception as e:
        print(f"Error creating note for Opportunity ID {note['opportunity_id']}: {e}")
