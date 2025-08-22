import os
import json
import requests
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

SF_INSTANCE_URL = os.getenv("SF_INSTANCE_URL", "").rstrip("/")
SF_ACCESS_TOKEN = os.getenv("SF_ACCESS_TOKEN")
API_VERSION = "v61.0"  # works with v52+; bump if your org supports newer

class SalesforceClient:
    def __init__(self, instance_url: str = SF_INSTANCE_URL, access_token: str = SF_ACCESS_TOKEN):
        assert instance_url and access_token, "Missing Salesforce config"
        self.base = f"{instance_url}/services/data/{API_VERSION}"
        self.headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    def _soql(self, soql: str):
        url = f"{self.base}/query/?q={quote(soql)}"  # URL-encode SOQL
        r = requests.get(url, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    # Task 1 functions
    def search_opportunities(self, opportunity_name: str):
        safe_name = opportunity_name.replace("'", "''")  # escape single quotes for SOQL
        soql = f"SELECT Id, Name FROM Opportunity WHERE Name LIKE '%{safe_name}%'"
        return self._soql(soql)

    def get_opportunity(self, opportunity_id: str):
        soql = f"SELECT Id, Name, StageName FROM Opportunity WHERE Id = '{opportunity_id}'"
        return self._soql(soql)

    def search_accounts(self, account_name: str):
        safe_name = account_name.replace("'", "''")
        soql = f"SELECT Id, Name FROM Account WHERE Name LIKE '%{safe_name}%'"
        return self._soql(soql)

    def create_note(self, opportunity_id: str, new_note_body: str, title: str = "Meeting Notes"):
        # NOTE: Classic Note fields are Title, Body, ParentId (Body is long text)
        url = f"{self.base}/sobjects/Note/"
        payload = {"Title": title, "Body": new_note_body, "ParentId": opportunity_id}
        r = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        return r.json()
    
    def get_latest_notes(self, opportunity_id: str, limit: int = 3):
        # Classic Note object; you used it for create_note so read from it too
        soql = (
            f"SELECT Id, Title, Body, LastModifiedDate, CreatedDate "
            f"FROM Note WHERE ParentId = '{opportunity_id}' "
            f"ORDER BY LastModifiedDate DESC LIMIT {int(limit)}"
        )
        return self._soql(soql)

