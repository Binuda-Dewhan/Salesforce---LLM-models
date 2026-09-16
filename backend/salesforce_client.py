import os
import json
import xml.etree.ElementTree as ET
from urllib.parse import quote
from typing import Dict, Any, Optional
import requests

def load_env_file(filepath=".env"):
    """Pure-Python .env file parser with fallback across parent directories."""
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
                        # Always update with latest value from .env
                        os.environ[k] = v
            break

load_env_file()

API_VERSION = os.getenv("SALESFORCE_API_VERSION", "v61.0")


class SalesforceClient:
    """
    Robust Salesforce REST Client supporting:
    1. Dynamic Bearer Token (SF_ACCESS_TOKEN) with live .env hot-reloading
    2. Auto-login via SOAP Partner API using Username/Password + Security Token
    3. Automatic token refresh on 401 Unauthorized
    """

    def __init__(self, instance_url: Optional[str] = None, access_token: Optional[str] = None):
        self._custom_instance_url = instance_url
        self._custom_access_token = access_token
        self.api_version = API_VERSION

    @property
    def instance_url(self) -> str:
        load_env_file()
        return (self._custom_instance_url or os.getenv("SF_INSTANCE_URL", "")).strip().rstrip("/")

    @property
    def access_token(self) -> str:
        load_env_file()
        return (self._custom_access_token or os.getenv("SF_ACCESS_TOKEN", "")).strip()

    @property
    def username(self) -> str:
        load_env_file()
        return os.getenv("SALESFORCE_USERNAME", "").strip()

    @property
    def password(self) -> str:
        load_env_file()
        return os.getenv("SALESFORCE_PASSWORD", "").strip()

    @property
    def security_token(self) -> str:
        load_env_file()
        return os.getenv("SALESFORCE_SECURITY_TOKEN", "").strip()

    @property
    def base_url(self) -> str:
        return f"{self.instance_url}/services/data/{self.api_version}"

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    def authenticate(self) -> bool:
        """Authenticates using Salesforce SOAP Partner Login to obtain a fresh session ID."""
        if not self.username or not self.password:
            return False

        soap_body = f"""<?xml version="1.0" encoding="utf-8" ?>
<env:Envelope xmlns:xsd="http://www.w3.org/2001/XMLSchema"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:env="http://schemas.xmlsoap.org/soap/envelope/">
  <env:Body>
    <n1:login xmlns:n1="urn:partner.soap.sforce.com">
      <n1:username>{self.username}</n1:username>
      <n1:password>{self.password}{self.security_token}</n1:password>
    </n1:login>
  </env:Body>
</env:Envelope>"""

        headers = {
            "Content-Type": "text/xml; charset=UTF-8",
            "SOAPAction": "login"
        }

        endpoints = [
            f"{self.instance_url}/services/Soap/u/{self.api_version.replace('v', '')}" if self.instance_url else None,
            f"https://login.salesforce.com/services/Soap/u/{self.api_version.replace('v', '')}",
            f"https://test.salesforce.com/services/Soap/u/{self.api_version.replace('v', '')}"
        ]

        for ep in filter(None, endpoints):
            try:
                r = requests.post(ep, data=soap_body.encode("utf-8"), headers=headers, timeout=15)
                if r.status_code == 200:
                    root = ET.fromstring(r.text)
                    namespaces = {
                        "soapenv": "http://schemas.xmlsoap.org/soap/envelope/",
                        "result": "urn:partner.soap.sforce.com"
                    }
                    sid_el = root.find(".//result:sessionId", namespaces)
                    server_url_el = root.find(".//result:serverUrl", namespaces)
                    if sid_el is not None and server_url_el is not None:
                        self.access_token = sid_el.text
                        self.instance_url = server_url_el.text.split("/services")[0]
                        os.environ["SF_ACCESS_TOKEN"] = self.access_token
                        os.environ["SF_INSTANCE_URL"] = self.instance_url
                        return True
            except Exception:
                continue
        return False

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        
        # Initial request
        r = requests.request(method, url, headers=self.headers, **kwargs)
        
        # Auto-retry on 401 Token Expiration
        if r.status_code == 401:
            if self.authenticate():
                r = requests.request(method, url, headers=self.headers, **kwargs)
            else:
                raise PermissionError(
                    "Salesforce authentication expired (401). Please update SF_ACCESS_TOKEN in your .env file "
                    "or verify your SALESFORCE_USERNAME, PASSWORD, and SECURITY_TOKEN."
                )

        r.raise_for_status()
        return r.json() if r.text else {}

    def _soql(self, query: str) -> Dict[str, Any]:
        """Executes a SOQL query safely."""
        encoded_query = quote(query)
        return self._request("GET", f"query/?q={encoded_query}")

    def search_opportunities(self, opportunity_name: str) -> Dict[str, Any]:
        """Finds opportunities matching partial or full name."""
        safe_name = opportunity_name.replace("'", "\\'")
        soql = f"SELECT Id, Name, StageName, Amount, CloseDate FROM Opportunity WHERE Name LIKE '%{safe_name}%' LIMIT 10"
        return self._soql(soql)

    def get_opportunity(self, opportunity_id: str) -> Dict[str, Any]:
        """Fetches detailed Opportunity record by 15 or 18 character Id."""
        safe_id = opportunity_id.replace("'", "\\'")
        soql = f"SELECT Id, Name, StageName, Amount, CloseDate, AccountId, Description FROM Opportunity WHERE Id = '{safe_id}'"
        return self._soql(soql)

    def search_accounts(self, account_name: str) -> Dict[str, Any]:
        """Finds accounts matching partial or full name."""
        safe_name = account_name.replace("'", "\\'")
        soql = f"SELECT Id, Name, Type, Industry FROM Account WHERE Name LIKE '%{safe_name}%' LIMIT 10"
        return self._soql(soql)

    def create_note(self, opportunity_id: str, new_note_body: str, title: str = "Meeting Notes") -> Dict[str, Any]:
        """Creates a standard Note attached to an Opportunity."""
        payload = {
            "Title": title,
            "Body": new_note_body,
            "ParentId": opportunity_id
        }
        return self._request("POST", "sobjects/Note/", json=payload)

    def get_latest_notes(self, opportunity_id: str, limit: int = 3) -> Dict[str, Any]:
        """Fetches the latest notes attached to an Opportunity ordered by modification date."""
        safe_id = opportunity_id.replace("'", "\\'")
        soql = (
            f"SELECT Id, Title, Body, LastModifiedDate, CreatedDate "
            f"FROM Note WHERE ParentId = '{safe_id}' "
            f"ORDER BY LastModifiedDate DESC LIMIT {int(limit)}"
        )
        return self._soql(soql)

    def execute_soql(self, query: str) -> Dict[str, Any]:
        """
        Executes an arbitrary read-only SOQL query against Salesforce.
        Enforces read-only safety guardrails (only SELECT queries permitted).
        """
        clean_q = query.strip()
        if not clean_q.upper().startswith("SELECT"):
            raise ValueError("Security violation: Only SELECT queries are permitted in execute_soql.")
        return self._soql(clean_q)
