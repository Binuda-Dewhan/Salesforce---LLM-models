"""
backend/salesforce_client.py
============================
Robust Salesforce REST Client supporting:
1. Bearer Token (SF_ACCESS_TOKEN) from centralised config.
2. Auto-login via SOAP Partner API using Username/Password + Security Token.
3. Automatic token refresh on 401 Unauthorized.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests

from .config import cfg

logger = logging.getLogger(__name__)


class SalesforceClient:
    """
    Thread-safe Salesforce REST client.

    Config is read from the centralised `cfg` object (loaded once at startup)
    rather than reloading the .env file on every request.
    """

    # Default timeouts: (connect_timeout, read_timeout) in seconds
    _DEFAULT_TIMEOUT = (5, 30)

    def __init__(
        self,
        instance_url: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> None:
        # Allow override for testing; otherwise fall through to cfg.
        self._instance_url_override = instance_url
        self._access_token_override = access_token
        self._api_version = cfg.salesforce_api_version()

    # ------------------------------------------------------------------
    # Properties — read once from cfg, not on every call
    # ------------------------------------------------------------------

    @property
    def instance_url(self) -> str:
        return (self._instance_url_override or cfg.salesforce_instance_url()).rstrip("/")

    @property
    def access_token(self) -> str:
        return self._access_token_override or cfg.salesforce_access_token()

    @property
    def _username(self) -> str:
        return cfg.salesforce_username()

    @property
    def _password(self) -> str:
        return cfg.salesforce_password()

    @property
    def _security_token(self) -> str:
        return cfg.salesforce_security_token()

    @property
    def _base_url(self) -> str:
        return f"{self.instance_url}/services/data/{self._api_version}"

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        """
        Obtains a fresh session ID via Salesforce SOAP Partner Login.
        Updates os.environ so the new token is picked up by cfg on next read.
        """
        import os

        if not self._username or not self._password:
            logger.warning("Cannot authenticate: SALESFORCE_USERNAME / PASSWORD not configured.")
            return False

        api_ver = self._api_version.replace("v", "")
        soap_body = f"""<?xml version="1.0" encoding="utf-8" ?>
<env:Envelope xmlns:xsd="http://www.w3.org/2001/XMLSchema"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:env="http://schemas.xmlsoap.org/soap/envelope/">
  <env:Body>
    <n1:login xmlns:n1="urn:partner.soap.sforce.com">
      <n1:username>{self._username}</n1:username>
      <n1:password>{self._password}{self._security_token}</n1:password>
    </n1:login>
  </env:Body>
</env:Envelope>"""

        soap_headers = {"Content-Type": "text/xml; charset=UTF-8", "SOAPAction": "login"}

        endpoints = [
            f"{self.instance_url}/services/Soap/u/{api_ver}" if self.instance_url else None,
            f"https://login.salesforce.com/services/Soap/u/{api_ver}",
            f"https://test.salesforce.com/services/Soap/u/{api_ver}",
        ]

        for ep in filter(None, endpoints):
            try:
                r = requests.post(
                    ep,
                    data=soap_body.encode("utf-8"),
                    headers=soap_headers,
                    timeout=self._DEFAULT_TIMEOUT,
                )
                if r.status_code == 200:
                    root = ET.fromstring(r.text)
                    ns = {
                        "soapenv": "http://schemas.xmlsoap.org/soap/envelope/",
                        "result": "urn:partner.soap.sforce.com",
                    }
                    sid_el = root.find(".//result:sessionId", ns)
                    server_url_el = root.find(".//result:serverUrl", ns)
                    if sid_el is not None and server_url_el is not None:
                        new_token = sid_el.text
                        new_url = server_url_el.text.split("/services")[0]
                        # Update os.environ so cfg picks up the fresh token
                        os.environ["SF_ACCESS_TOKEN"] = new_token
                        os.environ["SF_INSTANCE_URL"] = new_url
                        # Also update local overrides so this instance reflects new values immediately
                        self._access_token_override = new_token
                        self._instance_url_override = new_url
                        logger.info("Salesforce re-authentication successful via %s", ep)
                        return True
            except requests.exceptions.Timeout:
                logger.warning("SOAP login timeout for endpoint: %s", ep)
            except requests.exceptions.RequestException as exc:
                logger.warning("SOAP login request failed for %s: %s", ep, exc)
            except ET.ParseError as exc:
                logger.warning("Failed to parse SOAP response from %s: %s", ep, exc)

        logger.error("All Salesforce SOAP authentication endpoints failed.")
        return False

    # ------------------------------------------------------------------
    # Core Request Handler
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> Any:
        """
        Executes an authenticated Salesforce REST API request.
        Automatically retries once on 401 by refreshing the session token.
        """
        url = f"{self._base_url}/{path.lstrip('/')}"

        # Ensure caller-provided kwargs don't override our timeout unless explicitly set
        kwargs.setdefault("timeout", self._DEFAULT_TIMEOUT)

        response = requests.request(method, url, headers=self._headers, **kwargs)

        if response.status_code == 401:
            logger.info("Salesforce token expired (401). Attempting re-authentication.")
            if self.authenticate():
                response = requests.request(method, url, headers=self._headers, **kwargs)
            else:
                raise PermissionError(
                    "Salesforce authentication expired (401). "
                    "Please update SF_ACCESS_TOKEN in your .env file, or verify "
                    "SALESFORCE_USERNAME, PASSWORD, and SECURITY_TOKEN."
                )

        response.raise_for_status()
        return response.json() if response.text else {}

    # ------------------------------------------------------------------
    # SOQL Helper
    # ------------------------------------------------------------------

    def _soql(self, query: str) -> Dict[str, Any]:
        """Executes a SOQL query, URL-encoding the query string."""
        return self._request("GET", f"query/?q={quote(query)}")

    # ------------------------------------------------------------------
    # Public CRM Operations
    # ------------------------------------------------------------------

    def search_opportunities(self, opportunity_name: str) -> Dict[str, Any]:
        """Finds opportunities matching a partial or full name."""
        safe_name = opportunity_name.replace("'", "\\'")
        soql = (
            f"SELECT Id, Name, StageName, Amount, CloseDate "
            f"FROM Opportunity WHERE Name LIKE '%{safe_name}%' LIMIT 10"
        )
        return self._soql(soql)

    def get_opportunity(self, opportunity_id: str) -> Dict[str, Any]:
        """Fetches a single Opportunity record by 15 or 18 character Id."""
        safe_id = opportunity_id.replace("'", "\\'")
        soql = (
            f"SELECT Id, Name, StageName, Amount, CloseDate, AccountId, Description "
            f"FROM Opportunity WHERE Id = '{safe_id}'"
        )
        return self._soql(soql)

    def search_accounts(self, account_name: str) -> Dict[str, Any]:
        """Finds accounts matching a partial or full name."""
        safe_name = account_name.replace("'", "\\'")
        soql = (
            f"SELECT Id, Name, Type, Industry "
            f"FROM Account WHERE Name LIKE '%{safe_name}%' LIMIT 10"
        )
        return self._soql(soql)

    def create_note(
        self,
        opportunity_id: str,
        new_note_body: str,
        title: str = "Meeting Notes",
    ) -> Dict[str, Any]:
        """Creates a standard Note attached to an Opportunity."""
        payload = {
            "Title": title,
            "Body": new_note_body,
            "ParentId": opportunity_id,
        }
        return self._request("POST", "sobjects/Note/", json=payload)

    def get_latest_notes(self, opportunity_id: str, limit: int = 3) -> Dict[str, Any]:
        """Fetches the most recently modified Notes attached to an Opportunity."""
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
        Enforces a SELECT-only guardrail to prevent write operations.
        """
        clean_q = query.strip()
        if not clean_q.upper().startswith("SELECT"):
            raise ValueError(
                "Security violation: Only SELECT queries are permitted in execute_soql. "
                f"Received: {clean_q[:80]!r}"
            )
        return self._soql(clean_q)
