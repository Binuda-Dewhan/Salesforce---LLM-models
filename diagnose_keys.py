"""
Quick API key diagnostic - run this to identify what is broken.
Usage: .\\venv\\Scripts\\python.exe diagnose_keys.py
"""
import io
import os
import sys
import requests

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, ".")
from backend.config import cfg

def ok(msg):  return f"    STATUS : [OK]   {msg}"
def fail(msg): return f"    STATUS : [FAIL] {msg}"
def warn(msg): return f"    STATUS : [WARN] {msg}"

print("=" * 60)
print("  API Key Diagnostic")
print("=" * 60)

# --- Check Gemini ---
gemini_key = cfg.gemini_api_key()
print(f"\n[1] GEMINI_API_KEY")
if not gemini_key:
    print(fail("MISSING - not set in .env"))
elif len(gemini_key) < 20:
    print(fail(f"TOO SHORT - value looks wrong: {gemini_key!r}"))
else:
    print(f"    KEY    : {gemini_key[:8]}...{gemini_key[-4:]} ({len(gemini_key)} chars)")
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {gemini_key}", "Content-Type": "application/json"},
            json={"model": "gemini-2.0-flash", "messages": [{"role": "user", "content": "Say hi"}]},
            timeout=10,
        )
        if r.status_code == 200:
            print(ok("WORKING - gemini-2.0-flash responded"))
        elif r.status_code == 401:
            print(fail("INVALID KEY - API returned 401 Unauthorized"))
            print(f"    BODY   : {r.text[:300]}")
        elif r.status_code == 404:
            print(fail("KEY/MODEL ISSUE - API returned 404"))
            print(f"    BODY   : {r.text[:400]}")
            print()
            print("    FIX    : Get a fresh key from https://aistudio.google.com/apikey")
            print("             Your current key may not have the OpenAI-compat endpoint enabled.")
        elif r.status_code == 429:
            print(warn("QUOTA EXCEEDED - key is valid but rate limited (free tier limit hit)"))
        else:
            print(fail(f"Unexpected HTTP {r.status_code}"))
            print(f"    BODY   : {r.text[:300]}")
    except requests.exceptions.Timeout:
        print(fail("Request timed out"))
    except Exception as e:
        print(fail(f"Request failed: {e}"))

# --- Check Mistral ---
mistral_key = cfg.mistral_api_key()
print(f"\n[2] MISTRAL_API_KEY")
if not mistral_key:
    print(fail("MISSING - not set in .env"))
elif len(mistral_key) < 10:
    print(fail(f"TOO SHORT - value looks wrong: {mistral_key!r}"))
else:
    print(f"    KEY    : {mistral_key[:6]}...{mistral_key[-4:]} ({len(mistral_key)} chars)")
    try:
        r = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {mistral_key}", "Content-Type": "application/json"},
            json={"model": "mistral-small-latest", "messages": [{"role": "user", "content": "Hi"}]},
            timeout=10,
        )
        if r.status_code == 200:
            print(ok("WORKING - mistral-small-latest responded"))
        elif r.status_code == 401:
            print(fail("INVALID KEY - API returned 401 Unauthorized"))
            print("    FIX    : Get a new key from https://console.mistral.ai/api-keys")
        elif r.status_code == 429:
            print(warn("QUOTA EXCEEDED - key valid but rate limited"))
        else:
            print(fail(f"HTTP {r.status_code}: {r.text[:200]}"))
    except requests.exceptions.Timeout:
        print(fail("Request timed out"))
    except Exception as e:
        print(fail(f"Request failed: {e}"))

# --- Check Salesforce ---
sf_url = cfg.salesforce_instance_url()
sf_tok = cfg.salesforce_access_token()
print(f"\n[3] Salesforce")
if not sf_url:
    print(fail("SF_INSTANCE_URL not set in .env"))
elif not sf_tok:
    print(fail("SF_ACCESS_TOKEN not set in .env"))
else:
    print(f"    URL    : {sf_url}")
    print(f"    TOKEN  : {sf_tok[:8]}...{sf_tok[-4:]}")
    try:
        r = requests.get(
            f"{sf_url}/services/data/v61.0/",
            headers={"Authorization": f"Bearer {sf_tok}"},
            timeout=10,
        )
        if r.status_code == 200:
            print(ok("WORKING - Salesforce org is reachable"))
        elif r.status_code == 401:
            print(warn("TOKEN EXPIRED - Salesforce returned 401 (will auto-refresh via SOAP login)"))
            print("    FIX    : Either update SF_ACCESS_TOKEN or ensure SALESFORCE_USERNAME/PASSWORD are set")
        else:
            print(fail(f"HTTP {r.status_code}: {r.text[:200]}"))
    except requests.exceptions.Timeout:
        print(fail("Request timed out - check SF_INSTANCE_URL"))
    except Exception as e:
        print(fail(f"Request failed: {e}"))

print("\n" + "=" * 60)
print("  Done.")
print("=" * 60)
