import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL = "mistral-small-latest"  # or open-mistral-7b

client = Mistral(api_key=MISTRAL_API_KEY)

def chat_with_tools(messages, tools):
    resp = client.chat.complete(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    choice = resp.choices[0].message
    tool_calls = getattr(choice, "tool_calls", None)
    return choice, tool_calls
