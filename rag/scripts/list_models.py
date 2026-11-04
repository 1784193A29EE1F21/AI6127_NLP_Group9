"""List the models available for your API key and send one test request.

Usage: uv run python scripts/list_models.py
"""
import os

import httpx

from rag import config

if config.LLM_PROVIDER == "openai":
    r = httpx.get(f"{config.OPENAI_BASE_URL.rstrip('/')}/models",
                  headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}, timeout=30)
    print("status:", r.status_code)
    if r.status_code != 200:
        print(r.text[:500])
    else:
        ids = sorted(m["id"] for m in r.json().get("data", []))
        print("available models:")
        for i in ids:
            print("  ", i)
else:
    from google import genai

    c = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    for m in c.models.list():
        print("  ", m.name)

print("\ncurrent LLM_MODEL in .env:", config.LLM_MODEL)
from rag.llm import LLM

try:
    print("test answer:", LLM().generate("Say hello in one short sentence.", use_cache=False))
except Exception as e:
    print("TEST FAILED:", e)
