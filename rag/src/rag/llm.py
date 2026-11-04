"""LLM client (Gemini or OpenAI-compatible API) with cache, rate limit and retries."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class QuotaExceeded(RuntimeError):
    """Raised when the provider's daily request/token quota is exhausted."""


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 cache_dir: Path | None = None, rpm: int | None = None,
                 base_url: str | None = None, api_key: str | None = None):
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.model = model or config.LLM_MODEL
        self.cache_dir = Path(cache_dir or config.PATHS.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rpm = config.LLM_RPM if rpm is None else rpm
        self.stats = LLMStats()
        self._lock = threading.Lock()
        self._last_call = 0.0
        if self.provider == "gemini":
            key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("Set GOOGLE_API_KEY in the .env file (never hard-code it).")
            from google import genai

            self._client = genai.Client(api_key=key)
        elif self.provider == "openai":
            import httpx

            self._http = httpx.Client(timeout=120)
            self._base = (base_url or config.OPENAI_BASE_URL).rstrip("/")
            self._key = api_key or os.environ.get("OPENAI_API_KEY", "ollama")
        else:
            raise ValueError(f"Unknown LLM_PROVIDER {self.provider}")

    def _cache_path(self, prompt: str, temperature: float, json_mode: bool) -> Path:
        endpoint = getattr(self, "_base", "")
        h = hashlib.sha256(f"{self.provider}|{endpoint}|{self.model}|{temperature}|{json_mode}|{prompt}".encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.json"

    def _throttle(self) -> None:
        if self.rpm and self.rpm > 0:
            with self._lock:
                wait = 60.0 / self.rpm - (time.time() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                self._last_call = time.time()

    def generate(self, prompt: str, temperature: float = 0.0, json_mode: bool = False,
                 use_cache: bool = True, max_retries: int = 8) -> str:
        cp = self._cache_path(prompt, temperature, json_mode)
        if use_cache and cp.exists():
            self.stats.cache_hits += 1
            return json.loads(cp.read_text(encoding="utf-8"))["text"]
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                self._throttle()
                t0 = time.perf_counter()
                text, pt, ot = self._call(prompt, temperature, json_mode)
                self.stats.seconds += time.perf_counter() - t0
                self.stats.calls += 1
                self.stats.prompt_tokens += pt
                self.stats.output_tokens += ot
                if use_cache:
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    cp.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
                return text
            except Exception as e:  # rate limits / transient network errors
                last_err = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (400, 401, 403, 404):  # wrong key / model name: retrying will not help
                    body = getattr(e.response, "text", "")[:500]
                    hint = {401: "API key is wrong.", 404: "model name not found - run scripts/list_models.py",
                            403: "key has no access.", 400: "bad request."}[status]
                    raise RuntimeError(f"LLM error {status} ({hint}) Server says: {body}") from e
                if status == 429:
                    body = getattr(e.response, "text", "")
                    if "per day" in body.lower() or "tpd" in body.lower() or "rpd" in body.lower():
                        raise QuotaExceeded(
                            f"Daily quota of {self.model} is used up - try again tomorrow (cached results "
                            f"are kept) or switch model/provider in .env. Server says: {body[:300]}") from e
                    try:  # the server tells us how long to wait
                        wait = float(e.response.headers.get("retry-after", 0))
                    except (TypeError, ValueError):
                        wait = 0
                    time.sleep(min(90, max(wait, 5 * (attempt + 1))))
                    continue
                time.sleep(min(60, 2 ** attempt * 2))
        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")

    def _call(self, prompt: str, temperature: float, json_mode: bool) -> tuple[str, int, int]:
        if self.provider == "gemini":
            from google.genai import types

            cfg = types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json" if json_mode else None,
            )
            r = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
            um = getattr(r, "usage_metadata", None)
            pt = getattr(um, "prompt_token_count", 0) or 0
            ot = getattr(um, "candidates_token_count", 0) or 0
            return r.text or "", pt, ot
        body = {"model": self.model, "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self._http.post(f"{self._base}/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {self._key}"})
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {}) or {}
        return (d["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0),
                u.get("completion_tokens", 0))

    def generate_json(self, prompt: str, temperature: float = 0.0, default=None):
        text = self.generate(prompt, temperature=temperature, json_mode=True)
        return parse_json(text, default)


def parse_json(text: str, default=None):
    """Robustly parse the first JSON object/array in an LLM response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = text.find(open_c), text.rfind(close_c)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except Exception:
                continue
    return default
