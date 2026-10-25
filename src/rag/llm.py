"""Pluggable LLM backends. Select with env LLM_PROVIDER = gemini | ollama."""
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config


class LLM:
    name = "base"

    def generate(self, system: str, prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError


class GeminiLLM(LLM):
    """Google Gemini via the free-tier API key from https://aistudio.google.com/apikey."""

    def __init__(self, model: str | None = None, fallbacks: list[str] | None = None, retries: int = 3):
        from google import genai
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        # Free-tier quotas are per model, so a fallback chain multiplies the daily budget.
        self.fallbacks = fallbacks or [m.strip() for m in os.getenv(
            "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite").split(",") if m.strip()]
        self.retries = int(os.getenv("GEMINI_RETRIES", retries))
        self.name = f"gemini/{self.model}"
        # Retries are handled below; the SDK's own retry loop would multiply the wait.
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=genai.types.HttpOptions(
            timeout=int(os.getenv("GEMINI_TIMEOUT_MS", "30000")),
            retry_options=genai.types.HttpRetryOptions(attempts=1)))

    def _call(self, model, system, prompt, temperature):
        from google.genai import types
        r = self.client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=temperature))
        return r.text or ""

    def generate(self, system, prompt, temperature=0.0):
        from google.genai import errors
        # Free tier returns 503 (overloaded) / 429 (rate limit) often: back off, then try the fallbacks.
        for model in [self.model, *self.fallbacks]:
            for attempt in range(self.retries):
                try:
                    return self._call(model, system, prompt, temperature)
                except errors.APIError as e:
                    if e.code not in (429, 500, 503, 504):
                        raise
                    if e.code == 504 or (e.code == 429 and "quota" in str(e).lower()):
                        break  # timed out, or this model's daily quota is gone: move to the next model
                    time.sleep(2 ** attempt * 2)
                except Exception as e:  # noqa: BLE001 - client-side timeouts surface as httpx errors
                    if "timeout" not in type(e).__name__.lower():
                        raise
                    break
        raise RuntimeError("Gemini unavailable after retries on primary and fallback models")


class OllamaLLM(LLM):
    """Local model served by Ollama (https://ollama.com), e.g. `ollama pull qwen2.5:7b`."""

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.name = f"ollama/{self.model}"

    def _call(self, system, prompt, temperature):
        payload = {
            "model": self.model, "stream": False,
            # Greedy decoding on long prompts can loop; a mild repeat penalty + fixed seed keeps runs stable.
            "options": {"temperature": temperature, "num_ctx": 8192, "repeat_penalty": 1.1, "seed": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        # Small local models drift out of JSON; Ollama can constrain decoding to valid JSON.
        if "json only" in system.lower():
            payload["format"] = "json"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"]

    def _reload(self):
        """Unload the model so the next request starts a fresh runner."""
        body = json.dumps({"model": self.model, "keep_alive": 0}).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120):
            pass

    @staticmethod
    def _degenerate(text: str) -> bool:
        head = text.strip()[:60]
        return not head or re.fullmatch(r"(.)\1{15,}.*", head, re.S) is not None

    def generate(self, system, prompt, temperature=0.0):
        # After many requests the Ollama runner (0.40 on an RTX 50xx) can get into a bad state and
        # emit "8888..." until it aborts with HTTP 500 ("token repeat limit"). Reloading the model
        # fixes it, so on a degenerate reply we unload and retry.
        for attempt in range(3):
            try:
                out = self._call(system, prompt, temperature)
                if not self._degenerate(out):
                    return out
            except urllib.error.HTTPError as e:
                if e.code != 500:
                    raise
            self._reload()
        raise RuntimeError("Ollama returned degenerate output after reloads")


class CachedLLM(LLM):
    """SQLite-backed response cache: reruns of evals are free and interrupted runs resume."""

    def __init__(self, inner: LLM, path: Path):
        self.inner, self.name = inner, inner.name
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
        self.lock = threading.Lock()

    def _key(self, system, prompt, temperature):
        return hashlib.sha256(json.dumps([self.inner.name, system, prompt, temperature]).encode()).hexdigest()

    def contains(self, system, prompt, temperature=0.0) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM cache WHERE key = ?",
                                   (self._key(system, prompt, temperature),)).fetchone() is not None

    def generate(self, system, prompt, temperature=0.0):
        key = self._key(system, prompt, temperature)
        with self.lock:
            row = self.db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            return row[0]
        out = self.inner.generate(system, prompt, temperature)
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, out))
            self.db.commit()
        return out


def parse_json(text: str):
    """Extract the first JSON object/array from a model reply (tolerates ```json fences)."""
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    return json.loads(m.group(0))


def get_llm(provider: str | None = None, model: str | None = None, cache: bool = True) -> LLM:
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")
    if provider == "gemini":
        llm = GeminiLLM(model)
    elif provider == "ollama":
        llm = OllamaLLM(model)
    else:
        raise ValueError(f"unknown LLM provider {provider}")
    return CachedLLM(llm, config.DATA / "cache" / "llm_cache.sqlite") if cache else llm
