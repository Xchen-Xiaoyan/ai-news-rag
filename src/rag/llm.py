"""Pluggable LLM backends. Select with env LLM_PROVIDER = gemini | ollama."""
import json
import os
import time
import urllib.request


class LLM:
    name = "base"

    def generate(self, system: str, prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError


class GeminiLLM(LLM):
    """Google Gemini via the free-tier API key from https://aistudio.google.com/apikey."""

    def __init__(self, model: str | None = None, fallback: str | None = None, retries: int = 3):
        from google import genai
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.fallback = fallback or os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")
        self.retries = retries
        self.name = f"gemini/{self.model}"
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _call(self, model, system, prompt, temperature):
        from google.genai import types
        r = self.client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=temperature))
        return r.text or ""

    def generate(self, system, prompt, temperature=0.0):
        from google.genai import errors
        # Free tier returns 503 (overloaded) / 429 (rate limit) often: back off, then try the fallback model.
        for model in [self.model, self.fallback]:
            for attempt in range(self.retries):
                try:
                    return self._call(model, system, prompt, temperature)
                except errors.APIError as e:
                    if e.code not in (429, 500, 503):
                        raise
                    time.sleep(2 ** attempt * 2)
        raise RuntimeError("Gemini unavailable after retries on primary and fallback models")


class OllamaLLM(LLM):
    """Local model served by Ollama (https://ollama.com), e.g. `ollama pull qwen2.5:7b`."""

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.name = f"ollama/{self.model}"

    def generate(self, system, prompt, temperature=0.0):
        body = json.dumps({
            "model": self.model, "stream": False, "options": {"temperature": temperature, "num_ctx": 8192},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"]


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")
    if provider == "gemini":
        return GeminiLLM(model)
    if provider == "ollama":
        return OllamaLLM(model)
    raise ValueError(f"unknown LLM provider {provider}")
