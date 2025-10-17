"""
scripts/providers.py
---------------------------------------------------------
Purpose:
  Provide a unified LLM interface for three modes:
    - "sim"    : simulated miss (sleep), no real LLM
    - "gemini" : Google Gemini via google-generativeai
    - "hf"     : Hugging Face Inference API (text generation)

Usage:
  from scripts.providers import get_provider
  llm = get_provider(provider="gemini")   # or "hf" / "sim"
  text, latency_s = llm.chat("your prompt here")
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional

# Optional but nice: load .env automatically if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ---------------------------
# Base interface
# ---------------------------
class LLMProvider:
    """Abstract LLM interface with a single .chat(prompt) method."""
    def chat(self, prompt: str) -> Tuple[str, float]:
        """
        Returns:
          text: model response
          latency_s: measured call latency in seconds
        """
        raise NotImplementedError


# ---------------------------
# Simulated provider (no API calls)
# ---------------------------
@dataclass
class SimProvider(LLMProvider):
    """Simulate an LLM miss for pipeline/dev testing."""
    sleep_s: float = 0.05

    def chat(self, prompt: str) -> Tuple[str, float]:
        t0 = time.time()
        time.sleep(self.sleep_s)
        return "[SIMULATED ANSWER]", time.time() - t0


# ---------------------------
# Gemini provider (google-generativeai)
# ---------------------------
class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 0.0):
        """
        Uses google-generativeai. Requires:
          - pip install google-generativeai
          - GEMINI_API_KEY (preferred) or GOOGLE_API_KEY in env/.env
          - GEMINI_MODEL (default: gemini-1.5-flash)
        """
        try:
            import google.generativeai as genai
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "google-generativeai is not installed. Run: "
                "`pip install google-generativeai` in your active env."
            ) from e

        # Accept either env var name to avoid confusion
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY/GOOGLE_API_KEY (set it in .env).")

        genai.configure(api_key=self.api_key)
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.temperature = temperature
        self._model = genai.GenerativeModel(self.model_name)

        
    def chat(self, prompt: str) -> Tuple[str, float]:
        """
        Single-turn generation with simple backoff on 429 rate limits.
        Returns (text, latency_s).
        """
        import google.api_core.exceptions as gexc
        t_start = time.time()
        max_retries = 3
        delay = 5.0  # fallback delay if server doesn't provide one
        for attempt in range(max_retries + 1):
            try:
                resp = self._model.generate_content(
                    prompt,
                    generation_config={"temperature": self.temperature}
                )
                text = getattr(resp, "text", "") or ""
                return text.strip(), time.time() - t_start
            except gexc.ResourceExhausted as e:
                # Use server-suggested retry delay if present
                suggested = getattr(e, "retry_delay", None)
                sleep_s = float(getattr(suggested, "seconds", 0) or 0) or delay
                if attempt >= max_retries:
                    raise
                time.sleep(sleep_s)
            except Exception:
                # Bubble up other errors (network, auth, etc.)
                raise


# ---------------------------
# Hugging Face Inference API provider
# ---------------------------
class HFProvider(LLMProvider):
    def __init__(self, token: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.0, max_new_tokens: int = 256):
        import requests  # ensure installed
        self.requests = requests
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise RuntimeError("Missing HF_TOKEN (set it in .env)")
        self.model = model or os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
        self.api_url = os.getenv("HF_API_URL", f"https://api-inference.huggingface.co/models/{self.model}")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def chat(self, prompt: str) -> Tuple[str, float]:
        """
        Uses HF text-generation endpoint.
        NOTE: schema varies slightly across models; this payload works for most
        text-generation-inference compatible models on serverless.
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "return_full_text": False,
            }
        }
        t0 = time.time()
        r = self.requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # serverless returns a list[ { "generated_text": "..." } ]
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"]
        else:
            # fallback for other schemas
            text = str(data)
        return text.strip(), time.time() - t0


# ---------------------------
# Ollama provider (local OpenAI-compatible /v1)
# ---------------------------
class OllamaProvider(LLMProvider):
    """
    Calls a local Ollama server using the OpenAI-compatible /v1/chat/completions API.
    Env:
      OLLAMA_BASE_URL  (default: http://127.0.0.1:11434/v1)
      OLLAMA_MODEL     (default: llama3.1:8b)
      OLLAMA_TIMEOUT_S (optional, default: 120)
    """
    def __init__(self, base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 timeout_s: Optional[float] = None):
        import requests  # local dependency
        self.requests = requests
        self.base_url = (base_url or
                         os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/"))
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.temperature = float(temperature)
        self.timeout_s = float(os.getenv("OLLAMA_TIMEOUT_S", timeout_s or 120))

        # Quick sanity: ensure the server is up (best-effort)
        try:
            _ = self.requests.get(self.base_url, timeout=2)
        except Exception:
            # Don’t crash here — we’ll error on first call if it’s really down.
            pass

    def chat(self, prompt: str) -> Tuple[str, float]:
        """
        Single-turn, non-streaming call to /v1/chat/completions, returns (text, latency_s).
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": False,
        }
        t0 = time.time()
        r = self.requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        # OpenAI-style shape
        text = ""
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception:
            text = str(data)
        return (text or "").strip(), time.time() - t0

# ---------------------------
# Factory
# ---------------------------
def get_provider(provider: Optional[str] = None, temperature: float = 0.0, sleep_s: float = 0.05) -> LLMProvider:
    """
    provider: "gemini" | "hf" | "ollama" | "sim" (default from env PROVIDER)
    """
    name = (provider or os.getenv("PROVIDER", "sim")).lower()
    if name == "gemini":
        return GeminiProvider(temperature=temperature)
    if name == "hf":
        return HFProvider(temperature=temperature)
    if name == "ollama":
        return OllamaProvider(temperature=temperature)
    if name == "sim":
        return SimProvider(sleep_s=float(sleep_s))
    raise ValueError(f"Unknown provider '{name}'. Choose 'gemini' | 'hf' | 'sim'.")


