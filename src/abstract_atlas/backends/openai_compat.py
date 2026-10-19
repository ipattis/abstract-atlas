"""Local (or any OpenAI-compatible) inference with schema-constrained JSON output.

Works with anything that serves /v1/chat/completions and honours `response_format: json_schema`:
Ollama, llama.cpp's llama-server, vLLM, LM Studio, SGLang, or a hosted OpenAI-compatible API.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .base import Backend

THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_json(text):
    """Pull a JSON object out of a model reply (tolerates <think> blocks and ```json fences)."""
    text = THINK.sub("", text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


class OpenAICompatBackend(Backend):
    def __init__(self, base_url="http://localhost:11434/v1", model=None, api_key="not-needed", workers=4,
                 temperature=0.0, timeout=600, retries=3, **_):
        if not model:
            raise SystemExit("The local backend needs `model:` (the name your server knows, e.g. an Ollama tag)")
        self.base_url, self.model, self.workers = base_url.rstrip("/"), model, workers
        self.temperature, self.timeout, self.retries = temperature, timeout, retries
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.name = f"openai-compat:{model}"

    def _post(self, payload):
        last = None
        for attempt in range(self.retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers, timeout=self.timeout)
                if r.status_code >= 500 or r.status_code == 429:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, RuntimeError) as e:
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(repr(last)[:300])

    def _messages(self, system, user):
        return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]

    def _one(self, req):
        payload = {
            "model": self.model,
            "messages": self._messages(req.system, req.prompt),
            "temperature": self.temperature,
            "max_tokens": req.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": req.name, "schema": req.schema, "strict": True}},
        }
        resp = self._post(payload)
        usage = resp.get("usage") or {}
        return {
            "output": parse_json(resp["choices"][0]["message"]["content"]),
            "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
        }

    def run(self, requests_, job="job"):
        def safe(item):
            rid, req = item
            try:
                return rid, self._one(req)
            except Exception as e:
                return rid, {"error": repr(e)[:300]}

        with ThreadPoolExecutor(self.workers) as pool:
            return dict(pool.map(safe, requests_.items()))

    def chat_text(self, system, user, max_tokens=1024):
        resp = self._post({"model": self.model, "messages": self._messages(system, user), "temperature": self.temperature,
                           "max_tokens": max_tokens})
        return THINK.sub("", resp["choices"][0]["message"]["content"]).strip()
