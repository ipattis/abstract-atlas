"""Offline stand-in that returns schema-valid placeholder output. For tests and dry runs; never for real results."""
import json
import re

from .base import Backend


def fake_value(schema, seed):
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][seed % len(schema["enum"])]
    if t == "string":
        return f"placeholder text {seed}"
    if t == "integer":
        return seed % 50
    if t == "boolean":
        return bool(seed % 2)
    if t == "array":
        item = schema.get("items", {})
        return [fake_value(item, seed)] if "enum" in item else ["placeholder"]
    if t == "object":
        return {k: fake_value(v, seed) for k, v in schema.get("properties", {}).items()}
    return None


class MockBackend(Backend):
    name = "mock"

    def __init__(self, **_):
        pass

    def run(self, requests, job="job"):
        out = {}
        for i, (rid, req) in enumerate(requests.items()):
            out[rid] = {"output": fake_value(req.schema, sum(map(ord, rid)) + i), "usage": {"input_tokens": 0, "output_tokens": 0}}
        return out

    def chat_text(self, system, user, max_tokens=1024):
        # Good enough for Toponymy's JSON-ish naming prompts in smoke tests.
        n = len(re.findall(r"\w+", user)) % 97
        return json.dumps({"topic_name": f"Mock topic {n}", "topic_specificity": 0.5})
