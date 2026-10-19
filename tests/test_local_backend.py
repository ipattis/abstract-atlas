"""The local backend against a tiny fake OpenAI-compatible server (no model needed)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from abstract_atlas.backends import Request
from abstract_atlas.backends.openai_compat import OpenAICompatBackend

SEEN = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        content = '<think>x</think>{"summary": "ok"}' if "response_format" in body else "Mock Topic"
        out = json.dumps({"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_structured_request_and_text(server):
    b = OpenAICompatBackend(base_url=server, model="m", workers=2)
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}
    res = b.run({"a": Request("hi", schema, system="sys"), "b": Request("yo", schema)})
    assert res["a"]["output"] == {"summary": "ok"} and res["a"]["usage"] == {"input_tokens": 5, "output_tokens": 2}
    sent = next(s for s in SEEN if s["messages"][-1]["content"] == "hi")
    assert sent["response_format"]["json_schema"]["schema"] == schema and sent["messages"][0]["role"] == "system"
    assert b.chat_text(None, "name this") == "Mock Topic"


def test_failure_is_reported_not_raised():
    b = OpenAICompatBackend(base_url="http://127.0.0.1:1/v1", model="m", retries=1)
    res = b.run({"a": Request("hi", {"type": "object"})})
    assert "error" in res["a"]
