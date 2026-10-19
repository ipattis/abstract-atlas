"""LLM backends. `make_backend` turns a config entry into one."""
from .base import Backend, Request  # noqa: F401


def make_backend(spec, state_dir):
    """spec: {"type": "local" | "bedrock" | "bedrock-batch" | "anthropic" | "mock", ...options}."""
    kind = spec["type"]
    opts = {k: v for k, v in spec.items() if k not in {"type", "name"}}
    if kind == "mock":
        from .mock import MockBackend

        return MockBackend()
    if kind in {"local", "openai"}:
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(**opts)
    from . import claude

    if kind == "anthropic":
        return claude.ClaudeBackend(provider="anthropic", state_dir=state_dir, **opts)
    if kind == "bedrock":
        return claude.ClaudeBackend(provider="bedrock", state_dir=state_dir, **opts)
    if kind == "bedrock-batch":
        return claude.BedrockBatchBackend(state_dir=state_dir, **opts)
    raise SystemExit(f"Unknown backend type {kind!r} (local, bedrock, bedrock-batch, anthropic, mock)")
