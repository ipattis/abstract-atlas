"""The one interface the pipeline needs from an LLM: structured requests in, structured results out."""
from dataclasses import dataclass


@dataclass
class Request:
    prompt: str
    schema: dict            # JSON Schema of the object to return
    name: str = "record"    # tool / schema name
    description: str = "Record the analysis."
    system: str | None = None
    max_tokens: int = 1500


class Backend:
    """run() maps {record_id: Request} to {record_id: result}.

    A result is {"output": {...}, "usage": {"input_tokens": n, "output_tokens": n}} or {"error": "..."}.
    Failures are reported per record, never raised, so one bad record doesn't lose a run.
    """

    name = "backend"

    def run(self, requests: dict, job: str = "job") -> dict:
        raise NotImplementedError

    def chat_text(self, system: str | None, user: str, max_tokens: int = 1024) -> str:
        """Plain-text completion (used for cluster naming)."""
        raise NotImplementedError
