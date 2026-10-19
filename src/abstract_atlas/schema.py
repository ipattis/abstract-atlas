"""Extraction schema: a list of typed fields, rendered as a JSON Schema for any LLM backend."""
from dataclasses import dataclass, field

TYPES = {"text", "enum", "enum_list", "list", "int", "bool"}
# How the audit judges a field: free text is "faithful / minor_issue / unsupported", categorical labels are
# "correct / debatable / wrong". "skip" leaves the field out of the audit.
AUDIT_ROLES = {"text", "label", "skip"}


@dataclass
class Field:
    name: str
    type: str
    description: str = ""
    options: list = field(default_factory=list)
    required: bool = True
    audit: str = ""

    def __post_init__(self):
        if self.type not in TYPES:
            raise ValueError(f"field {self.name!r}: unknown type {self.type!r} (expected one of {sorted(TYPES)})")
        if self.type in {"enum", "enum_list"} and not self.options:
            raise ValueError(f"field {self.name!r}: {self.type} needs options")
        if not self.audit:
            self.audit = "text" if self.type == "text" else "skip"
        if self.audit not in AUDIT_ROLES:
            raise ValueError(f"field {self.name!r}: audit must be one of {sorted(AUDIT_ROLES)}")

    def json_schema(self):
        base = {
            "text": {"type": "string"},
            "enum": {"type": "string", "enum": self.options},
            "enum_list": {"type": "array", "items": {"type": "string", "enum": self.options}},
            "list": {"type": "array", "items": {"type": "string"}},
            "int": {"type": "integer"},
            "bool": {"type": "boolean"},
        }[self.type]
        return {**base, "description": self.description} if self.description else base


def fields_from_config(items):
    return [Field(**item) for item in items]


def object_schema(fields):
    """JSON Schema for one extraction record (also used as a tool's input_schema)."""
    return {
        "type": "object",
        "properties": {f.name: f.json_schema() for f in fields},
        "required": [f.name for f in fields if f.required],
        "additionalProperties": False,
    }


def subset(fields, names):
    by_name = {f.name: f for f in fields}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"unknown fields: {unknown}")
    return [by_name[n] for n in names]


NOT_STATED = {"", "none", "not stated", "n/a", "[]"}


def _as_list(v):
    """Models occasionally return a list field as a string ('USA, Canada', '[]')."""
    import json

    if isinstance(v, list):
        return v
    if not isinstance(v, str) or v.strip().lower() in NOT_STATED:
        return []
    try:
        parsed = json.loads(v)
        if isinstance(parsed, list):
            return parsed
    except ValueError:
        pass
    return [x.strip() for x in v.split(",") if x.strip()]


def validate_record(raw, fields):
    """Coerce and check one model output. Returns a clean dict or raises ValueError.

    Constrained decoding and tool calls make bad enums rare, not impossible, so this is the safety net: a bad
    record is dropped (and retried on the next run) rather than silently stored.
    """
    out = {}
    for f in fields:
        v = raw.get(f.name)
        if v is None:
            if f.required:
                raise ValueError(f"missing {f.name}")
            continue
        if f.type in {"list", "enum_list"}:
            v = [str(x) for x in _as_list(v)]
            if f.type == "enum_list":
                bad = [x for x in v if x not in f.options]
                if bad:
                    raise ValueError(f"{f.name}: {bad} not in options")
        elif f.type == "enum":
            if v not in f.options:
                raise ValueError(f"{f.name}: {v!r} not in options")
        elif f.type == "int":
            try:
                v = int(v)
            except (TypeError, ValueError):
                raise ValueError(f"{f.name}: not an integer") from None
        elif f.type == "bool":
            if isinstance(v, str):
                v = v.strip().lower() in {"true", "yes", "1"}
            v = bool(v)
        else:
            v = str(v).strip()
            if f.required and not v:
                raise ValueError(f"empty {f.name}")
        out[f.name] = v
    return out
