"""Rule-derived labels: the LLM extracts facts, a published rule turns them into a category.

    derived:
      evidence_tier:
        default: Unclear
        rules:
          - if: {study_type: {startswith: [Randomized, Quasi]}, outcomes: {contains: Learning}}
            then: Causal evidence on learning

Rules are checked top to bottom and the first match wins. Because the rule is plain data, it can be read, argued
with and changed, and every paper is re-labelled for free (no new LLM calls).
"""


def _startswith(v, arg):
    return isinstance(v, str) and v.startswith(tuple(arg) if isinstance(arg, list) else arg)


def _contains(v, arg):
    if isinstance(v, (list, tuple, set)):
        return arg in v
    return isinstance(v, str) and isinstance(arg, str) and arg in v


OPS = {
    "equals": lambda v, a: v == a,
    "in": lambda v, a: v in a,
    "startswith": _startswith,
    "contains": _contains,
    "contains_any": lambda v, a: any(_contains(v, x) for x in a),
    "gte": lambda v, a: v is not None and v >= a,
    "lte": lambda v, a: v is not None and v <= a,
    "not_empty": lambda v, a: bool(v) == bool(a),
}


def matches(cond, rec):
    for name, test in cond.items():
        if not isinstance(test, dict):
            test = {"equals": test}
        for op, arg in test.items():
            if op not in OPS:
                raise ValueError(f"unknown rule operator {op!r} (expected one of {sorted(OPS)})")
            if not OPS[op](rec.get(name), arg):
                return False
    return True


def derive(rec, spec):
    for rule in spec.get("rules", []):
        if matches(rule["if"], rec):
            return rule["then"]
    return spec.get("default", "Unclear")


def derive_all(rec, derived):
    return {name: derive(rec, spec) for name, spec in (derived or {}).items()}
