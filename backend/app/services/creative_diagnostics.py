"""Small, content-free diagnostics shared by creative workflow runners."""
import json
from contextlib import contextmanager
from contextvars import ContextVar

from loguru import logger
from pydantic import ValidationError


_context = ContextVar("creative_diagnostics", default={})


@contextmanager
def diagnostic_context(*, task_id, workflow):
    token = _context.set({"task_id": task_id, "workflow": workflow})
    try:
        yield
    finally:
        _context.reset(token)


def diagnostic_event(event, *, level="INFO", **details):
    # Callers supply controlled summaries/counts only, never prompts, raw
    # responses, request kwargs, exception repr/tracebacks or credentials.
    record = {**_context.get(), "event": event, **details}
    logger.log(level, "创作诊断 {}", json.dumps(record, ensure_ascii=False))


def validation_summary(exc: ValidationError, schema: dict) -> str:
    """Locate fields without echoing unknown keys, input values or validator ctx."""
    known_fields = set()

    def collect(value):
        if isinstance(value, dict):
            known_fields.update(value.get("properties", {}))
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(schema)
    issues = exc.errors(include_input=False, include_context=False, include_url=False)
    messages = []
    for issue in issues[:8]:
        path = ".".join(str(p) if type(p) is int or p in known_fields else "[未知字段]" for p in issue["loc"])
        messages.append(f"{path or 'output'}: {issue['type']}")
    return "; ".join(messages)[:500]


def exception_details(exc):
    # Arbitrary transport exceptions often include URLs, headers and bodies.
    # Only CreativeError is explicitly restricted to user-safe messages.
    from backend.app.services.creative_analysis import CreativeError

    if isinstance(exc, CreativeError):
        return {"error_code": exc.code, "reason": str(exc)[:500]}
    return {"error_code": "creative_failed", "exception_type": type(exc).__name__}
