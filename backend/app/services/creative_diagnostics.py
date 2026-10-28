"""Bounded diagnostics shared by creative workflow runners."""
import json
import os
import re
import traceback
import unicodedata
from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from difflib import SequenceMatcher

from loguru import logger
from pydantic import ValidationError


_context = ContextVar("creative_diagnostics", default={})
_raw_quotes = ContextVar("creative_raw_quotes", default=None)
_raw_model_response = ContextVar("creative_raw_model_response", default=None)
MAX_LOGGED_MODEL_RESPONSE_CHARS = 8000


@contextmanager
def capture_model_quotes():
    quotes = {}
    token = _raw_quotes.set(quotes)
    try:
        yield quotes
    finally:
        _raw_quotes.reset(token)


@contextmanager
def capture_model_response():
    response = {}
    token = _raw_model_response.set(response)
    try:
        yield response
    finally:
        _raw_model_response.reset(token)


def record_model_response(output):
    response = _raw_model_response.get()
    if response is not None and isinstance(output, str):
        response["text"] = output


def record_model_quotes(output):
    """Capture only known quote fields before TypedDict strips whitespace."""
    quotes = _raw_quotes.get()
    if quotes is None or not isinstance(output, dict):
        return

    def store(path, item):
        if isinstance(item, dict) and isinstance(item.get("quote"), str):
            text = item["quote"]
            quotes[tuple(path) + ("quote",)] = (text[:600], len(text))

    observations = output.get("observations")
    if isinstance(observations, list):
        for index, item in enumerate(observations[:50]):
            store(("observations", index), item)
    for field in ("topic_patterns", "style_patterns", "value_patterns", "avoid_patterns"):
        findings = output.get(field)
        if isinstance(findings, list):
            for index, finding in enumerate(findings[:8]):
                evidence = finding.get("evidence") if isinstance(finding, dict) else None
                if isinstance(evidence, list):
                    for evidence_index, item in enumerate(evidence[:8]):
                        store((field, index, "evidence", evidence_index), item)


@contextmanager
def diagnostic_context(*, task_id, workflow):
    token = _context.set({"task_id": task_id, "workflow": workflow})
    try:
        yield
    finally:
        _context.reset(token)


def diagnostic_event(event, *, level="INFO", **details):
    # Callers may include a bounded failed model response, never prompts,
    # request kwargs or credentials.
    record = {**_context.get(), "event": event, **details}
    logger.log(level, "创作诊断 {}", json.dumps(record, ensure_ascii=False))


def failed_model_response(response):
    """Return bounded raw provider text for a failed structured response."""
    raw = (response or {}).get("text")
    if not isinstance(raw, str):
        return None
    return {
        "model_response": raw[:MAX_LOGGED_MODEL_RESPONSE_CHARS],
        "model_response_chars": len(raw),
        "model_response_truncated": len(raw) > MAX_LOGGED_MODEL_RESPONSE_CHARS,
    }


def quote_mismatch_details(*, path, quote, source, source_position, field=None):
    """Describe literal mismatches without relaxing acceptance or guessing intent.

    Each source window is at most 360 characters; body segments stay separate.
    Similarity selects a comparison window, not a proven intended quotation.
    """
    candidates = [("title", None, source["title"])] + [
        ("body", index, text) for index, text in enumerate(source["body_segments"])
    ]
    compact = lambda text: "".join(text.split())
    normalized = lambda text: compact(unicodedata.normalize("NFKC", text))
    comparisons = []
    for source_field, segment_index, text in candidates:
        matcher = SequenceMatcher(None, quote, text, autojunk=False)
        anchor = max(matcher.get_matching_blocks(), key=lambda block: block.size)
        start = min(max(0, anchor.b - anchor.a - 40), max(0, len(text) - 360))
        end = min(len(text), start + 360)
        window = text[start:end]
        ratio = SequenceMatcher(None, quote, window, autojunk=False).ratio()
        comparisons.append({
            "field": source_field, "segment_index": segment_index, "allowed": field is None or field == source_field,
            "source_chars": len(text), "window_start": start, "window_end": end, "source_window": window,
            "exact_match": quote in text, "whitespace_match": compact(quote) in compact(text),
            "nfkc_whitespace_match": normalized(quote) in normalized(text), "similarity": round(ratio, 4),
        })
    allowed = [candidate for candidate in comparisons if candidate["allowed"]]
    best = max(allowed, key=lambda candidate: candidate["similarity"])
    differences = []
    for tag, a, b, c, d in SequenceMatcher(None, quote, best["source_window"], autojunk=False).get_opcodes():
        if tag != "equal":
            differences.append({
                "operation": tag, "model_start": a, "model_end": b,
                "source_start": best["window_start"] + c, "source_end": best["window_start"] + d,
                "model_text": quote[a:b][:80], "source_text": best["source_window"][c:d][:80],
                "text_truncated": b - a > 80 or d - c > 80,
            })
    return {
        "quote_path": list(path), "source_position": source_position, "expected_field": field or "title_or_body",
        "validated_quote": quote, "source_candidates": comparisons,
        "closest_candidate_index": comparisons.index(best), "differences": differences[:12],
        "differences_truncated": len(differences) > 12,
    }


def failed_quote_diagnostic(exc, output, raw_quotes=None):
    """Recover the original parsed model string before Pydantic whitespace trim."""
    detail = getattr(exc, "quote_diagnostic", None)
    if detail is None:
        return None
    detail = deepcopy(detail)
    raw_quote = output
    try:
        for key in detail["quote_path"]:
            raw_quote = raw_quote[key]
    except (KeyError, IndexError, TypeError):
        raw_quote = None
    captured = (raw_quotes or {}).get(tuple(detail["quote_path"]))
    if captured is not None or isinstance(raw_quote, str):
        raw_quote, raw_chars = captured if captured is not None else (raw_quote[:600], len(raw_quote))
        detail.update(model_quote=raw_quote, model_quote_chars=raw_chars,
                      model_quote_truncated=raw_chars > 600,
                      model_quote_changed_by_validation=raw_quote != detail["validated_quote"])
    else:
        detail["model_quote"] = None
    return detail


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
    frames = traceback.extract_tb(exc.__traceback__)[-8:]
    details = {
        "error_code": "creative_failed",
        "exception_type": type(exc).__name__,
        "traceback_frames": [
            {"file": os.path.basename(frame.filename), "function": frame.name, "line": frame.lineno}
            for frame in frames
        ],
    }
    if isinstance(exc, TypeError) and exc.args and isinstance(exc.args[0], str):
        message = exc.args[0]
        duplicate = re.search(
            r"multiple values for (?:keyword )?argument\s+(['\"])([A-Za-z_][A-Za-z0-9_]*)\1", message,
        )
        unexpected = re.search(
            r"unexpected keyword argument\s+(['\"])([A-Za-z_][A-Za-z0-9_]*)\1", message,
        )
        if duplicate:
            details["type_error_kind"] = "duplicate_keyword"
            details["argument_name"] = duplicate.group(2)
        elif unexpected:
            details["type_error_kind"] = "unexpected_keyword"
            details["argument_name"] = unexpected.group(2)
    return details
