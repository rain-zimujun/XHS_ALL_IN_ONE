"""Build the Chat Completions JSON Schema parameter from local output models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def json_schema_response_format(output_model: type[BaseModel], *, max_body_chars: int | None = None) -> dict[str, Any]:
    schema = output_model.model_json_schema()

    def make_strict(node: dict[str, Any]) -> None:
        # Strict output requires every property, including local defaulted
        # fields. Nullable unions stay nullable; empty lists stay valid where
        # allowed by the original model. Omit local defaults in the API schema.
        node.pop("default", None)
        if node.get("type") == "object":
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties", {}))
            body = node.get("properties", {}).get("body")
            if max_body_chars is not None and isinstance(body, dict) and body.get("type") == "string":
                body["maxLength"] = min(body.get("maxLength", max_body_chars), max_body_chars)
        for key in ("$defs", "properties"):
            for child in node.get(key, {}).values():
                make_strict(child)
        if isinstance(node.get("items"), dict):
            make_strict(node["items"])
        for key in ("anyOf", "oneOf", "allOf"):
            for child in node.get(key, []):
                make_strict(child)

    make_strict(schema)
    return {
        "type": "json_schema",
        "json_schema": {"name": output_model.__name__, "strict": True, "schema": schema},
    }
