"""Role-specific model inputs; full artifacts remain in graph state for checks."""
from copy import deepcopy


def _finding(pattern):
    return {**{k: pattern[k] for k in ("dimension", "finding") if k in pattern},
            "note_ids": list(dict.fromkeys(e["note_id"] for e in pattern["evidence"]))}


def _audience(brief, segment_ids=None, *, include_shared=False):
    fields = ("segment_id", "audience_terms", "role_context", "primary_scenario", "attention_points",
              "value_opportunities", "suitable_formats", "uncertainties")
    result = {"summary": brief["summary"], "segments": [
        {key: s[key] for key in fields} for s in brief["segments"]
        if segment_ids is None or s["segment_id"] in segment_ids
    ], "avoid_assumptions": brief["avoid_assumptions"], "limitations": brief["limitations"]}
    if include_shared:
        result["shared_needs"] = [{k: v for k, v in n.items() if k != "evidence"} for n in brief["shared_needs"]]
        result["format_recommendations"] = brief["format_recommendations"]
    return result


def production_context(role, payload):
    """An allowlist prevents newly added state fields leaking into prompts."""
    common = ("keyword", "requirements", "requested_idea_count", "sample_note_ids")
    result = {key: payload[key] for key in common}
    sample = payload["sample_analysis"]
    if role == "topic":
        result["audience_segment_ids"] = payload["audience_segment_ids"]
        result["sample_analysis"] = {
            "summary": sample["summary"],
            "observations": [{k: v for k, v in o.items() if k != "quote"} for o in sample["observations"]],
            **{field: [_finding(p) for p in sample[field]] for field in ("topic_patterns", "value_patterns", "avoid_patterns")},
            "limitations": sample["limitations"],
        }
        result["audience_brief"] = _audience(payload["audience_brief"], include_shared=True)
    else:
        ideas = payload["topic_brief"]["ideas"]
        note_ids = {n for idea in ideas for n in idea["source_note_ids"]}
        result["sample_note_ids"] = [n for n in payload["sample_note_ids"] if n in note_ids]
        result["topic_brief"] = {"ideas": [{k: v for k, v in i.items() if k != "evidence"} for i in ideas],
                                 "selection_limits": payload["topic_brief"]["selection_limits"]}
        result["audience_brief"] = _audience(payload["audience_brief"], {i["primary_segment_id"] for i in ideas})
        patterns = [p for p in sample["style_patterns"]
                    if role == "copy" or p["dimension"] == "title"]
        result["sample_analysis"] = {"style_summary": sample["style_summary"],
                                     "style_patterns": [_finding(p) for p in patterns],
                                     "avoid_patterns": [_finding(p) for p in sample["avoid_patterns"]],
                                     "limitations": sample["limitations"]}
        for field in ("style_patterns", "avoid_patterns"):
            for finding in result["sample_analysis"][field]:
                finding["note_ids"] = [n for n in finding["note_ids"] if n in note_ids]
        if role == "title":
            result.update({key: payload[key] for key in ("topic_revision", "candidate_count", "title_max_length")})
        elif role == "copy":
            result["variant_count"] = payload["variant_count"]
            result["locked_titles"] = [{k: v for k, v in t.items() if k != "rationale"} for t in payload["locked_titles"]]
        else:
            raise ValueError("unknown production role")
    # Canonical bindings stay available to the adapter and local validators.
    # Include fingerprint as data for compatibility, never ask the model to copy it.
    for key in ("input_fingerprint", "prompt_version"):
        if key in payload:
            result[key] = payload[key]
    return deepcopy(result)
