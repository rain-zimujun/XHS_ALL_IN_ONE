"""Split whole-brief title/copy production into one request per idea.

A 5-idea title request carried about 30k characters (22k prompt + 8k schema) and
sat at the relay's 524 timeout. One idea per request keeps every call small,
while the merged output still passes the original full-coverage validation.

The batch payload carries a private index hint so program-owned IDs stay in the
global order (``title_<idea>_<n>`` / ``copy_<title>_<n>``); the hint is dropped
by the role projection before the model sees the request.
"""
from copy import deepcopy

from backend.app.services.creative_agents.title_generation import content_fingerprint
from backend.app.services.creative_analysis import CreativeError


TITLE_BATCH_VERSION = "title-by-idea-v1"
COPY_BATCH_VERSION = "copy-by-title-v1"


def _fingerprint(batch: dict) -> str:
    return content_fingerprint({key: value for key, value in batch.items() if key != "input_fingerprint"})


def prepare_title_batches(payload: dict, revision: dict | None = None) -> list[dict]:
    claims = payload.get("approved_claims_by_idea") or {}
    saved_ideas = {idea["idea_id"]: idea
                   for idea in ((revision or {}).get("previous_output") or {}).get("ideas", [])}
    targets = set((revision or {}).get("target_ids") or [])
    batches = []
    for idea_index, idea in enumerate(payload["topic_brief"]["ideas"]):
        idea_id = idea["idea_id"]
        saved = saved_ideas.get(idea_id)
        idea_targets = [candidate["candidate_id"] for candidate in (saved or {}).get("candidates", [])
                        if candidate["candidate_id"] in targets]
        if revision and not idea_targets:
            continue
        batch = deepcopy(payload)
        batch["topic_brief"] = {**payload["topic_brief"], "ideas": [idea]}
        batch["requested_idea_count"] = 1
        batch["_idea_index"] = idea_index
        if claims:
            batch["approved_claims_by_idea"] = {idea_id: claims.get(idea_id, [])}
        batch["_revision"] = None
        if revision:
            batch["_revision"] = {
                "stage": "title", "target_ids": idea_targets,
                "previous_output": {"ideas": [saved], "limitations": revision["previous_output"].get("limitations", [])},
                "review_feedback": revision["review_feedback"], "rules": revision["rules"],
            }
            briefs = [brief for brief in revision.get("revision_briefs", [])
                      if brief.get("candidate_id") in set(idea_targets)]
            if briefs:
                batch["_revision"]["revision_briefs"] = briefs
        batch["input_fingerprint"] = _fingerprint(batch)
        batches.append(batch)
    if not batches:
        raise CreativeError("标题生成没有可处理的选题。", "title_input_invalid")
    return batches


def merge_title_batches(results, payload: dict, revision: dict | None = None):
    by_idea, limitations = {}, []
    for result in results:
        for idea in result.ideas:
            by_idea[idea.idea_id] = idea.model_dump(mode="json")
        limitations.extend(result.limitations)
    if revision:
        for idea in revision["previous_output"]["ideas"]:
            if idea["idea_id"] not in by_idea:
                by_idea[idea["idea_id"]] = deepcopy(idea)
        limitations = [*revision["previous_output"].get("limitations", []), *limitations]
    ideas = [by_idea[idea["idea_id"]] for idea in payload["topic_brief"]["ideas"] if idea["idea_id"] in by_idea]
    if len(ideas) != len(payload["topic_brief"]["ideas"]):
        raise CreativeError("标题生成批次合并缺少选题。", "title_input_invalid")
    from backend.app.schemas.creative_agents import TitleCandidates
    return TitleCandidates.model_validate({
        "topic_revision": payload["topic_revision"], "input_fingerprint": payload["input_fingerprint"],
        "ideas": ideas, "limitations": list(dict.fromkeys(limitations))[:8],
    })


def prepare_copy_batches(payload: dict, revision: dict | None = None) -> list[dict]:
    claims = payload.get("approved_claims_by_idea") or {}
    topics = {idea["idea_id"]: idea for idea in payload["topic_brief"]["ideas"]}
    previous = {idea["idea_id"]: idea
                for idea in ((revision or {}).get("previous_output") or {}).get("ideas", [])}
    targets = set((revision or {}).get("target_ids") or [])
    targets_by_idea = {
        idea_id: [variant["variant_id"] for variant in idea.get("variants", []) if variant["variant_id"] in targets]
        for idea_id, idea in previous.items()
    }
    batches = []
    for title_index, title in enumerate(payload["locked_titles"]):
        idea_id = title["idea_id"]
        if revision and not targets_by_idea.get(idea_id):
            continue
        batch = deepcopy(payload)
        batch["locked_titles"] = [title]
        batch["topic_brief"] = {**payload["topic_brief"], "ideas": [topics[idea_id]]}
        batch["requested_idea_count"] = 1
        batch["_title_index"] = title_index
        if claims:
            batch["approved_claims_by_idea"] = {idea_id: claims.get(idea_id, [])}
        batch["_revision"] = None
        if revision:
            batch["_revision"] = {
                "stage": "copy", "target_ids": targets_by_idea[idea_id],
                "previous_output": {"ideas": [previous[idea_id]],
                                    "limitations": revision["previous_output"].get("limitations", [])},
                "review_feedback": revision["review_feedback"], "rules": revision["rules"],
            }
        batch["input_fingerprint"] = _fingerprint(batch)
        batches.append(batch)
    if not batches:
        raise CreativeError("正文生成没有可处理的锁定标题。", "copy_input_invalid")
    return batches


def merge_copy_batches(results, payload: dict, revision: dict | None = None):
    by_idea, limitations = {}, []
    for result in results:
        for idea in result.ideas:
            by_idea[idea.idea_id] = idea.model_dump(mode="json")
        limitations.extend(result.limitations)
    if revision:
        for idea in revision["previous_output"]["ideas"]:
            if idea["idea_id"] not in by_idea:
                by_idea[idea["idea_id"]] = deepcopy(idea)
        limitations = [*revision["previous_output"].get("limitations", []), *limitations]
    ideas = [by_idea[title["idea_id"]] for title in payload["locked_titles"] if title["idea_id"] in by_idea]
    if len(ideas) != len(payload["locked_titles"]):
        raise CreativeError("正文生成批次合并缺少锁定标题。", "copy_input_invalid")
    from backend.app.schemas.creative_agents import CopyVariants
    return CopyVariants.model_validate({
        "input_fingerprint": payload["input_fingerprint"],
        "ideas": ideas, "limitations": list(dict.fromkeys(limitations))[:8],
    })
