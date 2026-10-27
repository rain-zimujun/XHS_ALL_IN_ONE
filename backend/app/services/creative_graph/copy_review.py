"""Review all variants of one locked title per bounded, resumable request."""
from copy import deepcopy
from dataclasses import replace

from backend.app.services.creative_agents.copy_review import COPY_REVIEW_AGENT as BASE_AGENT
from backend.app.services.creative_agents.title_generation import content_fingerprint
from backend.app.services.creative_graph.title_review import merge_title_review_results
from backend.app.services.creative_graph.adaptive_batches import downstream_batch_policy
from backend.app.services.creative_analysis import CreativeError


COPY_REVIEW_VERSION = "copy-review-adaptive-v3"
COPY_REVIEW_AGENT = replace(BASE_AGENT, system_prompt=BASE_AGENT.system_prompt + """
当前只评一个锁定标题下的当前正文版本；若 context 提供 peer_summaries，只能将其作为简短对照资料。
context 仅提供当前选题、锁定标题、主要受众和相关样本证据，不需要其他选题的信息。
按 response_format 返回；每项评分理由用简短一句话，问题与修改指令具体简洁。
sample_analysis 是文字分析和引句依据，不是原图或完整市场调研，不据此补造事实。
若 context 提供 approved_claims 且 item 提供 evidence，逐版对照其 claim/source 绑定判断正文承诺；来源 ID 合法不等于正文语义已得到支持，不能只凭候选自报通过。
related_item_ids 仅限本批其他版本；不得引用别的标题下的正文。
版本号和输入指纹由程序绑定，无须输出。""")


def prepare_copy_review_batches(payload: dict, *, sample_batches: int = 1) -> list[dict]:
    context = payload["context"]
    batches, covered = [], []
    for title in context["locked_titles"]:
        idea = next(i for i in context["topic_brief"]["ideas"] if i["idea_id"] == title["idea_id"])
        segment = next(s for s in context["audience_brief"]["segments"] if s["segment_id"] == idea["primary_segment_id"])
        items = [item for item in payload["items"] if item["idea_id"] == title["idea_id"] and item["candidate_id"] == title["candidate_id"]]
        if len(items) not in (2, 3):
            raise CreativeError("正文评审批次必须包含同一锁定标题下的 2～3 个版本。", "review_input_invalid")
        notes = [note for note in payload["sample_note_ids"] if note in idea["source_note_ids"]]
        sample = context["sample_analysis"]
        compact_sample = {key: sample[key] for key in ("style_summary", "limitations")}
        compact_sample["observations"] = [o for o in sample["observations"] if o["note_id"] in notes]
        for field in ("style_patterns", "value_patterns", "avoid_patterns"):
            compact_sample[field] = []
            for pattern in sample[field]:
                evidence = [e for e in pattern["evidence"] if e["note_id"] in notes]
                if evidence:
                    compact_sample[field].append({**pattern, "evidence": evidence})
        compact = {
            "keyword": context["keyword"], "requirements": context["requirements"],
            "variant_count": context["variant_count"],
            "locked_titles": [title], "topic_brief": {"ideas": [idea]},
            "approved_claims": context.get("approved_claims_by_idea", {}).get(title["idea_id"], []),
            "audience": {key: segment[key] for key in (
                "audience_terms", "role_context", "primary_scenario", "attention_points",
            )},
            "sample_analysis": compact_sample,
        }
        if sample_batches > 1:
            for current in items:
                peers = [
                    {"label": item["fields"].get("label", ""), "opening": item["fields"].get("body", "")[:120]}
                    for item in items if item is not current
                ][:2]
                current_context = {**compact, "peer_summaries": peers}
                review_item = _with_evidence(current)
                batches.append({
                    "kind": "copy", "revision": payload["revision"], "context": current_context,
                    "items": [review_item], "sample_note_ids": notes,
                    "rubric": payload["rubric"], "pass_threshold": payload["pass_threshold"],
                    "policy_version": COPY_REVIEW_VERSION,
                    "batch_policy": downstream_batch_policy(sample_batches, default_size=len(items)),
                })
        else:
            batches.append({
                "kind": "copy", "revision": payload["revision"], "context": compact,
                "items": [_with_evidence(item) for item in items], "sample_note_ids": notes,
                "rubric": payload["rubric"], "pass_threshold": payload["pass_threshold"],
                "policy_version": COPY_REVIEW_VERSION,
                "batch_policy": downstream_batch_policy(sample_batches, default_size=len(items)),
            })
        covered.extend(item["item_id"] for item in items)
    if not batches or len(covered) != len(set(covered)) or set(covered) != {i["item_id"] for i in payload["items"]}:
        raise CreativeError("正文评审批次没有完整且唯一地覆盖所有版本。", "review_input_invalid")
    for index, batch in enumerate(batches, 1):
        batch["batch"] = {"index": index, "count": len(batches)}
        batch["input_fingerprint"] = content_fingerprint(batch)
    return deepcopy(batches)


def _with_evidence(item: dict) -> dict:
    fields = item["fields"]
    return {
        **{key: value for key, value in item.items() if key != "fields"},
        "fields": {key: value for key, value in fields.items() if key not in {"claim_ids", "source_note_ids"}},
        "evidence": {"claim_ids": fields.get("claim_ids", []), "source_note_ids": fields.get("source_note_ids", [])},
    }


def merge_copy_review_results(results, payload):
    # The same deterministic merge verifies coverage, quote/score constraints
    # and the canonical fingerprint; it does not change scores or thresholds.
    return merge_title_review_results(results, payload)
