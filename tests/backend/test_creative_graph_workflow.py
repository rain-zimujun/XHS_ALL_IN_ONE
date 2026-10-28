import copy
import json
from threading import Event, Lock
from time import sleep
from types import SimpleNamespace

import pytest
import requests

from backend.app.services.creative_analysis import CreativeError
from backend.app.services.creative_agents.revision import PipelineCancelled
from backend.app.services.creative_graph.state import SampleUpdate
from backend.app.services.creative_graph.recovery import prepare_resume
from backend.app.services.creative_graph.workflow import LangGraphWorkflow, estimate_graph_request_budget, title_revision_briefs
from backend.app.services.creative_pipeline import validate_pipeline_state
from test_creative_sample_agent import SEARCH, sources
from test_creative_workflow import WorkflowCompletion


class GraphCompletion(WorkflowCompletion):
    """Use the existing independent synthetic outputs with the new wire schema."""
    NAMES = {
        "SampleAnalysisResponse": "SampleAnalysis", "SampleReviewResponse": "SampleReviewResponse",
        "AudienceGroupingResponse": "AudienceBrief", "AudienceNeedsResponse": "AudienceBrief",
        "TopicResponse": "TopicBrief", "TopicReviewResponse": "TopicReviewResponse",
        "TitleResponse": "TitleCandidates", "CopyResponse": "CopyVariants", "ImageResponse": "ImagePlans",
        "TitleReviewResponse": "ItemReviewResponse", "CopyReviewResponse": "ItemReviewResponse", "FinalReviewResponse": "ItemReviewResponse",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wire_calls = []

    def __call__(self, **kwargs):
        self.wire_calls.append(copy.deepcopy(kwargs))
        prompt, end = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
        wire_name = kwargs["response_format"]["json_schema"]["name"]
        if wire_name == "TopicResponse" and "quality_revision" in prompt:
            ideas = []
            for idea in prompt["quality_revision"]["previous_target_ideas"]:
                ideas.append({key: value for key, value in idea.items()
                              if key not in {"idea_id", "source_note_ids"}})
            return json.dumps({"summary": "返工测试", "ideas": ideas, "selection_limits": ["测试限制"]}, ensure_ascii=False)
        if wire_name == "SampleReviewResponse":
            role = "sample_review"
            self.calls.append((role, prompt, kwargs))
            self.counts[role] += 1
            if role == self.timeout_stage:
                raise requests.ReadTimeout("secret-transport-details")
            output = {"decisions": [
                {"claim_id": claim["claim_id"], "supported": True,
                 "note_ids": claim["candidate_note_ids"][:1], "reason": "来源内容支持该归纳。"}
                for claim in prompt["input"]["claims"]
            ]}
            if self.fail_stage == role and (self.always or self.counts[role] == 1):
                output["decisions"].pop()
            return json.dumps(output, ensure_ascii=False)
        prompt.setdefault("output_schema", copy.deepcopy(kwargs["response_format"]["json_schema"]["schema"]))
        prompt["output_schema"]["title"] = self.NAMES[prompt["output_schema"]["title"]]
        raw = super().__call__(**{**kwargs, "user_prompt": json.dumps(prompt, ensure_ascii=False) + kwargs["user_prompt"][end:]})
        output = json.loads(raw)
        if wire_name == "AudienceGroupingResponse":
            output = {"segments": [{k: v for k, v in group.items() if k != "segment_id"} for group in output["segments"]]}
        elif wire_name == "AudienceNeedsResponse":
            mapping = {group["segment_id"]: prompt["input"]["segments"][i]["segment_id"] for i, group in enumerate(output["segments"])}
            for item in output["shared_needs"]:
                item["applies_to_segment_ids"] = [mapping[ref] for ref in item["applies_to_segment_ids"]]
            for item in output["format_recommendations"]:
                item["for_segment_ids"] = [mapping[ref] for ref in item["for_segment_ids"]]
            for key in ("segments", "requested_audience_terms"):
                output.pop(key)
        if wire_name in {"AudienceGroupingResponse", "AudienceNeedsResponse"}:
            claims = prompt["input"].get("approved_claims", [])
            claim_for_note = {note_id: claim["claim_id"] for claim in claims for note_id in claim["note_ids"]}
            collections = [output.get("segments", []), output.get("shared_needs", [])]
            for group in collections[0]:
                for evidence in group.get("evidence", []):
                    evidence["claim_ids"] = list(dict.fromkeys(
                        claim_for_note[note_id] for note_id in evidence["note_ids"] if note_id in claim_for_note
                    )) if evidence["basis"] == "sample_evidence" else []
            for need in collections[1]:
                for evidence in need.get("evidence", []):
                    evidence["claim_ids"] = list(dict.fromkeys(
                        claim_for_note[note_id] for note_id in evidence["note_ids"] if note_id in claim_for_note
                    )) if evidence["basis"] == "sample_evidence" else []
        elif wire_name == "TopicResponse":
            claims = prompt["input"].get("approved_claims", [])
            claim_for_note = {note_id: claim["claim_id"] for claim in claims for note_id in claim["note_ids"]}
            for idea in output["ideas"]:
                idea.pop("idea_id", None)
                idea.pop("source_note_ids", None)
                idea["primary_segment_id"] = prompt["input"]["audience_brief"]["segments"][0]["segment_id"]
                for evidence in idea["evidence"]:
                    evidence["claim_ids"] = list(dict.fromkeys(
                        claim_for_note[note_id] for note_id in evidence["note_ids"] if note_id in claim_for_note
                    )) if evidence["basis"] != "user_requirement" else []
        elif wire_name == "TitleResponse":
            claims_by_idea = {
                idea["idea_id"]: idea.get("approved_claims", [])
                for idea in prompt["input"]["topic_brief"]["ideas"]
            }
            output = {
                "ideas": {
                    idea["idea_id"]: {
                        "candidates": [
                            {
                                **{key: value for key, value in candidate.items() if key != "candidate_id"},
                                "claim_ids": [claim["claim_id"] for claim in claims_by_idea[idea["idea_id"]][:1]],
                                "source_note_ids": claims_by_idea[idea["idea_id"]][0]["note_ids"][:1]
                                if claims_by_idea[idea["idea_id"]] else [],
                            }
                            for candidate in idea["candidates"]
                        ]
                    }
                    for idea in output["ideas"]
                },
                "limitations": output["limitations"],
            }
        elif wire_name == "CopyResponse":
            claims_by_idea = prompt["input"].get("approved_claims_by_idea", {})
            for idea in output["ideas"]:
                claims = claims_by_idea.get(idea["idea_id"], [])
                for variant in idea["variants"]:
                    variant["claim_ids"] = [claim["claim_id"] for claim in claims[:2]]
                    variant["source_note_ids"] = list(dict.fromkeys(
                        note_id for claim in claims[:2] for note_id in claim["note_ids"]
                    ))[:2] or prompt["input"]["sample_note_ids"][:2]
        for key in ("input_fingerprint", "revision", "topic_revision"):
            output.pop(key, None)
        for review in output.get("reviews", []):
            if "item_id" in review:
                for issue in review["issues"]:
                    issue.setdefault("responsible_stage", None)
                if wire_name == "TitleReviewResponse":
                    review["comment"] = review["scores"][0]["comment"]
                    review["scores"] = {score["criterion"]: score["score"] for score in review["scores"]}
                    for issue in review["issues"]:
                        issue.pop("responsible_stage")
                        issue.pop("note_ids")
        return json.dumps(output, ensure_ascii=False)


def workflow(model, count=1, variants=2, **kwargs):
    return LangGraphWorkflow(search=SEARCH, sources=sources(), options={"model_id": 1, "idea_count": count, "variant_count": variants}, complete=model, **kwargs)


@pytest.mark.parametrize("source_count,ideas,variants,minimum", [
    (3, 3, 3, 72), (22, 3, 3, 200), (50, 5, 3, 350),
])
def test_request_budget_scales_with_sample_and_output_fanout(source_count, ideas, variants, minimum):
    budget = estimate_graph_request_budget(source_count, {"idea_count": ideas, "variant_count": variants})
    assert budget >= minimum
    assert budget <= 512
    assert budget == estimate_graph_request_budget(source_count, {"idea_count": ideas, "variant_count": variants})


@pytest.mark.parametrize("count", [1, 3, 5])
@pytest.mark.parametrize("variants", [2, 3])
def test_graph_executes_ten_roles_and_persists_serializable_global_state(count, variants):
    model, saved = GraphCompletion(), []
    w = workflow(model, count, variants, on_graph_state=saved.append)
    result, metadata = w.run()
    assert len(result["ideas"]) == count
    assert all("claim_ids" not in idea for idea in result["ideas"])
    assert all("claim_ids" not in variant for idea in result["ideas"] for variant in idea["variants"])
    final_calls = [call for call in model.wire_calls if call["response_format"]["json_schema"]["name"] == "FinalReviewResponse"]
    assert len(final_calls) == count * variants
    assert all(json.loads(call["user_prompt"])["input"]["items"][0]["evidence"]["approved_claims"]
               for call in final_calls)
    assert all(len(i["variants"]) == variants for i in result["ideas"])
    assert metadata["workflow"] == "langgraph"
    assert len(model.calls) == metadata["request_count"] == 8 + 2 * count + 2 * count * variants
    assert w.state["sample_review"]["status"] == "passed"
    approved_claims = w.state["sample_review"]["approved_claims"]
    assert len(approved_claims) == w.state["sample_review"]["claim_count"] - w.state["sample_review"]["rejected_claim_count"]
    assert all(claim["claim_id"].startswith("claim_001_") for claim in approved_claims)
    assert all(set(claim["note_ids"]) <= {source["note_id"] for source in sources()}
               for claim in approved_claims)
    assert w.state["pipeline"]["stages"]["sample_analysis"]["metadata"]["quality_review_status"] == "passed"
    assert saved[-1]["result"] == result
    assert saved[-1]["pipeline"]["status"] == "completed"
    assert saved[-1]["revision_plan"] == {}
    for state in saved:
        validate_pipeline_state(json.loads(json.dumps(state))["pipeline"])
    assert {c["response_format"]["json_schema"]["name"] for c in model.wire_calls} == set(GraphCompletion.NAMES)
    assert "fixture-private" not in json.dumps(w.state)


def test_topic_response_schema_and_adapter_prevent_duplicate_source_references():
    class DuplicateTopicSources(GraphCompletion):
        def __call__(self, **kwargs):
            raw = super().__call__(**kwargs)
            if kwargs["response_format"]["json_schema"]["name"] == "TopicResponse":
                output = json.loads(raw)
                idea = output["ideas"][0]
                idea["source_note_ids"].append(idea["source_note_ids"][0])
                for evidence in idea["evidence"]:
                    if evidence["note_ids"]:
                        evidence["note_ids"].append(evidence["note_ids"][0])
                        if evidence["claim_ids"]:
                            evidence["claim_ids"].append(evidence["claim_ids"][0])
                return json.dumps(output, ensure_ascii=False)
            return raw

    model = DuplicateTopicSources()
    result = workflow(model, count=1, variants=2).run()[0]

    call = next(item for item in model.wire_calls if item["response_format"]["json_schema"]["name"] == "TopicResponse")
    schema = call["response_format"]["json_schema"]["schema"]
    evidence = schema["$defs"]["BoundTopicEvidence"]["properties"]
    assert evidence["note_ids"]["uniqueItems"] is True
    assert evidence["claim_ids"]["uniqueItems"] is True
    assert schema["$defs"]["BoundTopicIdea"]["properties"]["source_note_ids"]["uniqueItems"] is True
    assert len(result["ideas"]) == 1


def test_topic_review_adapter_binds_issue_quotes_to_the_current_idea_field():
    class InvalidIssueQuote(GraphCompletion):
        def __call__(self, **kwargs):
            raw = super().__call__(**kwargs)
            if kwargs["response_format"]["json_schema"]["name"] == "TopicReviewResponse":
                output = json.loads(raw)
                output["reviews"][0]["issues"] = [{
                    "criterion": "reader_value", "severity": "blocking", "field": "reader_promise",
                    "quote": "这是来自样本、并非选题字段的引用。", "problem": "读者价值表达需要更具体。",
                    "revision_instruction": "明确读者能获得的内容。", "note_ids": [], "related_idea_ids": [],
                }]
                output["reviews"][0]["scores"][0]["score"] = 5
                return json.dumps(output, ensure_ascii=False)
            return raw

    model = InvalidIssueQuote()
    w = workflow(model, count=1, variants=2)

    with pytest.raises(CreativeError):
        w.run()

    judgement = w.state["topic_review"]["reviews"][0]["judgement"]
    issue = judgement["issues"][0]
    idea = w.state["topic"]["ideas"][0]
    assert issue["quote"] == idea["reader_promise"][:180]
    assert "来自样本" not in issue["quote"]


def test_graph_routes_sample_preprocessing_to_its_dedicated_model_only():
    creative_model, preprocess_model = GraphCompletion(), GraphCompletion()
    w = workflow(creative_model, count=1, variants=2, preprocess_complete=preprocess_model)

    w.run()

    assert preprocess_model.wire_calls
    assert all(call["response_format"]["json_schema"]["name"] == "SampleAnalysisResponse"
               for call in preprocess_model.wire_calls)
    assert any(call["response_format"]["json_schema"]["name"] == "SampleReviewResponse"
               for call in creative_model.wire_calls)
    assert all(call["response_format"]["json_schema"]["name"] != "SampleAnalysisResponse"
               for call in creative_model.wire_calls)
    assert len(preprocess_model.calls) + len(creative_model.calls) == w.budget.request_count


def test_passed_legacy_sample_review_rebuilds_registry_from_cache_without_model_calls():
    model = GraphCompletion()
    w = workflow(model, count=1, variants=2)
    w.run()
    before = w.budget.request_count
    legacy_summary = {key: value for key, value in w.state["sample_review"].items()
                      if key != "approved_claims"}
    w.state["sample_review"] = legacy_summary

    update = w.sample_review_node(w.state)

    assert update["sample_review"]["approved_claims"]
    assert w.budget.request_count == before


def test_sample_preprocessing_runs_at_most_three_batches_concurrently_and_merges_in_source_order():
    class ConcurrentCompletion(GraphCompletion):
        def __init__(self):
            super().__init__()
            self.lock = Lock()
            self.in_flight = 0
            self.max_in_flight = 0

        def __call__(self, **kwargs):
            if kwargs["response_format"]["json_schema"]["name"] == "SampleAnalysisResponse":
                with self.lock:
                    self.in_flight += 1
                    self.max_in_flight = max(self.max_in_flight, self.in_flight)
                sleep(0.03)
                try:
                    return super().__call__(**kwargs)
                finally:
                    with self.lock:
                        self.in_flight -= 1
            return super().__call__(**kwargs)

    model = ConcurrentCompletion()
    sample_sources = sources(11)
    w = LangGraphWorkflow(search=SEARCH, sources=sample_sources,
                          options={"model_id": 1, "idea_count": 1, "variant_count": 2}, complete=model)

    result, _ = w.run()

    sample_calls = [call for call in model.wire_calls
                    if call["response_format"]["json_schema"]["name"] == "SampleAnalysisResponse"]
    assert len(sample_calls) == 4
    assert 1 < model.max_in_flight <= 3
    assert [item["note_id"] for item in w.state["sample_analysis"]["observations"]] == [
        source["note_id"] for source in sample_sources
    ]
    assert len(result["ideas"]) == 1


def test_sample_evidence_reviews_run_at_most_three_batches_concurrently():
    class ConcurrentReviewer(GraphCompletion):
        def __init__(self):
            super().__init__()
            self.lock = Lock()
            self.in_flight = 0
            self.max_in_flight = 0

        def __call__(self, **kwargs):
            if kwargs["response_format"]["json_schema"]["name"] == "SampleReviewResponse":
                with self.lock:
                    self.in_flight += 1
                    self.max_in_flight = max(self.max_in_flight, self.in_flight)
                sleep(0.03)
                try:
                    return super().__call__(**kwargs)
                finally:
                    with self.lock:
                        self.in_flight -= 1
            return super().__call__(**kwargs)

    model = ConcurrentReviewer()
    w = LangGraphWorkflow(search=SEARCH, sources=sources(11),
                          options={"model_id": 1, "idea_count": 1, "variant_count": 2}, complete=model)

    result, _ = w.run()

    review_calls = [call for call in model.wire_calls
                    if call["response_format"]["json_schema"]["name"] == "SampleReviewResponse"]
    assert len(review_calls) == 4
    assert 1 < model.max_in_flight <= 3
    assert w.state["sample_review"]["status"] == "passed"
    assert len(result["ideas"]) == 1


def test_graph_full_chain_normalizes_title_review_blocking_severity_without_retry():
    class LowScoreSuggestion(GraphCompletion):
        def __call__(self, **kwargs):
            prompt, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
            raw = super().__call__(**kwargs)
            if kwargs["response_format"]["json_schema"]["name"] != "TitleReviewResponse":
                return raw
            output = json.loads(raw)
            review = output["reviews"][0]
            review["scores"]["reader_benefit"] = 5.0
            review["issues"] = [{
                "criterion": "reader_benefit", "severity": "suggestion", "field": "title",
                "quote": prompt["input"]["items"][0]["fields"]["title"],
                "problem": "读者收益不够具体。", "revision_instruction": "补出明确的观察收益。",
                "related_item_ids": [],
            }]
            return json.dumps(output, ensure_ascii=False)

    model = LowScoreSuggestion()
    w = workflow(model, count=1, variants=2)
    result, metadata = w.run()
    assert len(result["ideas"]) == 1
    assert all(len(idea["variants"]) == 2 for idea in result["ideas"])
    assert model.counts["title_review"] == 1
    assert metadata["request_count"] == 14


def test_graph_title_revision_restores_approved_candidates_changed_by_model():
    class LooseTitleRevision(GraphCompletion):
        def __init__(self):
            super().__init__()
            self.failed_idea_once = False
            self.corrupted_revision_once = False

        def __call__(self, **kwargs):
            prompt, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
            wire_name = kwargs["response_format"]["json_schema"]["name"]
            raw = super().__call__(**kwargs)
            if wire_name == "TitleReviewResponse":
                output = json.loads(raw)
                idea_ids = {item["idea_id"] for item in prompt["input"]["items"]}
                if "idea_2" in idea_ids and not self.failed_idea_once:
                    self.failed_idea_once = True
                    for review, item in zip(output["reviews"], prompt["input"]["items"]):
                        review["scores"]["hook"] = 4.0
                        review["issues"] = [{
                            "criterion": "hook", "severity": "blocking", "field": "title",
                            "quote": item["fields"]["title"], "problem": "标题钩子不够明确。",
                            "revision_instruction": "补出具体问题或收益。", "related_item_ids": [],
                        }]
                return json.dumps(output, ensure_ascii=False)
            if wire_name == "TitleResponse" and "quality_revision" in prompt and not self.corrupted_revision_once:
                self.corrupted_revision_once = True
                output = json.loads(raw)
                output["ideas"]["idea_1"]["candidates"][0]["title"] = "模型错误改动的已通过标题"
                return json.dumps(output, ensure_ascii=False)
            return raw

    model = LooseTitleRevision()
    w = workflow(model, count=3, variants=2)
    result, _ = w.run()
    assert len(result["ideas"]) == 3
    assert all(len(idea["variants"]) == 2 for idea in result["ideas"])
    assert model.counts["TitleCandidates"] == 2
    assert model.counts["title_review"] == 6
    assert w.state["pipeline"]["status"] == "completed"


def test_title_revision_briefs_bind_scores_and_actionable_issues_to_each_candidate():
    feedback = {"review": {"reviews": [
        {"judgement": {
            "item_id": "title_0_3",
            "scores": [{"criterion": "naturalness", "score": 3.5}],
            "issues": [{"criterion": "naturalness", "severity": "suggestion",
                        "problem": "“姿势换”搭配不自然。", "revision_instruction": "改成语义完整的自然短句。"}],
            "preserve": ["保留AI服装Lookbook主题。"],
        }},
        {"judgement": {"item_id": "title_1_0", "scores": [], "issues": [], "preserve": []}},
    ]}}
    briefs = title_revision_briefs(feedback, ["title_0_3"])
    assert briefs == [{
        "candidate_id": "title_0_3",
        "scores_below_minimum": [{"criterion": "naturalness", "score": 3.5, "minimum": 5.0}],
        "must_fix": [{"criterion": "naturalness", "severity": "suggestion",
                      "problem": "“姿势换”搭配不自然。", "revision_instruction": "改成语义完整的自然短句。"}],
        "preserve": ["保留AI服装Lookbook主题。"],
    }]


@pytest.mark.parametrize("cache_fault", [None, "version", "fingerprint", "review"])
def test_title_review_resume_reuses_only_valid_batches_and_keeps_context_compact(cache_fault):
    model = GraphCompletion()

    def complete(**kwargs):
        prompt, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
        if kwargs["response_format"]["json_schema"]["name"] == "TitleReviewResponse":
            batch = prompt["input"]["batch"]
            if batch["index"] == 2:
                raise CreativeError("模型服务返回错误（HTTP 401）", "ai_http_error", http_status=401)
            assert "output_schema" not in prompt
            assert kwargs["response_format"]["json_schema"]["schema"]["properties"]["reviews"]
            context = prompt["input"]["context"]
            assert set(context) == {"keyword", "requirements", "title_max_length", "topic_brief", "audience", "title_style", "style_summary"}
            assert len(prompt["input"]["items"]) == 5
            assert len(context["topic_brief"]["ideas"]) == 1
        return model(**kwargs)

    original = workflow(complete, count=3)
    with pytest.raises(CreativeError):
        original.run()
    cache = original.state["pipeline"]["stages"]["title"]["metadata"]["title_review_batches"]
    assert 1 <= len(cache["results"]) < 3
    assert original.state["pipeline"]["stages"]["title"]["metadata"]["failed_role"] == "title_review"

    saved = copy.deepcopy(original.state)
    saved["options"]["workflow"] = "langgraph"
    stored_cache = saved["pipeline"]["stages"]["title"]["metadata"]["title_review_batches"]
    if cache_fault == "version":
        stored_cache["version"] = "outdated"
    elif cache_fault == "fingerprint":
        stored_cache["input_fingerprint"] = "0" * 64
    elif cache_fault == "review":
        next(iter(stored_cache["results"].values()))["reviews"][0]["item_id"] = "title_unknown"
    task = SimpleNamespace(status="failed", error_type="ai_http_error", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })
    before = copy.deepcopy(task.payload)
    restored, role = prepare_resume(task)
    recovery = GraphCompletion()
    resumed = LangGraphWorkflow(search=restored["search"], sources=restored["sources"], options=restored["options"],
                                complete=recovery, resume_state=restored, resume_role=role)
    result, _ = resumed.run()
    assert len(result["ideas"]) == 3
    valid_cached = len(cache["results"])
    if cache_fault in ("version", "fingerprint"):
        valid_cached = 0
    elif cache_fault == "review":
        valid_cached -= 1
    calls = [c for c in recovery.wire_calls if c["response_format"]["json_schema"]["name"] == "TitleReviewResponse"]
    indices = [json.loads(c["user_prompt"])["input"]["batch"]["index"] for c in calls]
    assert len(calls) == 3 - valid_cached
    assert len(set(indices)) == len(indices)
    metadata = resumed.state["pipeline"]["stages"]["title"]["metadata"]
    assert metadata["title_review_reused_batches"] == valid_cached
    assert metadata["title_review_completed_batches"] == 3
    assert task.payload == before


@pytest.mark.parametrize("stage", ["TopicReviewResponse", "title_review", "copy_review"])
def test_conditional_review_edge_returns_to_producer_and_only_revises_failed_items(stage):
    model = GraphCompletion(fail_stage=stage)
    w = workflow(model, 3, 3)
    result, _ = w.run()
    assert len(result["ideas"]) == 3
    assert len(model.calls) == (34 if stage == "TopicReviewResponse" else 36)
    producer = {"TopicReviewResponse": "topic", "title_review": "title", "copy_review": "copy"}[stage]
    item = w.state["pipeline"]["stages"][producer]
    assert item["quality_revisions"] == 1
    history = [h for h in item["history"] if h["role"] == "producer"]
    assert history[0]["output"]["ideas"][-1] == history[-1]["output"]["ideas"][-1]


@pytest.mark.parametrize("owner", ["topic", "title", "copy", "image"])
def test_final_conditional_route_invalidates_and_rebuilds_only_dependencies(owner):
    model, states = GraphCompletion(fail_stage="final_review", final_owner=owner), []
    w = workflow(model, 3, 2, on_graph_state=states.append)
    result, _ = w.run()
    assert result and model.counts["final_review"] == 12
    assert any(s["pipeline"]["stages"]["final_review"]["status"] == "invalidated" and s["final_review"] is None for s in states)
    assert w.state["pipeline"]["stages"][owner]["quality_revisions"] == 1
    for stage in ("topic", "title", "copy", "image"):
        history = [h["output"] for h in w.state["pipeline"]["stages"][stage]["history"] if h["role"] == "producer"]
        if len(history) > 1:
            field = "variants" if stage == "image" else "ideas"
            assert history[0][field][-1] == history[-1][field][-1]


@pytest.mark.parametrize("stage,owner", [("TopicReviewResponse", "topic"), ("title_review", "title"), ("copy_review", "copy"), ("final_review", "image")])
def test_exhausted_review_cycle_stops_with_drafts_and_no_final_result(stage, owner):
    model = GraphCompletion(fail_stage=stage, always=True, final_owner=owner)
    w = workflow(model)
    with pytest.raises(CreativeError) as caught:
        w.run()
    assert caught.value.code == "creative_quality_exhausted"
    assert w.state["result"] is None
    assert w.state["pipeline"]["status"] == "revision_required"
    assert w.state["pipeline"]["stages"][owner]["quality_revisions"] == 2
    assert model.counts[stage] == (6 if stage == "final_review" else 3)


def test_budget_and_timeout_stop_without_further_requests_and_preserve_global_outputs():
    model = GraphCompletion()
    w = workflow(model, max_requests=4)
    with pytest.raises(CreativeError) as caught:
        w.run()
    assert caught.value.code == "creative_budget_exhausted"
    assert len(model.calls) == w.state["pipeline"]["request_count"] == 4
    assert w.state["audience"] and w.state["topic"] is None and w.state["result"] is None
    model = GraphCompletion(timeout_stage="CopyVariants")
    w = workflow(model)
    with pytest.raises(requests.ReadTimeout):
        w.run()
    assert model.counts["CopyVariants"] == 1
    assert w.state["title_review"]["status"] == "passed"
    assert w.state["copy"] is None
    assert "secret-transport-details" not in json.dumps(w.state)


def test_parallel_sample_analysis_persists_completed_batches_and_resume_retries_only_the_failure():
    notes = sources(7)
    peers_saved = Event()
    def fail_second_batch(**kwargs):
        prompt = json.loads(kwargs["user_prompt"])
        if (kwargs["response_format"]["json_schema"]["name"] == "SampleAnalysisResponse"
                and prompt["input"]["batch"]["index"] == 2):
            assert peers_saved.wait(5)
            raise CreativeError("模型服务返回错误（HTTP 401）", "ai_http_error", http_status=401)
        return model(**kwargs)

    model = GraphCompletion()
    def save_checkpoint(state):
        cached = state["pipeline"]["stages"]["sample_analysis"]["metadata"].get("sample_analysis_batches", {}).get("results", {})
        if len(cached) >= 2:
            peers_saved.set()

    original = LangGraphWorkflow(search=SEARCH, sources=notes,
                                 options={"model_id": 1, "workflow": "langgraph", "idea_count": 1, "variant_count": 2},
                                 complete=fail_second_batch, on_graph_state=save_checkpoint)
    with pytest.raises(CreativeError):
        original.run()
    saved = copy.deepcopy(original.state)
    metadata = saved["pipeline"]["stages"]["sample_analysis"]["metadata"]
    assert metadata["sample_batch_count"] == 3
    assert metadata["sample_completed_batches"] == 2
    assert len(metadata["sample_analysis_batches"]["results"]) == 2
    task = SimpleNamespace(status="failed", error_type="ai_http_error", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })
    restored, role = prepare_resume(task)
    resumed_model = GraphCompletion()
    resumed = LangGraphWorkflow(search=restored["search"], sources=restored["sources"], options=restored["options"],
                                complete=resumed_model, resume_state=restored, resume_role=role)
    result, pipeline_metadata = resumed.run()
    assert len(result["ideas"]) == 1
    sample = resumed.state["pipeline"]["stages"]["sample_analysis"]["metadata"]
    assert sample["sample_completed_batches"] == 3
    assert sample["sample_reused_batches"] == 2
    sample_calls = [call for call in resumed_model.wire_calls
                    if call["response_format"]["json_schema"]["name"] == "SampleAnalysisResponse"]
    assert len(sample_calls) == 1
    assert [json.loads(call["user_prompt"])["input"]["batch"]["index"] for call in sample_calls] == [2]
    # The split sample run also uses one-object title/copy reviews downstream.
    assert pipeline_metadata["request_count"] == 24


def test_failed_sample_review_resumes_only_unreviewed_batch():
    notes = sources(7)
    peers_saved = Event()
    model = GraphCompletion()

    def fail_second_review_batch(**kwargs):
        prompt = json.loads(kwargs["user_prompt"])
        if kwargs["response_format"]["json_schema"]["name"] == "SampleReviewResponse":
            index = prompt["input"]["batch"]["index"]
            if index == 2:
                assert peers_saved.wait(5)
                raise CreativeError("模型服务返回错误（HTTP 401）", "ai_http_error", http_status=401)
        return model(**kwargs)

    def save_checkpoint(state):
        cached = state["pipeline"]["stages"]["sample_analysis"]["metadata"].get("sample_review_batches", {}).get("results", {})
        if len(cached) >= 2:
            peers_saved.set()

    original = LangGraphWorkflow(search=SEARCH, sources=notes,
                                 options={"model_id": 1, "workflow": "langgraph", "idea_count": 1, "variant_count": 2},
                                 complete=fail_second_review_batch, on_graph_state=save_checkpoint)
    with pytest.raises(CreativeError):
        original.run()
    saved = copy.deepcopy(original.state)
    metadata = saved["pipeline"]["stages"]["sample_analysis"]["metadata"]
    assert saved["sample_analysis"] is not None and saved["sample_review"] is None
    assert saved["pipeline"]["stages"]["sample_analysis"]["metadata"]["failed_role"] == "sample_review"
    assert len(metadata["sample_review_batches"]["results"]) == 2
    task = SimpleNamespace(status="failed", error_type="ai_http_error", payload={
        "analysis": saved["options"], "search": saved["search"], "sources": saved["sources"], "graph_state": saved,
    })
    restored, role = prepare_resume(task)
    recovery_model = GraphCompletion()
    resumed = LangGraphWorkflow(search=restored["search"], sources=restored["sources"], options=restored["options"],
                                complete=recovery_model, resume_state=restored, resume_role=role)

    result, _ = resumed.run()

    review_calls = [call for call in recovery_model.wire_calls
                    if call["response_format"]["json_schema"]["name"] == "SampleReviewResponse"]
    sample_calls = [call for call in recovery_model.wire_calls
                    if call["response_format"]["json_schema"]["name"] == "SampleAnalysisResponse"]
    assert role == "sample_review"
    assert len(review_calls) == 1
    assert json.loads(review_calls[0]["user_prompt"])["input"]["batch"]["index"] == 2
    assert sample_calls == []
    assert resumed.state["sample_review"]["status"] == "passed"
    assert result["ideas"]


def test_semantically_rejected_sample_claim_is_excluded_and_workflow_continues():
    model = GraphCompletion()

    def reject_claim(**kwargs):
        raw = model(**kwargs)
        if kwargs["response_format"]["json_schema"]["name"] == "SampleReviewResponse":
            output = json.loads(raw)
            output["decisions"][0].update(supported=False, note_ids=[], reason="来源材料不能支持这个概括。")
            return json.dumps(output, ensure_ascii=False)
        return raw

    w = workflow(reject_claim)
    result, _ = w.run()

    assert result["ideas"]
    summary = w.state["sample_review"]
    assert summary["status"] == "passed"
    assert summary["rejected_claim_count"] == 1
    assert summary["rejected_claims"][0]["reason"] == "来源材料不能支持这个概括。"
    assert all(claim["claim_id"] != "claim_001_001" for claim in summary["approved_claims"])
    audience_call = next(call for call in model.wire_calls
                         if call["response_format"]["json_schema"]["name"] == "AudienceGroupingResponse")
    sent = json.loads(audience_call["user_prompt"])["input"]["sample_analysis"]
    assert "claim_001_001" not in json.dumps(sent, ensure_ascii=False)


def test_sample_evidence_below_minimum_stops_with_coverage_diagnostic():
    model = GraphCompletion()

    def reject_most_claims(**kwargs):
        raw = model(**kwargs)
        if kwargs["response_format"]["json_schema"]["name"] == "SampleReviewResponse":
            output = json.loads(raw)
            for decision in output["decisions"][2:]:
                decision.update(supported=False, note_ids=[], reason="当前样本没有充分支撑该结论。")
            return json.dumps(output, ensure_ascii=False)
        return raw

    w = workflow(reject_most_claims)
    with pytest.raises(CreativeError) as caught:
        w.run()

    assert caught.value.code == "creative_sample_evidence_insufficient"
    metadata = w.state["pipeline"]["stages"]["sample_analysis"]["metadata"]
    assert metadata["sample_review_approved_claim_count"] == 2
    assert metadata["sample_review_rejected_claim_count"] >= 1
    assert metadata["failed_role"] == "sample_review"
    assert w.state["audience"] is None


def test_sample_relevance_filter_preserves_raw_sources_and_batches_only_selected_notes():
    notes = sources(7)
    notes.append({"note_id": "weak-note", "title": "AI绘图工具记录", "content": ("AI绘图工具使用记录。" + "补充。" * 300)})
    notes.append({"note_id": "unrelated-note", "title": "周末咖啡探店", "content": "這家咖啡店很安靜，甜品也不错。"})
    original = copy.deepcopy(notes)
    model = GraphCompletion()
    w = LangGraphWorkflow(search=SEARCH, sources=notes,
                          options={"model_id": 1, "workflow": "langgraph", "idea_count": 1, "variant_count": 2},
                          complete=model)

    update = w.sample_node(copy.deepcopy(w.state))

    assert notes == original
    sent = [json.loads(call["user_prompt"])["input"]
            for call in model.wire_calls if call["response_format"]["json_schema"]["name"] == "SampleAnalysisResponse"]
    sent_ids = [source["note_id"] for batch in sent for source in batch["sources"]]
    assert "unrelated-note" not in sent_ids
    assert "weak-note" in sent_ids
    weak_input = next(source for batch in sent for source in batch["sources"] if source["note_id"] == "weak-note")
    assert weak_input["relevance_tier"] == "weak"
    assert weak_input["relevance_body_truncated"] is True
    assert weak_input["original_body_chars"] > 500
    assert weak_input["relevance_original_body_chars"] == weak_input["original_body_chars"]
    relevance = update["pipeline"]["stages"]["sample_analysis"]["metadata"]["sample_relevance"]
    assert relevance["total_count"] == 9
    assert relevance["selected_count"] == 8
    assert relevance["excluded_note_ids"] == ["unrelated-note"]
    assert relevance["weak_note_ids"] == ["weak-note"]
    assert relevance["version"] == "sample-relevance-v1"
    assert len(sent) == 3


def test_sample_relevance_filter_rejects_too_few_effective_samples():
    notes = sources(2)
    notes.append({"note_id": "unrelated-note", "title": "周末咖啡", "content": "咖啡很香。"})
    model = GraphCompletion()
    w = LangGraphWorkflow(search=SEARCH, sources=notes,
                          options={"model_id": 1, "workflow": "langgraph", "idea_count": 1, "variant_count": 2},
                          complete=model)
    with pytest.raises(CreativeError) as caught:
        w.sample_node(copy.deepcopy(w.state))
    assert caught.value.code == "sample_relevance_insufficient"
    assert "仅剩 2 篇可用样本" in str(caught.value)
    assert model.calls == []


@pytest.mark.parametrize("race", [False, True])
def test_cancellation_at_request_boundary_keeps_results_uncommitted(race):
    model, states = GraphCompletion(timeout_stage="TopicBrief" if race else None), []
    def save(state):
        if len(model.calls) == 3 and state["pipeline"]["status"] != "cancelled":
            raise PipelineCancelled()
        states.append(state)
    w = workflow(model, on_graph_state=save)
    with pytest.raises(PipelineCancelled):
        w.run()
    assert len(model.calls) == 3
    assert states[-1]["pipeline"]["status"] == "cancelled"
    assert states[-1]["topic"] is None


def test_changed_title_selection_rebinds_copy_and_image_groups():
    model = GraphCompletion(fail_stage="final_review", final_owner="title")
    def complete(**kwargs):
        output = json.loads(model(**kwargs))
        if model.calls[-1][0] == "title_review" and model.counts["title_review"] == 4:
            output["reviews"][1]["scores"] = {key: 9.0 for key in output["reviews"][1]["scores"]}
        return json.dumps(output)
    w = workflow(complete, 3, 2)
    result, _ = w.run()
    assert result["ideas"][0]["candidate_id"] == "title_0_1"
    assert w.state["copy"]["ideas"][0]["candidate_id"] == "title_0_1"
    assert w.state["image"]["variants"][0]["candidate_id"] == "title_0_1"


def test_topic_revision_uses_compact_target_only_context():
    model = GraphCompletion(fail_stage="TopicReviewResponse")
    with pytest.raises(CreativeError):
        workflow(model, count=1, variants=2).run()
    revision_calls = []
    for call in model.wire_calls:
        if call["response_format"]["json_schema"]["name"] != "TopicResponse":
            continue
        envelope, _ = json.JSONDecoder().raw_decode(call["user_prompt"])
        if "quality_revision" in envelope:
            revision_calls.append(call)
    assert len(revision_calls) == 2
    for call in revision_calls:
        envelope, _ = json.JSONDecoder().raw_decode(call["user_prompt"])
        assert len(call["user_prompt"]) < 4000
        assert envelope["input"]["topic_brief"]["ideas"][0]["idea_id"] == "idea_1"
        assert "sample_analysis" not in envelope["input"]
        assert envelope["input"]["audience_brief"]["segments"]
        assert len(envelope["input"]["revision_briefs"]) == 1
    second_envelope, _ = json.JSONDecoder().raw_decode(revision_calls[1]["user_prompt"])
    assert "本次针对以下未变化字段至少修复一项" in revision_calls[1]["user_prompt"]
    assert "获批结论" in revision_calls[1]["user_prompt"]
    assert "捏造步骤、案例或效果" in revision_calls[1]["user_prompt"]


def test_graph_has_visible_conditional_edges_and_node_write_scope_is_enforced():
    w = workflow(GraphCompletion())
    diagram = w.graph.get_graph()
    assert any(e.source == "topic_review" and e.target == "topic" and e.conditional for e in diagram.edges)
    assert any(e.source == "plan_revision" and e.target == "image" and e.conditional for e in diagram.edges)
    original = copy.deepcopy(w.state)
    def out_of_scope(state):
        state["search"]["keyword"] = "mutated"
        return {"copy": {}}
    with pytest.raises(CreativeError) as caught:
        w._node("sample_analysis", out_of_scope, SampleUpdate)(w.state)
    assert caught.value.code == "creative_node_scope_invalid"
    assert w.state["search"] == original["search"]


def test_model_does_not_echo_server_owned_versions_and_every_artifact_binds_current_state():
    model = GraphCompletion()
    w = workflow(model)
    w.run()
    for call in model.wire_calls:
        properties = call["response_format"]["json_schema"]["schema"]["properties"]
        assert not set(properties).intersection({"revision", "topic_revision", "input_fingerprint"})
        assert "原样回填" not in call["system_prompt"]
    assert w.state["title"]["topic_revision"] == w.state["pipeline"]["stages"]["topic"]["revision"]
    assert len(w.state["copy"]["input_fingerprint"]) == 64
    assert w.state["final_review"]["revision"] == 1


def test_cancellation_before_final_result_persistence_does_not_publish_result():
    states = []
    def save(state):
        if state["result"] is not None:
            raise PipelineCancelled()
        states.append(state)
    w = workflow(GraphCompletion(), on_graph_state=save)
    with pytest.raises(PipelineCancelled):
        w.run()
    assert states[-1]["pipeline"]["status"] == "cancelled"
    assert states[-1]["result"] is None


@pytest.mark.parametrize("kind", ["schema", "handoff"])
def test_distinct_output_failure_reason_is_saved_in_global_state(kind):
    model = GraphCompletion()
    def complete(**kwargs):
        output = json.loads(model(**kwargs))
        if kind == "schema":
            output["observations"][0]["note_id"] = 123
        else:
            output["observations"][0]["quote"] = "未出现在样本中的引文"
        return json.dumps(output)
    w = workflow(complete)
    with pytest.raises(CreativeError) as caught:
        w.run()
    assert caught.value.code == w.state["pipeline"]["error_code"] == f"creative_{kind}_invalid"
    assert len(model.calls) == 3 and w.state["audience"] is None
    assert len(w.state["pipeline"]["stages"]["sample_analysis"]["metadata"]["failed_attempts"]) == 3
