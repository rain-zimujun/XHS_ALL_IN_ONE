import copy
import hashlib
import json

import pytest
from pydantic import TypeAdapter, ValidationError

from backend.app.services.creative_agents.reader_needs import prepare_audience_input, validate_audience_brief
from backend.app.services.creative_graph.agents import GraphOutputError, build_agents
from backend.app.services.creative_graph.audience import (
    AUDIENCE_STEPS_VERSION, _complete_coverage_locally, _coverage_failures_exhausted,
    needs_response, run_audience_steps,
)
from backend.app.services.creative_graph.model import CompatibleChatModel, response_parameter
from test_creative_reader_needs import AUDIENCE, REQUIREMENTS, sample_analysis, valid_output
from test_structured_output import assert_strict_contract


def payload():
    return prepare_audience_input("AI 服装", AUDIENCE, REQUIREMENTS, sample_analysis())


class Completion:
    def __init__(self, transform=None):
        self.calls = []
        self.transform = transform

    def __call__(self, **kwargs):
        envelope, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
        p = envelope["input"]
        name = kwargs["response_format"]["json_schema"]["name"]
        self.calls.append((name, copy.deepcopy(p), copy.deepcopy(kwargs)))
        output = valid_output(p)
        claims = p.get("approved_claims", [])
        claim_for_note = {note_id: claim["claim_id"] for claim in claims for note_id in claim["note_ids"]}
        for section in ("segments", "shared_needs"):
            for item in output.get(section, []):
                for evidence in item.get("evidence", []):
                    evidence["claim_ids"] = list(dict.fromkeys(
                        claim_for_note[note_id] for note_id in evidence["note_ids"] if note_id in claim_for_note
                    )) if evidence["basis"] == "sample_evidence" else []
        if name == "AudienceGroupingResponse":
            output = {"segments": [{k: v for k, v in s.items() if k != "segment_id"} for s in output["segments"]]}
        else:
            ids = p["allowed_segment_ids"]
            for need in output["shared_needs"]:
                need["applies_to_segment_ids"] = ids
            for form in output["format_recommendations"]:
                form["for_segment_ids"] = ids
            output.pop("segments")
            output.pop("requested_audience_terms")
        if self.transform:
            output = self.transform(name, p, output, len(self.calls))
        return json.dumps(output, ensure_ascii=False)


def run(model, metadata=None, p=None):
    p = p or payload()
    metadata = metadata if metadata is not None else {}
    agent = build_agents(CompatibleChatModel(model))["audience"]
    return run_audience_steps(agent, p, metadata, lambda _: None, lambda r: validate_audience_brief(r, p))


def test_two_calls_assign_ids_and_pass_enum_unique_items_in_actual_request():
    model, metadata = Completion(), {}
    result = run(model, metadata).output
    assert result.requested_audience_terms == payload()["audience_terms"]
    assert [s.segment_id for s in result.segments] == ["segment_1", "segment_2"]
    assert len(model.calls) == 2
    grouping_schema = model.calls[0][2]["response_format"]["json_schema"]["schema"]
    assert "segment_id" not in json.dumps(grouping_schema)
    schema = model.calls[1][2]["response_format"]["json_schema"]["schema"]
    assert_strict_contract(schema, schema)
    for name, field in [("BoundAudienceNeed", "applies_to_segment_ids"), ("BoundAudienceFormat", "for_segment_ids")]:
        refs = schema["$defs"][name]["properties"][field]
        assert refs["items"]["enum"] == ["segment_1", "segment_2"]
        assert refs["uniqueItems"] is True
    assert metadata["audience_steps"]["groups"]


def test_repeated_coverage_failure_with_rejected_aliases_can_fill_only_valid_groups():
    p = payload()
    valid_groups = [{key: value for key, value in segment.items() if key != "segment_id"}
                    for segment in valid_output(p)["segments"]]
    missing = [term for term in p["audience_terms"] if term not in {
        term for group in valid_groups for term in group["audience_terms"]
    }]
    errors = "[{'field': 'segments[1]', 'reason': \"未知受众词=['AI视觉从业者']，同组受众词不得重复\"}]"
    metadata = {"failed_attempts": [
        {"error": f"segments 分组校验失败；缺少受众词={missing}；错误={errors}；最多8组"},
        {"error": f"segments 分组校验失败；缺少受众词={missing}；错误=[]；最多8组"},
    ]}

    assert _coverage_failures_exhausted(metadata)
    assert _complete_coverage_locally(valid_groups, p) == missing
    assert {term for group in valid_groups for term in group["audience_terms"]} == set(p["audience_terms"])


def test_audience_evidence_binds_claim_ids_and_note_ids_in_both_actual_schemas():
    p = payload()
    p["approved_claims"] = [
        {"claim_id": "claim_001_001", "section": "observations", "text": "配色灵感", "note_ids": ["sample-0"]},
        {"claim_id": "claim_001_002", "section": "observations", "text": "轮廓灵感", "note_ids": ["sample-1"]},
    ]
    p["approved_claim_ids"] = [claim["claim_id"] for claim in p["approved_claims"]]
    model = Completion()

    result = run(model, p=p).output

    for name, _input, kwargs in model.calls:
        schema = kwargs["response_format"]["json_schema"]["schema"]
        serialized = json.dumps(schema)
        assert "claim_001_001" in serialized and "claim_001_002" in serialized
    evidence = result.segments[0].evidence[0]
    assert evidence.claim_ids == ["claim_001_001", "claim_001_002"]


def test_audience_grouping_rebinds_model_mismatched_claim_ids_from_note_ids():
    p = payload()
    p["approved_claims"] = [
        {"claim_id": "claim_001_001", "section": "observations", "text": "配色灵感", "note_ids": ["sample-0"]},
        {"claim_id": "claim_001_002", "section": "observations", "text": "轮廓灵感", "note_ids": ["sample-1"]},
    ]
    p["approved_claim_ids"] = [claim["claim_id"] for claim in p["approved_claims"]]

    def transform(name, _payload, output, _count):
        if name == "AudienceGroupingResponse":
            for group in output["segments"]:
                for evidence in group["evidence"]:
                    if evidence["basis"] == "sample_evidence":
                        evidence["note_ids"] = ["sample-0"]
                        evidence["claim_ids"] = ["claim_001_002"]
        return output

    result = run(Completion(transform), p=p).output

    for group in result.segments:
        for evidence in group.evidence:
            if evidence.basis == "sample_evidence":
                assert evidence.note_ids == ["sample-0"]
                assert evidence.claim_ids == ["claim_001_001"]


def test_audience_group_binding_selects_a_source_claim_without_truncating_note_coverage():
    p = payload()
    p["approved_claims"] = [
        {"claim_id": f"claim_{index:03d}", "section": "observations", "text": f"claim {index}",
         "note_ids": [f"sample-{index % 3}"]}
        for index in range(1, 16)
    ]
    p["approved_claim_ids"] = [claim["claim_id"] for claim in p["approved_claims"]]

    result = run(Completion(), p=p).output

    for group in result.segments:
        for evidence in group.evidence:
            if evidence.basis == "sample_evidence":
                assert set(evidence.note_ids) == {
                    note_id for claim_id in evidence.claim_ids
                    for claim in p["approved_claims"] if claim["claim_id"] == claim_id
                    for note_id in claim["note_ids"]
                }


def test_audience_validation_rejects_claims_that_do_not_cover_the_cited_note():
    from backend.app.schemas.creative_agents import AudienceBrief
    from backend.app.services.creative_orchestrator import AgentValidationError

    p = payload()
    p["approved_claims"] = [
        {"claim_id": "claim_001_001", "section": "observations", "text": "配色灵感", "note_ids": ["sample-0"]},
    ]
    output = valid_output(p)
    output["segments"][0]["evidence"][0]["claim_ids"] = ["claim_001_001"]
    output["segments"][0]["evidence"][0]["note_ids"] = ["sample-1"]

    with pytest.raises(AgentValidationError, match="属于本条 evidence 引用"):
        validate_audience_brief(AudienceBrief.model_validate(output), p)


def test_hypothesis_cannot_reference_a_claim_without_its_source_note():
    from backend.app.schemas.creative_agents import AudienceBrief
    from backend.app.services.creative_orchestrator import AgentValidationError

    p = payload()
    p["approved_claims"] = [
        {"claim_id": "claim_001_001", "section": "observations", "text": "配色灵感", "note_ids": ["sample-0"]},
    ]
    output = valid_output(p)
    output["segments"][0]["evidence"][0].update(note_ids=["sample-0"], claim_ids=["claim_001_001"])
    output["shared_needs"][0]["evidence"][0].update(note_ids=["sample-0"], claim_ids=["claim_001_001"])
    evidence = output["shared_needs"][1]["evidence"][0]
    evidence.update(note_ids=[], claim_ids=["claim_001_001"])

    with pytest.raises(AgentValidationError, match="必须同时提供 note_id"):
        validate_audience_brief(AudienceBrief.model_validate(output), p)


@pytest.mark.parametrize("refs", [["invented_id"], ["segment_1", "segment_1"]])
def test_dynamic_reference_contract_also_validates_at_runtime(refs):
    _, need, _ = needs_response(["segment_1", "segment_2"])
    item = valid_output(payload())["shared_needs"][0]
    item["applies_to_segment_ids"] = refs
    with pytest.raises(ValidationError):
        TypeAdapter(need).validate_python(item, strict=True)


def test_coverage_repair_names_missing_term_and_only_appends_new_group():
    original = valid_output(payload())["segments"]
    def transform(name, p, output, count):
        if name == "AudienceGroupingResponse" and count == 1:
            output["segments"][1]["audience_terms"].remove("后期从业者")
        elif name == "AudienceGroupingResponse":
            assert p["repair"]["missing_audience_terms"] == ["后期从业者"]
            assert "后期从业者" in p["repair"]["reason"]
            assert len(p["repair"]["accepted_segments"]) == 2
            group = output["segments"][1]
            group["audience_terms"] = ["后期从业者"]
            output["segments"] = [group]
        return output
    model = Completion(transform)
    result = run(model).output
    assert len(model.calls) == 3
    assert len(result.segments) == 3
    assert result.segments[0].model_dump(exclude={"segment_id"}) == {k: v for k, v in original[0].items() if k != "segment_id"}
    assert result.segments[-1].audience_terms == ["后期从业者"]


def test_exhausted_coverage_only_failure_fills_missing_terms_in_valid_groups_locally():
    p = payload()
    valid_groups = valid_output(p)["segments"]
    groups = [{key: value for key, value in group.items() if key != "segment_id"} for group in valid_groups]
    groups[0]["audience_terms"] = ["电商", "服装行业"]
    groups[1]["audience_terms"] = ["AI", "时尚行业"]
    fingerprint = hashlib.sha256(json.dumps(p, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    metadata = {
        "failed_attempts": [
            {"error": "segments 分组校验失败；缺少受众词=['营销行业', '美工', '后期从业者']；错误=[]；最多8组"},
            {"error": "segments 分组校验失败；缺少受众词=['营销行业', '美工', '后期从业者']；错误=[]；最多8组"},
        ],
        "audience_steps": {"version": AUDIENCE_STEPS_VERSION, "fingerprint": fingerprint, "groups": groups},
    }
    model = Completion()

    result = run(model, metadata, p).output

    assert [call[0] for call in model.calls] == ["AudienceNeedsResponse"]
    assert {term for segment in result.segments for term in segment.audience_terms} == set(p["audience_terms"])
    assert "营销行业" in result.segments[0].audience_terms
    assert "美工" in result.segments[1].audience_terms
    assert "后期从业者" in result.segments[0].audience_terms
    assert any("分组归属属于技术性补齐" in item for item in result.limitations)
    assert metadata["audience_steps"]["locally_completed_terms"] == ["营销行业", "美工", "后期从业者"]


def test_local_coverage_completion_is_not_used_for_other_grouping_errors():
    p = payload()
    groups = [{key: value for key, value in group.items() if key != "segment_id"}
              for group in valid_output(p)["segments"]]
    groups[0]["audience_terms"] = ["电商", "服装行业"]
    groups[1]["audience_terms"] = ["AI", "时尚行业"]
    fingerprint = hashlib.sha256(json.dumps(p, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    metadata = {
        "failed_attempts": [
            {"error": "segments 分组校验失败；缺少受众词=['营销行业']；错误=[来源引用失败]；最多8组"},
            {"error": "segments 分组校验失败；缺少受众词=['营销行业']；错误=[来源引用失败]；最多8组"},
        ],
        "audience_steps": {"version": AUDIENCE_STEPS_VERSION, "fingerprint": fingerprint, "groups": groups},
    }
    model = Completion()

    result = run(model, metadata, p).output

    assert [call[0] for call in model.calls] == ["AudienceGroupingResponse", "AudienceNeedsResponse"]
    assert {term for segment in result.segments for term in segment.audience_terms} == set(p["audience_terms"])
    assert "locally_completed_terms" not in metadata["audience_steps"]


@pytest.mark.parametrize("fault", ["unknown", "duplicate", "evidence"])
def test_analysis_repair_preserves_valid_items_and_groups_even_after_native_schema_error(fault):
    first = {}
    def transform(name, p, output, count):
        if name == "AudienceNeedsResponse" and count == 2:
            first.update(copy.deepcopy(output))
            if fault == "evidence":
                output["shared_needs"][0]["evidence"][0]["note_ids"] = ["unknown-note"]
            else:
                output["shared_needs"][0]["applies_to_segment_ids"] = ["wrong_id"] if fault == "unknown" else ["segment_1", "segment_1"]
        elif name == "AudienceNeedsResponse":
            repair = p["repair"]
            assert repair["rejected"][0]["field"] == "shared_needs[0]"
            assert repair["rejected"][0]["allowed_segment_ids"] == ["segment_1", "segment_2"]
            assert repair["accepted"]["shared_needs"] == first["shared_needs"][1:]
            assert repair["accepted"]["format_recommendations"] == first["format_recommendations"]
            output["summary"] = "不应覆盖已经通过的摘要"
            output["shared_needs"] = [first["shared_needs"][0]]
            output["format_recommendations"] = []
        return output
    model, metadata = Completion(transform), {}
    result = run(model, metadata).output
    assert len(model.calls) == 3
    assert [c[0] for c in model.calls].count("AudienceGroupingResponse") == 1
    assert result.summary == first["summary"]
    assert [n.model_dump() for n in result.shared_needs] == first["shared_needs"][1:] + first["shared_needs"][:1]
    assert [r.model_dump() for r in result.format_recommendations] == first["format_recommendations"]


def test_network_failure_keeps_groups_and_resume_skips_grouping():
    model, metadata = Completion(), {}
    def failed(**kwargs):
        if kwargs["response_format"]["json_schema"]["name"] == "AudienceNeedsResponse":
            raise ConnectionError("offline")
        return model(**kwargs)
    with pytest.raises(ConnectionError):
        run(failed, metadata)
    restored = json.loads(json.dumps(metadata))
    recovery = Completion()
    run(recovery, restored)
    assert [c[0] for c in recovery.calls] == ["AudienceNeedsResponse"]


@pytest.mark.parametrize("fault", ["version", "fingerprint", "groups", "analysis"])
def test_cache_revalidation_and_input_binding(fault):
    metadata = {}
    run(Completion(), metadata)
    cache = metadata["audience_steps"]
    if fault in ("version", "fingerprint"):
        cache[fault] = "stale"
    elif fault == "groups":
        cache["groups"][1]["evidence"][0].update(basis="sample_evidence", note_ids=["unknown"])
    else:
        cache["analysis"]["shared_needs"][0]["applies_to_segment_ids"] = ["unknown"]
    model = Completion()
    run(model, metadata)
    assert len(model.calls) >= 1
    if fault in ("version", "fingerprint", "groups"):
        assert model.calls[0][0] == "AudienceGroupingResponse"
    else:
        assert model.calls[0][0] == "AudienceNeedsResponse"


def test_completed_cache_is_revalidated_and_reused_without_model_calls():
    metadata = {}
    first = run(Completion(), metadata).output
    model = Completion()
    second = run(model, json.loads(json.dumps(metadata))).output
    assert second == first
    assert not model.calls


def test_changed_group_meaning_invalidates_analysis_even_when_ids_stay_the_same():
    metadata = {}
    run(Completion(), metadata)
    metadata["audience_steps"]["groups"][0]["role_context"] = "新的受众使用场景"
    model = Completion()
    run(model, metadata)
    assert [c[0] for c in model.calls] == ["AudienceNeedsResponse"]


def test_changed_creation_requirements_invalidate_both_cached_steps():
    metadata = {}
    run(Completion(), metadata)
    p = payload()
    p["requirements"] += "，重点写服装灵感分享"
    model = Completion()
    run(model, metadata, p)
    assert [c[0] for c in model.calls] == ["AudienceGroupingResponse", "AudienceNeedsResponse"]


def test_persistent_bad_id_stops_and_only_saves_accepted_items():
    def transform(name, p, output, count):
        if name == "AudienceNeedsResponse":
            output["shared_needs"][0]["applies_to_segment_ids"] = ["invalid-id"]
        return output
    model, metadata = Completion(transform), {}
    with pytest.raises(GraphOutputError):
        run(model, metadata)
    assert len(model.calls) == 3
    assert "invalid-id" not in json.dumps(metadata)
    assert metadata["audience_steps"]["groups"]


def test_empty_repair_cannot_silently_drop_the_invalid_item():
    def transform(name, p, output, count):
        if name == "AudienceNeedsResponse" and count == 2:
            output["shared_needs"][0]["applies_to_segment_ids"] = ["wrong"]
        elif name == "AudienceNeedsResponse":
            output["shared_needs"] = []
            output["format_recommendations"] = []
        return output
    with pytest.raises(GraphOutputError) as caught:
        run(Completion(transform))
    assert "shared_needs" in caught.value.attempts[-1].error


def test_network_failure_during_partial_repair_preserves_pending_work_on_resume():
    def transform(name, p, output, count):
        if name == "AudienceNeedsResponse" and count == 2:
            output["shared_needs"][0]["applies_to_segment_ids"] = ["wrong"]
        return output
    model, metadata = Completion(transform), {}
    def failed(**kwargs):
        envelope, _ = json.JSONDecoder().raw_decode(kwargs["user_prompt"])
        if "repair" in envelope["input"]:
            raise ConnectionError("offline")
        return model(**kwargs)
    with pytest.raises(ConnectionError):
        run(failed, metadata)
    assert metadata["audience_steps"]["analysis_target_counts"]["shared_needs"] == 2
    def complete_repair(name, p, output, count):
        assert name == "AudienceNeedsResponse"
        assert len(p["repair"]["accepted"]["shared_needs"]) == 1
        assert p["repair"]["required_counts"]["shared_needs"] == 2
        output["shared_needs"] = output["shared_needs"][:1]
        output["format_recommendations"] = []
        return output
    recovery = Completion(complete_repair)
    result = run(recovery, json.loads(json.dumps(metadata))).output
    assert len(recovery.calls) == 1
    assert len(result.shared_needs) == 2


def test_full_group_slots_can_regroup_only_needed_tail_for_twenty_audiences():
    p = payload()
    p["audience_terms"] = [f"行业{i}" for i in range(20)]
    def transform(name, request, output, count):
        if name == "AudienceGroupingResponse":
            base = output["segments"][0]
            if count == 1:
                term_groups = [[term] for term in p["audience_terms"][:8]]
            else:
                repair = request["repair"]
                assert repair["regrouped_for_capacity"]
                assert repair["remaining_group_slots"] > 0
                missing = repair["missing_audience_terms"]
                term_groups = [missing[i:i + 4] for i in range(0, len(missing), 4)]
            output["segments"] = [{**copy.deepcopy(base), "audience_terms": terms} for terms in term_groups]
        return output
    model = Completion(transform)
    result = run(model, p=p).output
    assert len(result.segments) <= 8
    assert {term for s in result.segments for term in s.audience_terms} == set(p["audience_terms"])
