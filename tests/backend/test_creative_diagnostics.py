import json
from contextlib import contextmanager

import pytest
from loguru import logger
from pydantic import ValidationError, TypeAdapter

from backend.app.schemas.creative_graph import SampleAnalysisResponse
from backend.app.services.creative_diagnostics import (
    MAX_LOGGED_MODEL_RESPONSE_CHARS, capture_model_quotes, diagnostic_context,
    diagnostic_event, failed_model_response, quote_mismatch_details, record_model_quotes,
    validation_summary,
)
from backend.app.services.creative_graph.agents import GraphOutputError, build_agents
from backend.app.services.creative_graph.model import CompatibleChatModel, response_parameter
from backend.app.services.creative_agents.sample_analysis import prepare_sample_input, validate_sample_evidence, run_sample_analysis
from backend.app.services.creative_orchestrator import AgentRunError
from test_creative_sample_agent import SEARCH, OPTIONS, sources, valid_output


@contextmanager
def capture_events():
    events = []
    sink = logger.add(lambda message: events.append(json.loads(message.record["message"].split("创作诊断 ", 1)[1])),
                      filter=lambda record: record["message"].startswith("创作诊断 "))
    try:
        yield events
    finally:
        logger.remove(sink)


@pytest.mark.parametrize("field", ["observation", "evidence"])
@pytest.mark.parametrize("workflow", ["langgraph", "multi_agent"])
def test_failed_quote_diagnostics_record_only_failed_quote_and_bounded_source_windows(field, workflow):
    payload = prepare_sample_input(SEARCH, sources(), OPTIONS)
    output = valid_output(payload)
    if field == "observation":
        output["observations"][1]["quote"] = "private-generated-quote"
        path = "observations[1].quote"
        position = 2
    else:
        output["topic_patterns"][0]["evidence"][0]["quote"] = "private-generated-quote"
        path = "topic_patterns[0].evidence[0].quote"
        position = 1
    output["summary"] = "unrelated-model-summary"
    complete = lambda **_: json.dumps(output)
    with capture_events() as events, diagnostic_context(task_id=123, workflow=workflow):
        if workflow == "langgraph":
            agent = build_agents(CompatibleChatModel(complete))["sample_analysis"]
            with pytest.raises(GraphOutputError) as caught:
                agent.invoke(payload, validate_output=lambda r: validate_sample_evidence(r, payload))
        else:
            with pytest.raises(AgentRunError) as caught:
                run_sample_analysis(search=SEARCH, sources=sources(), options=OPTIONS, complete=complete)
    failures = [e for e in events if e["event"] == "输出校验失败"]
    assert len(failures) == 2
    assert [e["will_retry"] for e in failures] == [True, False]
    assert all(e["task_id"] == 123 and e["stage"] == "sample_analysis" for e in failures)
    assert path in caught.value.attempts[-1].error
    assert f"输入样本序号 {position}" in failures[-1]["reason"]
    quotes = [event for event in events if event["event"] == "原文引用不匹配"]
    assert len(quotes) == 2
    assert quotes[0]["task_id"] == 123 and quotes[1]["attempt"] == 2
    assert quotes[0]["model_quote"] == "private-generated-quote"
    assert quotes[0]["validated_quote"] == "private-generated-quote"
    assert quotes[0]["source_position"] == position
    assert quotes[0]["source_candidates"][0]["source_window"] == payload["sources"][position - 1]["title"]
    assert caught.value.attempts[-1].quote_diagnostic["model_quote"] == "private-generated-quote"
    raw_outputs = [e for e in events if e["event"] == "模型原始输出（校验失败）"]
    if workflow == "langgraph":
        assert len(raw_outputs) == 2
        assert raw_outputs[0]["model_response"] == json.dumps(output)
        assert "private-generated-quote" in raw_outputs[0]["model_response"]
        assert "unrelated-model-summary" in raw_outputs[0]["model_response"]
    else:
        assert not raw_outputs


def test_graph_logs_actual_quote_before_typed_dict_whitespace_normalization():
    payload = prepare_sample_input(SEARCH, sources(), OPTIONS)
    output = valid_output(payload)
    output["observations"][0]["quote"] = " \n原文没有这句\t "
    agent = build_agents(CompatibleChatModel(lambda **_: json.dumps(output)))["sample_analysis"]
    with capture_events() as events, pytest.raises(GraphOutputError):
        agent.invoke(payload, validate_output=lambda r: validate_sample_evidence(r, payload))
    detail = next(event for event in events if event["event"] == "原文引用不匹配")
    assert detail["model_quote"] == " \n原文没有这句\t "
    assert detail["validated_quote"] == "原文没有这句"
    assert detail["model_quote_changed_by_validation"] is True
    assert detail["model_quote_truncated"] is False


def test_failed_model_response_log_is_bounded():
    raw = "x" * (MAX_LOGGED_MODEL_RESPONSE_CHARS + 10)
    detail = failed_model_response({"text": raw})
    assert len(detail["model_response"]) == MAX_LOGGED_MODEL_RESPONSE_CHARS
    assert detail["model_response_chars"] == len(raw)
    assert detail["model_response_truncated"] is True
    assert failed_model_response({}) is None


@pytest.mark.parametrize("kind", ["whitespace", "width", "wrong_field", "cross_segment", "long"])
def test_quote_comparison_hints_do_not_relax_literal_matching_or_join_segments(kind):
    source = {"title": "服装AI灵感", "body_segments": ["第一段：轻盈的裙摆\n设计细节", "第二段：海风"]}
    quote, field = "裙摆设计细节", "body"
    if kind == "width":
        source["body_segments"] = ["ＡＩ服装"]
        quote = "AI服装"
    elif kind == "wrong_field":
        quote, field = source["body_segments"][0], "title"
    elif kind == "cross_segment":
        quote = "设计细节第二段"
    elif kind == "long":
        source["body_segments"] = ["前" * 1400 + "原文：裙摆与海风" + "后" * 400]
        quote = "原文：裙摆和海风"
    detail = quote_mismatch_details(path=("observations", 0, "quote"), quote=quote,
                                    source=source, source_position=1, field=field)
    candidates = detail["source_candidates"]
    body = candidates[1]
    assert all(len(candidate["source_window"]) <= 360 for candidate in candidates)
    assert len(detail["differences"]) <= 12
    if kind == "whitespace":
        assert body["whitespace_match"] and not body["exact_match"]
    elif kind == "width":
        assert body["nfkc_whitespace_match"] and not body["whitespace_match"]
    elif kind == "wrong_field":
        assert body["exact_match"] and not body["allowed"]
    elif kind == "cross_segment":
        assert not any(candidate["whitespace_match"] for candidate in candidates)
    else:
        assert body["window_start"] > 0
        assert "原文：裙摆与海风" in body["source_window"]


def test_raw_quote_capture_is_scoped_and_bounded():
    with capture_model_quotes() as outer:
        record_model_quotes({"observations": [{"quote": "外层"}]})
        with pytest.raises(RuntimeError), capture_model_quotes() as inner:
            record_model_quotes({"observations": [{"quote": "内" * 1000}]})
            raise RuntimeError()
        assert inner[("observations", 0, "quote")] == ("内" * 600, 1000)
        record_model_quotes({"observations": [{"quote": "恢复外层"}]})
        assert outer[("observations", 0, "quote")] == ("恢复外层", 4)
    record_model_quotes({"observations": [{"quote": "不应写入"}]})
    assert outer[("observations", 0, "quote")][0] == "恢复外层"


def test_schema_diagnostics_locate_known_fields_but_hide_unknown_keys_and_values():
    output = valid_output(prepare_sample_input(SEARCH, sources(), OPTIONS))
    output["observations"][0]["note_id"] = 123
    output["private-key-cookie"] = "private-body"
    with pytest.raises(ValidationError) as caught:
        TypeAdapter(SampleAnalysisResponse).validate_python(output, strict=True)
    message = validation_summary(caught.value, response_parameter(SampleAnalysisResponse)["json_schema"]["schema"])
    assert "observations.0.note_id: string_type" in message
    assert "[未知字段]: extra_forbidden" in message
    assert "private" not in message


def test_transport_failure_logged_once_without_exception_message_or_retry():
    calls = []
    def complete(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("Authorization: Bearer private-key; private-response")
    agent = build_agents(CompatibleChatModel(complete))["sample_analysis"]
    with capture_events() as events:
        with pytest.raises(RuntimeError):
            agent.invoke({}, validate_output=lambda _: None)
    failure = events[-1]
    assert failure["event"] == "模型调用中断"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["will_retry"] is False
    assert len(calls) == 1
    assert "private" not in json.dumps(events)
    from backend.app.services.creative_diagnostics import exception_details
    try:
        raise RuntimeError("Authorization: Bearer private-key; private-response")
    except RuntimeError as exc:
        details = exception_details(exc)
    assert details["traceback_frames"]
    assert all(set(frame) == {"file", "function", "line"} for frame in details["traceback_frames"])
    assert "private-key" not in json.dumps(details)
    assert "private-response" not in json.dumps(details)


def test_type_error_diagnostic_extracts_only_python_keyword_argument_name():
    from backend.app.services.creative_diagnostics import exception_details

    try:
        raise TypeError("request failed for private body: got multiple values for keyword argument 'timeout'")
    except TypeError as exc:
        details = exception_details(exc)
    assert details["type_error_kind"] == "duplicate_keyword"
    assert details["argument_name"] == "timeout"
    assert "private body" not in json.dumps(details)


def test_diagnostic_context_is_restored_after_failure():
    with capture_events() as events:
        with pytest.raises(RuntimeError), diagnostic_context(task_id=99, workflow="langgraph"):
            diagnostic_event("测试")
            raise RuntimeError()
        diagnostic_event("测试")
    assert events[0]["task_id"] == 99
    assert "task_id" not in events[1]
