import json
from contextlib import contextmanager

import pytest
from loguru import logger
from pydantic import ValidationError, TypeAdapter

from backend.app.schemas.creative_graph import SampleAnalysisResponse
from backend.app.services.creative_diagnostics import diagnostic_context, diagnostic_event, validation_summary
from backend.app.services.creative_graph.agents import GraphOutputError, build_agents
from backend.app.services.creative_graph.model import CompatibleChatModel, response_parameter
from backend.app.services.creative_agents.sample_analysis import prepare_sample_input, validate_sample_evidence
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
def test_failed_quote_diagnostics_locate_source_without_logging_model_text(field):
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
    agent = build_agents(CompatibleChatModel(lambda **_: json.dumps(output)))["sample_analysis"]
    with capture_events() as events, diagnostic_context(task_id=123, workflow="langgraph"):
        with pytest.raises(GraphOutputError) as caught:
            agent.invoke(payload, validate_output=lambda r: validate_sample_evidence(r, payload))
    failures = [e for e in events if e["event"] == "输出校验失败"]
    assert len(failures) == 2
    assert [e["will_retry"] for e in failures] == [True, False]
    assert all(e["task_id"] == 123 and e["stage"] == "sample_analysis" for e in failures)
    assert path in caught.value.attempts[-1].error
    assert f"输入样本序号 {position}" in failures[-1]["reason"]
    assert "private-generated-quote" not in json.dumps(events)
    assert "sample-" not in json.dumps(events)


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


def test_diagnostic_context_is_restored_after_failure():
    with capture_events() as events:
        with pytest.raises(RuntimeError), diagnostic_context(task_id=99, workflow="langgraph"):
            diagnostic_event("测试")
            raise RuntimeError()
        diagnostic_event("测试")
    assert events[0]["task_id"] == 99
    assert "task_id" not in events[1]
