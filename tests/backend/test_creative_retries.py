import pytest
import requests

from backend.app.services.creative_analysis import CreativeError, request_analysis, retry_after_seconds
from backend.app.services.creative_agents.revision import PipelineCancelled, RequestBudget
from backend.app.services.creative_retries import request_with_retries


@pytest.mark.parametrize("success_at", [2, 3, 4, None])
def test_three_retries_are_additional_and_every_request_is_counted(monkeypatch, success_at):
    calls, failures, waits = [], [], []
    monkeypatch.setattr("backend.app.services.creative_retries.time.sleep", waits.append)
    def complete(**kwargs):
        calls.append(kwargs)
        if len(calls) == success_at:
            return "ok"
        raise CreativeError("模型服务返回错误（HTTP 524）", "ai_http_error", retryable=True, http_status=524)
    budget = RequestBudget(complete)
    def run():
        return request_with_retries(budget, before_attempt=lambda _: None, on_failure=failures.append, fixture="unchanged")
    if success_at:
        assert run() == "ok"
    else:
        with pytest.raises(CreativeError) as caught:
            run()
        assert caught.value.http_status == 524
    expected = success_at or 4
    assert len(calls) == budget.request_count == expected
    assert all(call == {"fixture": "unchanged"} for call in calls)
    assert len(failures) == expected - (1 if success_at else 0)
    assert sum(waits) == sum((15, 30, 60)[:expected - 1])
    if not success_at:
        assert [f["will_retry"] for f in failures] == [True, True, True, False]


@pytest.mark.parametrize("status,retryable", [(408, True), (429, True), (500, True), (502, True), (503, True), (504, True), (524, True), (400, False), (401, False), (403, False), (404, False), (422, False)])
def test_http_retry_classification_keeps_private_response_out_of_errors(status, retryable):
    response = requests.Response()
    response.status_code = status
    response.headers["Retry-After"] = "25"
    class Client:
        def _complete(self, **kwargs):
            raise requests.HTTPError("private-response", response=response)
    with pytest.raises(CreativeError) as caught:
        request_analysis(Client())
    assert caught.value.retryable is retryable
    assert caught.value.http_status == status
    assert caught.value.retry_after == 25
    assert "private" not in str(caught.value)


def test_request_analysis_overrides_caller_timeout_without_duplicate_keyword():
    seen = {}

    class Client:
        def _complete(self, **kwargs):
            seen.update(kwargs)
            return "ok"

    assert request_analysis(Client(), timeout=1, system_prompt="system") == "ok"
    assert seen["timeout"] == (10, 180)
    assert seen["system_prompt"] == "system"


def test_request_analysis_binding_diagnostic_logs_argument_names_only():
    from backend.app.services.creative_diagnostics import diagnostic_context
    from test_creative_diagnostics import capture_events

    class Client:
        def _complete(self, *, expected):
            return expected

    with capture_events() as events, diagnostic_context(task_id=999, workflow="langgraph"):
        with pytest.raises(TypeError):
            request_analysis(Client(), unexpected="private-value")
    event = next(item for item in events if item["event"] == "模型请求参数绑定失败")
    assert event["supplied_argument_names"] == ["timeout", "unexpected"]
    assert event["accepted_argument_names"] == ["expected"]
    assert "private-value" not in str(event)


def test_nontransient_and_business_errors_do_not_retry():
    calls = []
    def fail(**_):
        calls.append(1)
        raise CreativeError("参数不支持", "ai_http_error", http_status=400)
    with pytest.raises(CreativeError):
        request_with_retries(RequestBudget(fail), before_attempt=lambda _: None,
                             on_failure=lambda _: pytest.fail("not transient"))
    assert len(calls) == 1


def test_cancellation_and_budget_are_checked_while_waiting(monkeypatch):
    waits, failures = [], []
    monkeypatch.setattr("backend.app.services.creative_retries.time.sleep", waits.append)
    def fail(**_):
        raise CreativeError("超时", "ai_read_timeout", retryable=True)
    def cancel(index):
        if index is None:
            raise PipelineCancelled()
    budget = RequestBudget(fail)
    with pytest.raises(PipelineCancelled):
        request_with_retries(budget, before_attempt=cancel, on_failure=failures.append)
    assert budget.request_count == 1 and not waits
    budget = RequestBudget(fail, max_requests=1)
    with pytest.raises(CreativeError) as caught:
        request_with_retries(budget, before_attempt=lambda _: None, on_failure=failures.append)
    assert caught.value.code == "creative_budget_exhausted"
    assert budget.request_count == 1 and not waits


def test_retry_after_is_honored_without_retrying_earlier_than_long_provider_delay(monkeypatch):
    waits, calls = [], []
    monkeypatch.setattr("backend.app.services.creative_retries.time.sleep", waits.append)
    def complete(**_):
        calls.append(1)
        if len(calls) == 1:
            raise CreativeError("限流", "ai_http_error", retryable=True, retry_after=20)
        return "ok"
    assert request_with_retries(RequestBudget(complete), before_attempt=lambda _: None, on_failure=lambda _: None) == "ok"
    assert sum(waits) == 20
    calls.clear(); waits.clear()
    def fail(**_):
        calls.append(1)
        raise CreativeError("限流", "ai_http_error", retryable=True, retry_after=301)
    with pytest.raises(CreativeError):
        request_with_retries(RequestBudget(fail), before_attempt=lambda _: None, on_failure=lambda _: None)
    assert len(calls) == 1 and not waits
    assert retry_after_seconds("not-a-date") is None
    assert retry_after_seconds("nan") is None
    assert retry_after_seconds("15") == 15
