import copy
import json
from datetime import datetime
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest

from backend.app.services import creative_report
from backend.app.services.creative_report_reviews import current_decision, render_quality
from test_creative_sample_agent import SEARCH, sources
from test_creative_workflow import WorkflowCompletion, workflow


class Tags(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


@pytest.fixture
def report_case(tmp_path, monkeypatch):
    model = WorkflowCompletion(fail_stage="final_review", final_owner="image")
    w = workflow(model, count=3, variants=2)
    result, metadata = w.run()
    payload = {"search": SEARCH, "analysis": w.options.model_dump(), "result": result,
               "analysis_metadata": metadata, "sources": sources(), "failures": [], "pipeline": w.state,
               "private_config": "never-export-account-or-model-config"}
    task = SimpleNamespace(id=12, user_id=7, created_at=datetime(2026, 10, 21, 12), payload=payload)
    monkeypatch.setattr(creative_report, "get_settings", lambda: SimpleNamespace(storage_dir=tmp_path))
    return task, tmp_path, model


def exported(case):
    task, root, _ = case
    files = creative_report.export_report(task)["report_files"]
    content = {f["file_name"].rsplit(".", 1)[1]: (root / "exports" / f["file_name"]).read_text(encoding="utf-8") for f in files}
    return content["html"], json.loads(content["json"])


def test_export_preserves_current_pipeline_and_revision_history_without_task_config(report_case):
    task, _, model = report_case
    before, calls = copy.deepcopy(task.payload), len(model.calls)
    text, document = exported(report_case)
    assert document["schema_version"] == 3
    assert document["pipeline"] == task.payload["pipeline"]
    assert document["result"] == task.payload["result"]
    assert task.payload == before and len(model.calls) == calls
    assert "never-export-account-or-model-config" not in text + json.dumps(document)
    assert "多 Agent 阶段与质量记录" in text
    assert f"模型请求 {calls} / 36 次" in text
    assert "本次修改对象：copy_0_0" in text
    assert "历史评审不代表当前成稿通过" in text
    assert len(document["pipeline"]["stages"]["final_review"]["history"]) == 2


def test_html_scores_and_issue_details_are_bound_to_each_title_and_variant(report_case):
    task, _, _ = report_case
    # Different notes on each object detect index-based or first-version-only rendering.
    for i, idea in enumerate(task.payload["result"]["ideas"]):
        for v, variant in enumerate(idea["variants"]):
            decision = current_decision(task.payload["pipeline"], "copy", variant["variant_id"])
            decision["judgement"]["preserve"] = [f"unique-detail-{i}-{v}"]
    task.payload["pipeline"]["stages"]["copy"]["review_result"]["reviews"].reverse()
    text, _ = exported(report_case)
    articles = text.split('<article>')[1:]
    assert len(articles) == 3
    for i, article in enumerate(articles):
        content = article.split('</article>', 1)[0]
        assert "标题评分：8.00 分 · 通过" in content
        for v, variant in enumerate(task.payload["result"]["ideas"][i]["variants"]):
            start = content.index(f'<h3>文案 {v + 1} ·')
            end = content.find('<section><h3>文案', start + 1)
            block = content[start:end if end >= 0 else len(content)]
            assert variant["body"] in block
            assert f"unique-detail-{i}-{v}" in block
            assert "正文评分：8.00 分 · 通过" in block
            assert "终审评分：8.00 分 · 通过" in block
            assert variant["image_plan"][0]["prompt"] in block


def test_internal_review_details_are_collapsed_and_untrusted_text_is_escaped(report_case):
    task, _, _ = report_case
    stage = task.payload["pipeline"]["stages"]["final_review"]
    review = stage["review_result"]["reviews"][0]
    review["judgement"]["issues"] = [{"severity": "blocking", "field": "cover_text", "quote": '<img src=x onerror="alert(1)">',
                                      "problem": "没有依据", "revision_instruction": "删去承诺", "responsible_stage": "image"}]
    review["quality"]["passed"] = False
    review["quality"]["scores"][0]["comment"] = '<script>alert("judge")</script>'
    review["judgement"]["preserve"] = ["</summary><script>alert(1)</script>"]
    stage["history"][0]["target_ids"] = ['<img src=x>']
    stage["error"] = '<script>alert("stage")</script>'
    text, _ = exported(report_case)
    tags = Tags(text).tags
    assert not any(tag in ("script", "img", "iframe") for tag, _ in tags)
    assert all("open" not in attrs for tag, attrs in tags if tag == "details")
    assert "&lt;script&gt;" in text and "&lt;img src=x" in text
    assert "负责修改：配图" in text and "修改建议：删去承诺" in text
    assert "评审：8.00 分 · 待修改" in text


def test_invalidated_review_is_only_in_audit_and_missing_ids_never_match(report_case):
    task, _, _ = report_case
    stage = task.payload["pipeline"]["stages"]["copy"]
    stage["status"] = "invalidated"
    stage["review_result"]["reviews"][0]["judgement"]["preserve"] = ["stale-detail"]
    text, _ = exported(report_case)
    assert "stale-detail" not in text.split("多 Agent 阶段与质量记录", 1)[0]
    assert "stale-detail" in text.split("多 Agent 阶段与质量记录", 1)[1]
    assert "本阶段旧评审已失效" in text
    assert current_decision(task.payload["pipeline"], "title", None) is None
    assert current_decision(task.payload["pipeline"], "title", "missing") is None


@pytest.mark.parametrize("value", [None, {}, {"passed": "true", "overall_score": "9"}])
def test_malformed_scores_do_not_turn_into_passes(value):
    assert render_quality(value, label="正文评分") == "<p>正文评分：暂无有效评分记录。</p>"


def test_no_pipeline_keeps_v2_shape_without_fabricated_reviews(report_case):
    task, _, _ = report_case
    task.payload.pop("pipeline")
    for idea in task.payload["result"]["ideas"]:
        idea.pop("quality")
        for variant in idea["variants"]:
            variant.pop("quality")
    text, document = exported(report_case)
    assert document["schema_version"] == 2 and "pipeline" not in document
    assert "多 Agent 阶段与质量记录" not in text
    assert "评分：" not in text
