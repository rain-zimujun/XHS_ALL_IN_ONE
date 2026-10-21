"""Escaped, read-only presentation of saved pipeline assessments."""
from __future__ import annotations

import html

from pydantic import ValidationError

from backend.app.services.creative_quality import QualityReview, RUBRICS


STAGE_NAMES = {
    "sample_analysis": "样本分析", "audience": "读者需求", "topic": "选题",
    "title": "标题", "copy": "正文", "image": "配图", "final_review": "终审",
}
STATUS_NAMES = {
    "pending": "等待中", "running": "进行中", "completed": "已完成", "failed": "失败",
    "skipped": "已跳过", "revision_required": "待修改", "invalidated": "已失效", "cancelled": "已停止",
}
CRITERION_NAMES = {item.key: item.label for rubric in RUBRICS.values() for item in rubric}
FIELD_NAMES = {
    "topic": "选题", "title": "标题", "body": "正文", "tags": "话题标签",
    "cover_text": "封面文字", "image_plan": "配图方案", "material_notes": "素材说明",
    "source_note_ids": "参考来源", "core_angle": "核心切入点", "reader_promise": "读者承诺",
    "must_cover": "必须覆盖要点", "boundaries": "事实边界",
}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def p(value) -> str:
    return f"<p>{esc(value)}</p>"


def record(value) -> dict:
    return value if isinstance(value, dict) else {}


def strings(value) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def quality_model(value) -> QualityReview | None:
    try:
        return QualityReview.model_validate(value, strict=True)
    except ValidationError:
        return None


def review_batch(output) -> dict:
    outer = record(output)
    return record(outer["review"]) if "review" in outer else outer


def decisions(output) -> list[dict]:
    entries = review_batch(output).get("reviews")
    return [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []


def item_id(decision: dict) -> str:
    judgement = record(decision.get("judgement"))
    value = judgement.get("item_id") or judgement.get("idea_id")
    return value if isinstance(value, str) else ""


def current_decision(pipeline, stage: str, identifier) -> dict | None:
    state = record(record(record(pipeline).get("stages")).get(stage))
    if not identifier or state.get("status") != "completed":
        return None
    return next((entry for entry in decisions(state.get("review_result")) if item_id(entry) == identifier), None)


def render_quality(value, *, label: str, decision: dict | None = None) -> str:
    quality = quality_model(value)
    if quality is None:
        return p(f"{label}：暂无有效评分记录。")
    status = "通过" if quality.passed else "待修改"
    parts = [p(f"{label}：{quality.overall_score:.2f} 分 · {status}"),
             '<details><summary>分项评分与修改建议</summary>']
    for score in quality.scores:
        parts.append(p(f"{CRITERION_NAMES.get(score.criterion, score.criterion)} · {score.score:g} 分：{score.comment}"))
    judgement = record(record(decision).get("judgement"))
    issues = judgement.get("issues") if isinstance(judgement.get("issues"), list) else []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        severity = "阻断问题" if issue.get("severity") == "blocking" else "改进建议"
        field = issue.get("field", "")
        owner = issue.get("responsible_stage")
        parts.append(p(f"{severity} · {FIELD_NAMES.get(field, field)}"))
        if owner:
            parts.append(p(f"负责修改：{STAGE_NAMES.get(owner, owner)}"))
        for key, name in (("quote", "原文定位"), ("problem", "问题"), ("revision_instruction", "修改建议")):
            parts.append(p(f"{name}：{issue.get(key, '')}"))
    parts.extend(p(f"阻断：{issue}") for issue in quality.blocking_issues)
    parts.extend(p(f"修改建议：{instruction}") for instruction in quality.revision_instructions)
    parts.extend(p(f"建议保留：{value}") for value in strings(judgement.get("preserve")))
    if judgement.get("comparison"):
        parts.append(p(f"版本比较：{judgement['comparison']}"))
    if quality.reviewer_notes:
        parts.append(p(quality.reviewer_notes))
    if not decision:
        parts.append(p("此结果仅保存汇总评分，没有可关联的逐项问题记录。"))
    parts.append('</details>')
    return ''.join(parts)


def render_batch(output) -> str:
    batch = review_batch(output)
    parts = [p(batch["summary"])] if isinstance(batch.get("summary"), str) else []
    for entry in decisions(output):
        parts.extend([
            f'<details><summary>对象 {esc(item_id(entry))}</summary>',
            render_quality(entry.get("quality"), label="评审", decision=entry), '</details>',
        ])
    parts.extend(p(f"评审局限：{value}") for value in strings(batch.get("limitations")))
    return ''.join(parts) or p("尚无逐项评审记录。")


def render_pipeline(pipeline) -> str:
    pipeline = record(pipeline)
    if not pipeline:
        return ""
    stages = record(pipeline.get("stages"))
    status = pipeline.get("status", "")
    parts = ['<section><h2>多 Agent 阶段与质量记录</h2>',
             p(f"流程状态：{STATUS_NAMES.get(status, status)} · 模型请求 {pipeline.get('request_count', 0)} / {pipeline.get('max_requests', 36)} 次"),
             p("模型评分仅供辅助；历史评审不代表当前成稿通过。内部记录与正文分开，不属于发布文案。"),
             '<div class="table-scroll"><table><thead><tr><th>阶段</th><th>状态</th><th>产物版本</th><th>质量返工</th></tr></thead><tbody>']
    for key, name in STAGE_NAMES.items():
        state = record(stages.get(key))
        if not state:
            continue
        status = state.get("status", "")
        parts.append('<tr>' + ''.join(f'<td>{esc(value)}</td>' for value in (
            name, STATUS_NAMES.get(status, status), state.get("revision", 0), state.get("quality_revisions", 0),
        )) + '</tr>')
    parts.append('</tbody></table></div><details><summary>评审问题与返工历史（内部记录）</summary>')
    for key, name in STAGE_NAMES.items():
        state = record(stages.get(key))
        if not state:
            continue
        parts.append(f'<details><summary>{esc(name)} · 版本 {esc(state.get("revision", 0))}</summary>')
        if state.get("status") == "invalidated":
            parts.append(p("上游已变更，本阶段旧评审已失效。"))
        if state.get("error"):
            parts.append(p(state["error"]))
        if state.get("review_result"):
            parts.extend(['<h3>最近保存的评审</h3>', render_batch(state["review_result"])])
        history = state.get("history") if isinstance(state.get("history"), list) else []
        if history:
            parts.append('<details><summary>历史过程（仅供追溯）</summary>')
            for entry in history:
                if not isinstance(entry, dict):
                    continue
                role = "评审" if entry.get("role") == "reviewer" else "生成"
                parts.append(p(f"{role} · 版本 {entry.get('revision', 0)} · {entry.get('finished_at', '')}"))
                targets = strings(entry.get("target_ids"))
                if targets:
                    parts.append(p("本次修改对象：" + "、".join(targets)))
                if entry.get("role") == "reviewer":
                    parts.append(render_batch(entry.get("output")))
            parts.append('</details>')
        parts.append('</details>')
    parts.append('</details></section>')
    return ''.join(parts)
