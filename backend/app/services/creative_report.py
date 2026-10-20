"""Local, escaped HTML and machine-readable JSON; no original media copied."""
from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.models import Task
from backend.app.services.creative_service import valid_note_url


def escape(value) -> str:
    return html.escape(str(value), quote=True)


def paragraph(value) -> str:
    return f"<p>{escape(value)}</p>"


def source_link(source: dict) -> str:
    label = escape(source.get("title") or source["note_id"])
    url = source.get("note_url", "")
    if not valid_note_url(url):
        return label
    return f'<a href="{escape(url)}" target="_blank" rel="noopener noreferrer">{label}</a>'


def render_report(document: dict) -> str:
    result = document["result"]
    sources = {s["note_id"]: s for s in document["sources"]}
    metadata = document["analysis_metadata"]
    parts = [
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">',
        '<title>小红书热点创作报告</title><style>',
        'body{max-width:960px;margin:40px auto;padding:0 24px;font:16px/1.8 system-ui,sans-serif;color:#1f2937;background:#fafafa}',
        'article,section{background:white;border:1px solid #e5e7eb;border-radius:12px;padding:24px;margin:20px 0}',
        'p{white-space:pre-wrap;overflow-wrap:anywhere}h1,h2,h3{line-height:1.4}a{color:#2563eb}small{color:#64748b}',
        'pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border-top:1px solid #eee;padding:12px 0}',
        '@media print{body{margin:0;padding:0}article{break-inside:avoid}}',
        '</style></head><body><h1>小红书热点创作报告</h1>',
        paragraph(f"关键词：{document['search']['keyword']} · 任务 #{document['task_id']} · {document['created_at']}"),
        paragraph(metadata.get("disclaimer", "创作建议不保证流量，请人工核实。")),
        f'<small>有效正文 {len(sources)} 篇；每篇分析最多使用 {escape(metadata.get("excerpt_chars_per_note", ""))} 字符；{"长正文保留首尾" if metadata.get("excerpt_strategy") == "head_and_tail" else "正文从开头截取"}；完整正文见文末。模型：{escape(metadata.get("model_name", ""))}</small>',
        '<section><h2>样本分析</h2>', paragraph(result["summary"]),
        '<ul>' + ''.join(f'<li>{escape(p)}</li>' for p in result["patterns"]) + '</ul></section>',
    ]
    if result.get("style_summary"):
        parts.extend(['<section><h2>文案风格依据</h2>', paragraph(result["style_summary"]), '</section>'])
    for i, idea in enumerate(result["ideas"], 1):
        parts.append(f'<article><h2>标题 {i}：{escape(idea["title"])}</h2>')
        # Historical results have one real body. Preserve it without inventing variants.
        variants = idea.get("variants") or [{**idea, "label": "原有文案"}]
        if not idea.get("variants"):
            parts.append(paragraph("旧版结果：仅保留原有的 1 版文案和配图。"))
        for v, variant in enumerate(variants, 1):
            parts.extend([
                f'<section><h3>文案 {v} · {escape(variant["label"])}</h3>',
                '<h4>正文文案</h4>', paragraph(variant["body"]),
                '<h4>话题标签</h4>', paragraph(" ".join("#" + tag.lstrip("#") for tag in variant["tags"])),
                '<h4>配图建议与生图提示词</h4>', paragraph("封面文字：" + variant["cover_text"]),
            ])
            for j, plan in enumerate(variant["image_plan"], 1):
                parts.extend([
                    f'<h5>图 {j} · {escape(plan["purpose"])}</h5>', paragraph(plan["composition"]),
                    '<b>生图提示词</b>', paragraph(plan["prompt"]), '<b>负向提示词</b>', paragraph(plan["negative_prompt"]),
                ])
            parts.append('</section>')
        parts.extend(['<details><summary>推荐理由与参考来源</summary>', paragraph(idea["reason"])])
        if idea.get("alternative_titles"):
            parts.append('<h3>旧版备选标题</h3><ul>' + ''.join(f'<li>{escape(t)}</li>' for t in idea["alternative_titles"]) + '</ul>')
        parts.append('<h3>参考样本</h3><ul>' + ''.join(f'<li>{source_link(sources[n])}</li>' for n in idea["source_note_ids"] if n in sources) + '</ul></details></article>')
    parts.append('<section><h2>本次采集样本</h2>')
    for source in sources.values():
        parts.extend([
            '<details><summary>' + source_link(source) + '</summary>',
            paragraph(f"作者：{source.get('author_name', '')} · 点赞 {source.get('likes', 0)} · 收藏 {source.get('collects', 0)} · 评论 {source.get('comments', 0)}"),
            paragraph("采集时间：" + source.get("captured_at", "")),
            paragraph(source.get("content", "")), '</details>',
        ])
    parts.extend(['</section><section><h2>搜索与创作条件</h2><pre>', escape(json.dumps({
        "search": document["search"], "analysis": document["analysis"], "failures": document["failures"],
    }, ensure_ascii=False, indent=2)), '</pre></section></body></html>'])
    return ''.join(parts)


def atomic_write(path: Path, content: str):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".report-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_report(task: Task) -> dict:
    payload = task.payload
    document = {
        "schema_version": 2 if any(idea.get("variants") for idea in payload["result"]["ideas"]) else 1,
        "task_id": task.id, "created_at": task.created_at.isoformat(),
        "search": payload["search"], "analysis": payload["analysis"], "result": payload["result"],
        "analysis_metadata": payload.get("analysis_metadata", {}),
        "sources": [{k: v for k, v in note.items() if k != "raw"} for note in payload["sources"]],
        "failures": payload.get("failures", []),
    }
    directory = (Path(get_settings().storage_dir) / "exports").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    basename = f"xhs-report-u{task.user_id}-{task.created_at:%Y%m%d-%H%M%S}-t{task.id}"
    outputs = {"html": render_report(document), "json": json.dumps(document, ensure_ascii=False, indent=2)}
    files = []
    for extension, content in outputs.items():
        name = f"{basename}.{extension}"
        atomic_write(directory / name, content)
        files.append({"file_name": name, "download_url": f"/api/files/exports/{name}"})
    return {"report_directory": str(directory), "report_files": files, "export_error": None}
