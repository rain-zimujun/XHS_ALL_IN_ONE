"""Deterministic relevance filtering for collected sample notes."""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from backend.app.services.creative_analysis import CreativeError


FILTER_VERSION = "sample-relevance-v1"
WEAK_BODY_LIMIT = 500

_TOPIC_GROUPS = {
    "ai": ("人工智能", "生成式", "生成图", "生图", "提示词", "大模型", "智能设计", "视觉生成", "扩散模型", "机器学习", "ai绘画", "ai工具", "ai"),
    "fashion": ("服装", "穿搭", "面料", "版型", "廓形", "服饰", "款式", "时装", "时尚", "潮流", "鞋服", "女装", "男装", "服装设计"),
    "commercial_visual": ("电商", "电商视觉", "商品图", "营销素材", "广告图", "设计提案", "美工", "后期", "视觉方案", "模特图", "主图", "详情页", "商品摄影", "服装摄影", "营销"),
}
_UTILITY_TERMS = ("案例", "教程", "方法", "技巧", "流程", "工作流", "工具", "模板", "方案", "对比", "实测", "复盘", "设计", "创作")
_AUDIENCE_GENERIC = ("行业", "从业者", "工作者", "人群")


def _contains(text: str, term: str) -> bool:
    if term == "ai":
        return re.search(r"(?<![a-z0-9])ai(?![a-z0-9])", text, re.IGNORECASE) is not None
    return term.lower() in text


def _terms(value: str) -> list[str]:
    return [part.strip().lower() for part in re.split(r"[,，、;；|/\s]+", value) if len(part.strip()) >= 2]


def classify_sample(source: dict[str, Any], *, keyword: str, audience: str) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {"note_id": None, "tier": "excluded", "score": 0.0, "matched_terms": [],
                "reason": "样本结构无效，未进入模型输入。"}
    title = source.get("title") if isinstance(source.get("title"), str) else ""
    body = source.get("content") if isinstance(source.get("content"), str) else ""
    text = f"{title}\n{body[:3000]}".lower()
    matched_groups = [group for group, terms in _TOPIC_GROUPS.items() if any(_contains(text, term) for term in terms)]
    matched_keyword = [term for term in _terms(keyword) if _contains(text, term)]
    audience_terms = [term for term in _terms(audience) if not any(generic in term for generic in _AUDIENCE_GENERIC)]
    matched_audience = [term for term in audience_terms if _contains(text, term)]
    matched_utility = [term for term in _UTILITY_TERMS if term in text]

    # The score is a routing signal, not a quality score. A single broad AI or
    # fashion match can only be weak; high relevance needs multiple domains.
    score = min(1.0, 0.30 * len(matched_groups) + 0.15 * bool(matched_keyword)
                + 0.15 * min(1, len(matched_audience)) + 0.10 * bool(matched_utility))
    if len(matched_groups) >= 2 and score >= 0.65:
        tier = "high"
        reason = "同时涉及多个相关主题，可作为主要分析样本。"
    elif matched_groups and score >= 0.30:
        tier = "weak"
        reason = "命中部分主题或受众，但关联证据有限，仅作辅助参考。"
    else:
        tier = "excluded"
        reason = "未发现与指定主题及受众相关的有效线索。"
    matched_terms = list(dict.fromkeys([*matched_groups, *matched_keyword, *matched_audience, *matched_utility]))[:12]
    return {"note_id": source.get("note_id"), "tier": tier, "score": round(score, 3),
            "matched_terms": matched_terms, "reason": reason}


def filter_sample_sources(sources: list[dict[str, Any]], *, keyword: str, audience: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return model-input copies and lightweight decisions; never mutate originals."""
    selected, decisions = [], []
    for source in sources:
        decision = classify_sample(source, keyword=keyword, audience=audience)
        decisions.append(decision)
        if decision["tier"] == "excluded":
            continue
        prepared = deepcopy(source)
        prepared["relevance_tier"] = decision["tier"]
        if decision["tier"] == "weak":
            content = prepared.get("content")
            if isinstance(content, str) and len(content) > WEAK_BODY_LIMIT:
                head = 350
                tail = WEAK_BODY_LIMIT - head
                prepared["content"] = content[:head] + content[-tail:]
                prepared["relevance_original_body_chars"] = len(content)
                prepared["relevance_body_truncated"] = True
        selected.append(prepared)
    return selected, decisions


def require_minimum_samples(selected: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> None:
    if len(selected) >= 3:
        return
    high = sum(item["tier"] == "high" for item in decisions)
    weak = sum(item["tier"] == "weak" for item in decisions)
    excluded = sum(item["tier"] == "excluded" for item in decisions)
    raise CreativeError(
        f"相关性筛选后仅剩 {len(selected)} 篇可用样本（高相关 {high}、弱相关 {weak}、已排除 {excluded}），至少需要 3 篇。请调整关键词或增加采集数量。",
        "sample_relevance_insufficient",
    )
