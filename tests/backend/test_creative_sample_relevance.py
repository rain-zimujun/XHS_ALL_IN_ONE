import copy

import pytest

from backend.app.services.creative_agents.sample_relevance import (
    WEAK_BODY_LIMIT, classify_sample, filter_sample_sources, require_minimum_samples,
)
from backend.app.services.creative_analysis import CreativeError


KEYWORD = "AI 服装"
AUDIENCE = "电商、服装行业、AI、时尚行业、营销行业、美工、后期从业者"


def note(note_id, title, content):
    return {"note_id": note_id, "title": title, "content": content, "likes": 1}


def test_classifies_high_weak_and_unrelated_samples():
    high = note("high", "AI服装设计工作流", "用生成式AI做服装款式设计，整理了从提示词到电商主图的案例。")
    weak = note("weak", "AI绘图提示词记录", "分享AI绘图工具使用心得。")
    unrelated = note("unrelated", "周末咖啡探店", "新开的咖啡店很安静，甜品也不错。")

    assert classify_sample(high, keyword=KEYWORD, audience=AUDIENCE)["tier"] == "high"
    assert classify_sample(weak, keyword=KEYWORD, audience=AUDIENCE)["tier"] == "weak"
    result = classify_sample(unrelated, keyword=KEYWORD, audience=AUDIENCE)
    assert result["tier"] == "excluded"
    assert result["note_id"] == "unrelated"
    assert "正文" not in str(result)


def test_filter_keeps_high_trims_weak_and_does_not_mutate_original_sources():
    body = "AI绘图工具的使用记录。" + "补充记录。" * 200
    originals = [
        note("high", "AI服装设计工作流", "AI生成服装设计，再做电商主图。"),
        note("weak", "AI绘图工具记录", body),
        note("unrelated", "咖啡探店", "周末去喝咖啡。"),
    ]
    before = copy.deepcopy(originals)

    selected, decisions = filter_sample_sources(originals, keyword=KEYWORD, audience=AUDIENCE)

    assert originals == before
    assert [source["note_id"] for source in selected] == ["high", "weak"]
    assert len(selected[0]["content"]) == len(originals[0]["content"])
    assert len(selected[1]["content"]) == WEAK_BODY_LIMIT
    assert selected[1]["content"].startswith(body[:350])
    assert selected[1]["content"].endswith(body[-(WEAK_BODY_LIMIT - 350):])
    assert [item["tier"] for item in decisions] == ["high", "weak", "excluded"]


def test_minimum_sample_error_reports_tiers_without_including_source_text():
    selected, decisions = filter_sample_sources(
        [note("n1", "AI服装搭配", "AI做服装设计。"), note("n2", "咖啡", "私密正文")],
        keyword=KEYWORD, audience=AUDIENCE,
    )
    with pytest.raises(CreativeError, match="至少需要 3 篇") as caught:
        require_minimum_samples(selected, decisions)
    assert caught.value.code == "sample_relevance_insufficient"
    assert "私密正文" not in str(caught.value)


def test_filter_metadata_is_json_serializable_and_preserves_source_ids():
    originals = [
        note("id-a", "AI服装视觉方案", "生成式AI服装模特图与商品主图方案。"),
        note("id-b", "AI穿搭灵感", "AI辅助服装搭配灵感。"),
        note("id-c", "AI电商服装设计", "AI服装设计工作流案例。"),
    ]
    selected, decisions = filter_sample_sources(originals, keyword=KEYWORD, audience=AUDIENCE)
    require_minimum_samples(selected, decisions)
    assert [item["note_id"] for item in selected] == ["id-a", "id-b", "id-c"]
    assert all(item["note_id"] for item in decisions)
    import json
    json.dumps(decisions, ensure_ascii=False)
