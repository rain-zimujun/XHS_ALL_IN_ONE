import copy

import pytest

from backend.app.services.creative_copy import has_editorial_leak, normalize_note_copy, normalize_result_copy


@pytest.mark.parametrize("body,tags,expected_body,expected_tags", [
    ("雾白配深蓝。\n\n#AI服装 #设计灵感", ["AI服装", "设计灵感"], "雾白配深蓝。", ["AI服装", "设计灵感"]),
    ("海风感🌊\n\n#AI服装\n#设计灵感\n", ["#AI服装", "AI服装"], "海风感🌊", ["AI服装", "设计灵感"]),
    ("概念设计。\n＃AI服装[话题]＃ #设计灵感[话题]#", ["＃AI服装"], "概念设计。", ["AI服装", "设计灵感"]),
    ("# 设计灵感\n① 弧形领口\n② 雾白裙摆✨", ["服装"], "# 设计灵感\n① 弧形领口\n② 雾白裙摆✨", ["服装"]),
    ("喜欢 #AI服装 带来的想象空间。", ["AI服装"], "喜欢 #AI服装 带来的想象空间。", ["AI服装"]),
    ("#不是结尾的话题\n这是完整的一句说明。", [], "#不是结尾的话题\n这是完整的一句说明。", []),
    ("#AI服装", ["AI服装"], "", ["AI服装"]),
    ("正文。\n\n#服装#设计", ["服装"], "正文。", ["服装", "设计"]),
])
def test_topic_lists_are_separated_without_rewriting_prose(body, tags, expected_body, expected_tags):
    assert normalize_note_copy(body, tags) == (expected_body, expected_tags)


@pytest.mark.parametrize("body", [
    "正文文案：海风吹过的感觉。",
    "海风吹过的感觉。\n配图建议：主体居中",
    "以下是为你生成的文案：海风感。",
    "材质是【待填写面料】。",
    "海风感。\n生图提示词：fashion portrait",
])
def test_explicit_editorial_leaks_are_detected(body):
    assert has_editorial_leak(body)


@pytest.mark.parametrize("body", [
    "AI 服装概念设计，裙摆像海浪🌊\n你更喜欢雾白还是深蓝？",
    "不需要复杂的提示词，也能聊聊衣服的线条。",
    "【海风】是这组设计的灵感。",
    "① 灰蓝色\n② 银色细节\n适合喜欢冷色调的人。",
])
def test_subject_words_emoji_and_reader_facing_content_are_not_stripped(body):
    assert not has_editorial_leak(body)
    assert normalize_note_copy(body, ["服装"])[0] == body


def test_normalization_does_not_mutate_saved_result_or_drop_other_fields():
    original = {"ideas": [{"title": "海风", "variants": [{
        "body": "雾白配深蓝。\n#服装", "tags": ["#服装"], "image_plan": [{"prompt": "#not-a-topic"}],
    }]}]}
    before = copy.deepcopy(original)
    normalized = normalize_result_copy(original)
    assert original == before
    assert normalized["ideas"][0]["variants"][0]["body"] == "雾白配深蓝。"
    assert normalized["ideas"][0]["variants"][0]["image_plan"] == before["ideas"][0]["variants"][0]["image_plan"]
    assert normalize_result_copy(normalized) == normalized


def test_generated_tags_are_cleaned_deduplicated_and_capped_without_rejecting_copy():
    result = {"ideas": [{"variants": [{"body": "可直接发布的正文。", "tags": [
        "#AI服装", "ＡＩ服装", "设计", "  配色  ", "含 空格", "＃廓形", "风格", "材质", "细节", "多余",
    ]}]}]}
    normalized = normalize_result_copy(result)
    assert normalized["ideas"][0]["variants"][0]["tags"] == ["AI服装", "设计", "配色", "廓形", "风格", "材质", "细节", "多余"]


@pytest.mark.parametrize("value", [None, [], {"ideas": None}, {"ideas": [None]}, {"ideas": [{"variants": [{"body": 1, "tags": [None]}]}]}])
def test_malformed_output_remains_for_schema_validation(value):
    assert normalize_result_copy(value) == value
