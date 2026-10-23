"""Separate trailing topic lists from publishable copy without rewriting prose."""
from __future__ import annotations

import re
import unicodedata

from backend.app.schemas.creative import COPY_TAG_COUNT


TOPIC = r"[#＃]([^\s#＃]+)"
TOPIC_LINE = re.compile(rf"(?:{TOPIC}[#＃]?\s*)+")


def validate_generated_tags(tags: list[str]) -> None:
    """Generation-only checks; historical copy keeps its original tag count."""
    normalized = [unicodedata.normalize("NFKC", tag).casefold() for tag in tags]
    if (len(tags) != COPY_TAG_COUNT or len(set(normalized)) != COPY_TAG_COUNT
            or any(not tag or "#" in tag or any(c.isspace() for c in tag) for tag in normalized)):
        raise ValueError("tags 必须是与本版最终正文最相关的 6 个不同话题，按相关性从高到低排列，不含井号或空白；请根据正文重新选择，不要截取或填入无关标签。")


def normalize_note_copy(body: str, tags: list[str]) -> tuple[str, list[str]]:
    # Only standalone topic lists at the end are moved. Inline hashtags,
    # headings, emoji, numbering, and the actual prose remain untouched.
    lines = body.rstrip().split("\n")
    trailing_tags: list[str] = []
    while lines:
        line = lines[-1].strip()
        if not line:
            lines.pop()
        elif TOPIC_LINE.fullmatch(line):
            trailing_tags[0:0] = re.findall(TOPIC, line)
            lines.pop()
        else:
            break
    unique = []
    for tag in [*tags, *trailing_tags]:
        tag = tag.strip().strip("#＃").removesuffix("[话题]").strip()
        if tag and tag not in unique:
            unique.append(tag)
    return "\n".join(lines).strip(), unique


def normalize_result_copy(result):
    """Return a copy; malformed model output is left for schema validation."""
    if not isinstance(result, dict) or not isinstance(result.get("ideas"), list):
        return result

    def normalize_variant(variant):
        if not isinstance(variant, dict):
            return variant
        body, tags = variant.get("body"), variant.get("tags")
        if not isinstance(body, str) or not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            return variant
        body, tags = normalize_note_copy(body, tags)
        return {**variant, "body": body, "tags": tags}

    ideas = []
    for idea in result["ideas"]:
        if isinstance(idea, dict) and isinstance(idea.get("variants"), list):
            ideas.append({**idea, "variants": [normalize_variant(v) for v in idea["variants"]]})
        else:
            ideas.append(normalize_variant(idea))
    return {**result, "ideas": ideas}


EDITORIAL_HEADER = re.compile(
    r"^\s*(?:正文文案|发布文案|配图建议|生图提示词|负向提示词|推荐理由|参考样本|参考来源)\s*[:：]",
    re.MULTILINE,
)
ASSISTANT_PREFACE = re.compile(r"^\s*(?:以下是|下面是)(?:为你|为您)?(?:生成|整理|撰写|准备)的.{0,20}(?:文案|笔记)")
PLACEHOLDER = re.compile(r"(?:【|\[)(?:待填写|待补充|请填写|请替换)[^\]】\n]*(?:】|\])")


def has_editorial_leak(body: str) -> bool:
    # Do not ban subject words such as AI, design or tutorial. These checks
    # only catch explicit editor-facing labels, assistant prefaces and blanks.
    return bool(EDITORIAL_HEADER.search(body) or ASSISTANT_PREFACE.search(body) or PLACEHOLDER.search(body))
