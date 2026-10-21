"""Quality rubrics shared by creative agents and the final judge.

The rubric is deliberately deterministic. A model may provide scores and
comments later, but pass/fail thresholds and blocking criteria stay in code so
the pipeline cannot silently lower its own quality bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Score = Field(ge=0, le=10)
ReviewKind = Literal["topic", "title", "copy", "final"]


@dataclass(frozen=True)
class RubricCriterion:
    key: str
    label: str
    weight: float
    minimum: float = 0.0
    blocking: bool = False


TOPIC_RUBRIC = (
    RubricCriterion("reader_value", "读者价值", 0.30, 6.0, True),
    RubricCriterion("specificity", "具体程度", 0.25, 6.0, True),
    RubricCriterion("fit", "与搜索和受众的匹配度", 0.20, 6.0, True),
    RubricCriterion("novelty", "差异化", 0.15, 5.0),
    RubricCriterion("evidence", "样本依据充分度", 0.10, 5.0),
)

TITLE_RUBRIC = (
    RubricCriterion("hook", "点击吸引力", 0.25, 6.0, True),
    RubricCriterion("specificity", "具体程度", 0.20, 6.0, True),
    RubricCriterion("reader_benefit", "读者收益", 0.20, 5.5, True),
    RubricCriterion("relevance", "关键词与受众匹配度", 0.15, 6.0, True),
    RubricCriterion("difference", "与其他标题的差异化", 0.10, 5.0),
    RubricCriterion("naturalness", "自然度与可读性", 0.10, 5.0),
)

COPY_RUBRIC = (
    RubricCriterion("opening", "前两句吸引力", 0.20, 6.0, True),
    RubricCriterion("value", "具体信息与读者价值", 0.25, 6.0, True),
    RubricCriterion("audience_fit", "目标读者匹配度", 0.15, 6.0, True),
    RubricCriterion("platform_voice", "小红书语气与可读性", 0.15, 5.5, True),
    RubricCriterion("save_value", "收藏或转发价值", 0.15, 5.0),
    RubricCriterion("publishability", "直接发布可用度", 0.10, 6.0, True),
)

FINAL_RUBRIC = (
    RubricCriterion("title_body_fit", "标题与正文一致性", 0.30, 6.0, True),
    RubricCriterion("promise_support", "正文是否兑现标题承诺", 0.25, 6.0, True),
    RubricCriterion("image_fit", "正文与配图匹配度", 0.20, 5.0),
    RubricCriterion("tag_quality", "标签相关性与去重", 0.10, 5.0),
    RubricCriterion("safety", "事实、原创和素材边界", 0.15, 7.0, True),
)

RUBRICS: dict[ReviewKind, tuple[RubricCriterion, ...]] = {
    "topic": TOPIC_RUBRIC,
    "title": TITLE_RUBRIC,
    "copy": COPY_RUBRIC,
    "final": FINAL_RUBRIC,
}


class QualityScore(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    criterion: str = Field(min_length=1, max_length=60)
    score: float = Score
    comment: str = Field(min_length=1, max_length=600)


class QualityReview(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    kind: ReviewKind
    scores: list[QualityScore] = Field(min_length=1, max_length=12)
    overall_score: float = Score
    passed: bool
    blocking_issues: list[str] = Field(default_factory=list, max_length=12)
    revision_instructions: list[str] = Field(default_factory=list, max_length=12)
    reviewer_notes: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def score_is_consistent(self) -> "QualityReview":
        if self.passed and self.blocking_issues:
            raise ValueError("A passed review cannot contain blocking issues")
        return self


def rubric_for(kind: ReviewKind) -> tuple[RubricCriterion, ...]:
    return RUBRICS[kind]


def weighted_score(kind: ReviewKind, scores: list[QualityScore]) -> float:
    criteria = {item.key: item for item in rubric_for(kind)}
    weighted = sum(criteria[item.criterion].weight * item.score for item in scores if item.criterion in criteria)
    present_weight = sum(criteria[item.criterion].weight for item in scores if item.criterion in criteria)
    if present_weight <= 0:
        return 0.0
    return round(weighted / present_weight, 2)


def assess_review(kind: ReviewKind, scores: list[QualityScore], blocking_issues: list[str] | None = None) -> QualityReview:
    criteria = {item.key: item for item in rubric_for(kind)}
    blocking = list(blocking_issues or [])
    present = {item.criterion for item in scores}
    blocking.extend(f"缺少评分项：{criteria[key].label}" for key in criteria.keys() - present)
    for score in scores:
        criterion = criteria.get(score.criterion)
        if criterion and criterion.blocking and score.score < criterion.minimum:
            blocking.append(f"{criterion.label}低于 {criterion.minimum:g} 分")
    overall = weighted_score(kind, scores)
    passed = not blocking and overall >= 6.5
    return QualityReview(
        kind=kind,
        scores=scores,
        overall_score=overall,
        passed=passed,
        blocking_issues=list(dict.fromkeys(blocking)),
        revision_instructions=[] if passed else ["优先修复所有阻断项，再重新生成并评审。"],
    )
