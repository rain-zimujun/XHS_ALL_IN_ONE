from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class CreativeSearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    account_id: int = Field(gt=0)
    keyword: str = Field(min_length=1, max_length=120)
    limit: Literal[20, 30, 50] = 20
    sort_type_choice: int = Field(default=2, ge=0, le=4)
    note_time: int = Field(default=2, ge=0, le=3)
    note_type: int = Field(default=2, ge=0, le=2)


class CreativeAnalysisOptions(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    model_id: int = Field(gt=0)
    idea_count: Literal[1, 3, 5] = 3
    audience: str = Field(default="关注 AI 与服装设计的从业者", max_length=300)
    instruction: str = Field(default="", max_length=2000)


class CreativeCollectRequest(BaseModel):
    search_task_id: int = Field(gt=0)
    note_ids: list[str] = Field(min_length=3, max_length=50)
    analysis: CreativeAnalysisOptions | None = None


class CreativeAnalyzeRequest(CreativeAnalysisOptions):
    collection_task_id: int = Field(gt=0)


ShortText = Annotated[str, Field(min_length=1, max_length=1000)]


class CreativeImagePlan(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    purpose: ShortText
    composition: ShortText
    prompt: str = Field(min_length=1, max_length=3000)
    negative_prompt: ShortText


class CreativeIdea(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=80)
    alternative_titles: list[ShortText] = Field(min_length=3, max_length=3)
    body: str = Field(min_length=1, max_length=6000)
    tags: list[ShortText] = Field(min_length=1, max_length=10)
    reason: ShortText
    source_note_ids: list[ShortText] = Field(min_length=2, max_length=50)
    cover_text: ShortText
    image_plan: list[CreativeImagePlan] = Field(min_length=1, max_length=6)


class CreativeAnalysisResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=4000)
    patterns: list[ShortText] = Field(min_length=1, max_length=8)
    ideas: list[CreativeIdea] = Field(min_length=1, max_length=5)
