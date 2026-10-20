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
    variant_count: Literal[2, 3] = 3
    audience: str = Field(default="电商、服装行业、AI、时尚行业、营销行业、美工、后期从业者", max_length=300)
    instruction: str = Field(default="符合小红书爆款文案的调性，文案尽可能精简、抓眼球", max_length=2000)


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


class CreativeVariant(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    label: str = Field(min_length=1, max_length=40, description="该版本的表达角度或写法，需符合样本风格")
    body: str = Field(min_length=1, max_length=6000, description="给小红书读者看的纯正文成稿；默认自然分享作品或灵感，不写给发布者的指导。不得包含标题、话题标签、配图/生图说明、分析注释或待填占位符。")
    tags: list[ShortText] = Field(min_length=1, max_length=10, description="独立的话题名称，不带 #，不重复写进 body")
    cover_text: ShortText
    image_plan: list[CreativeImagePlan] = Field(min_length=1, max_length=6)


class CreativeIdea(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=80)
    reason: ShortText
    source_note_ids: list[ShortText] = Field(min_length=2, max_length=50)
    variants: list[CreativeVariant] = Field(min_length=2, max_length=3)


class CreativeAnalysisResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=4000)
    patterns: list[ShortText] = Field(min_length=1, max_length=8)
    style_summary: str = Field(min_length=1, max_length=2000, description="从本批正文归纳的开头、语气、分段、表情、话题和互动结尾特点")
    ideas: list[CreativeIdea] = Field(min_length=1, max_length=5)
