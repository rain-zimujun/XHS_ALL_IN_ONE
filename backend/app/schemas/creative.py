from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CreativeSearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    account_id: int = Field(gt=0)
    keyword: str = Field(min_length=1, max_length=120)
    limit: Literal[20, 30, 50] = 20
    sort_type_choice: int = Field(default=2, ge=0, le=4)
    note_time: int = Field(default=2, ge=0, le=3)
    note_type: int = Field(default=2, ge=0, le=2)


class CreativeCollectRequest(BaseModel):
    search_task_id: int = Field(gt=0)
    note_ids: list[str] = Field(min_length=3, max_length=50)
