from typing import Any

from pydantic import BaseModel, Field


class StructurePayload(BaseModel):
    format: str = Field(default="cif")
    text: str = Field(min_length=1)


class BackendSelector(BaseModel):
    profile: str = Field(default="default")


class RunTaskRequest(BaseModel):
    task_type: str = Field(min_length=1)
    execution_mode: str | None = Field(default=None)
    structure: StructurePayload
    backend: BackendSelector = Field(default_factory=BackendSelector)
    task_config: dict[str, Any] = Field(default_factory=dict)


class RunTaskResponse(BaseModel):
    status: str
    result: dict[str, Any]


class SubmitTaskResponse(BaseModel):
    status: str
    task: dict[str, Any]


class TaskStatusResponse(BaseModel):
    status: str
    task: dict[str, Any]


class TaskResultResponse(BaseModel):
    status: str
    task: dict[str, Any]
