import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException

from src.api.schemas import (
    RunTaskRequest,
    RunTaskResponse,
    SubmitTaskResponse,
    TaskResultResponse,
    TaskStatusResponse,
)
from src.sim import AtomisticSimulationService


SERVICE: AtomisticSimulationService | None = None
API_KEY: str | None = None


def _config_path() -> str:
    return os.environ.get("ATOMBIT_SIM_CONFIG", "configs/sim.server.example.yaml")


def _load_api_key(service: AtomisticSimulationService) -> str | None:
    env_api_key = os.environ.get("ATOMBIT_SIM_API_KEY", "").strip()
    if env_api_key:
        return env_api_key
    return service.api_key()


def _check_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global API_KEY, SERVICE
    SERVICE = AtomisticSimulationService(_config_path())
    API_KEY = _load_api_key(SERVICE)
    yield


app = FastAPI(
    title="Atomistic Simulation API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/service_info")
def service_info(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    _check_api_key(x_api_key)
    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")
    return SERVICE.service_info()


@app.get("/backends")
def backends(x_api_key: str | None = Header(default=None)) -> list[dict[str, Any]]:
    _check_api_key(x_api_key)
    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")
    return SERVICE.list_backends()


@app.post("/run_task", response_model=RunTaskResponse)
def run_task(request: RunTaskRequest, x_api_key: str | None = Header(default=None)) -> RunTaskResponse:
    _check_api_key(x_api_key)
    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")

    payload = request.model_dump()
    try:
        execution_mode = SERVICE.resolve_execution_mode(payload)
        if execution_mode != "sync":
            raise HTTPException(
                status_code=400,
                detail="Use /submit_task for async execution.",
            )
        result = SERVICE.run_task_sync(payload)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RunTaskResponse(status="ok", result=result)


@app.post("/submit_task", response_model=SubmitTaskResponse)
def submit_task(request: RunTaskRequest, x_api_key: str | None = Header(default=None)) -> SubmitTaskResponse:
    _check_api_key(x_api_key)
    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")

    payload = request.model_dump()
    try:
        execution_mode = SERVICE.resolve_execution_mode(payload)
        if execution_mode != "async":
            raise HTTPException(
                status_code=400,
                detail="Use /run_task for sync execution.",
            )
        task = SERVICE.submit_task(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SubmitTaskResponse(status="accepted", task=task)


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task(task_id: str, x_api_key: str | None = Header(default=None)) -> TaskStatusResponse:
    _check_api_key(x_api_key)
    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")

    try:
        task = SERVICE.get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found.") from exc

    return TaskStatusResponse(status="ok", task=task)


@app.get("/tasks/{task_id}/result", response_model=TaskResultResponse)
def get_task_result(task_id: str, x_api_key: str | None = Header(default=None)) -> TaskResultResponse:
    _check_api_key(x_api_key)
    if SERVICE is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")

    try:
        task = SERVICE.get_task_result(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Task not found.") from exc

    return TaskResultResponse(status="ok", task=task)
