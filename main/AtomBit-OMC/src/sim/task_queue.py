import queue
import threading
import uuid
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskQueue:
    def __init__(self, worker_fn, max_tasks: int = 1000):
        self.worker_fn = worker_fn
        self.max_tasks = int(max_tasks)
        self.tasks: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.queue: queue.Queue[str] = queue.Queue()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _worker_loop(self) -> None:
        while True:
            task_id = self.queue.get()
            try:
                self._run_task(task_id)
            finally:
                self.queue.task_done()

    def _run_task(self, task_id: str) -> None:
        with self.lock:
            task = self.tasks[task_id]
            task["status"] = "running"
            task["started_at"] = _utcnow()

        try:
            result = self.worker_fn(task["request"])
            with self.lock:
                task["status"] = "completed"
                task["completed_at"] = _utcnow()
                task["result"] = result
        except Exception as exc:
            with self.lock:
                task["status"] = "failed"
                task["completed_at"] = _utcnow()
                task["error"] = str(exc)

    def _evict_if_needed(self) -> None:
        if len(self.tasks) < self.max_tasks:
            return
        removable = [
            task_id
            for task_id, task in self.tasks.items()
            if task["status"] in {"completed", "failed"}
        ]
        for task_id in removable[: max(1, len(self.tasks) - self.max_tasks + 1)]:
            del self.tasks[task_id]

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        with self.lock:
            self._evict_if_needed()
            self.tasks[task_id] = {
                "task_id": task_id,
                "status": "pending",
                "created_at": _utcnow(),
                "started_at": None,
                "completed_at": None,
                "request": request,
                "result": None,
                "error": None,
            }
        self.queue.put(task_id)
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self.lock:
            if task_id not in self.tasks:
                raise KeyError(task_id)
            task = self.tasks[task_id]
            return {
                "task_id": task["task_id"],
                "status": task["status"],
                "created_at": task["created_at"],
                "started_at": task["started_at"],
                "completed_at": task["completed_at"],
                "error": task["error"],
            }

    def get_result(self, task_id: str) -> dict[str, Any]:
        with self.lock:
            if task_id not in self.tasks:
                raise KeyError(task_id)
            task = self.tasks[task_id]
            return {
                "task_id": task["task_id"],
                "status": task["status"],
                "result": task["result"],
                "error": task["error"],
                "created_at": task["created_at"],
                "started_at": task["started_at"],
                "completed_at": task["completed_at"],
            }
