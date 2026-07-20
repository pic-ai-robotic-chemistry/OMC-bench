# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
MACE Crystal Structure Relaxation MCP Server

Wraps the MACE interatomic potential model as an MCP (Model Context Protocol)
service with an additional REST API.  Given one or more crystal structures in
CIF format, the server relaxes them with an ASE geometry optimiser (BFGS,
FIRE, LBFGS, …) backed by a MACE calculator and returns the optimised
structures as CIF strings.

Usage
-----
    python mcp_server.py                          # stdio transport, default config.yaml
    python mcp_server.py --config my_config.yaml  # custom config
    python mcp_server.py --transport sse          # HTTP/SSE mode (REST API enabled)
    python mcp_server.py --transport sse --port 8080

MCP tools exposed
-----------------
    optimize_crystal_structures   – relax structures, return CIF strings
    save_optimized_structures     – relax structures, save CIF files to disk

REST API (--transport sse)
--------------------------
    POST  /api/optimize                    submit relaxation task
    GET   /api/tasks/{task_id}             query task status
    GET   /api/tasks/{task_id}/download    download ZIP of optimised CIFs
"""

import argparse
import io
import json
import os
import queue
import re
import sys
import textwrap
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------
_MAX_STRUCTURES_PER_REQUEST = 100   # hard cap on structures per request
_MAX_CIF_STRING_BYTES = 1 * 1024 * 1024   # 1 MB per CIF string
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024      # 50 MB ZIP upload limit
_MAX_STEPS = 5000                   # prevent runaway CPU via high step count
_MAX_TASK_STORE = 1000              # evict oldest tasks beyond this limit
_VALID_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

# Rate limiting: max requests per IP within the rolling window
_RATE_LIMIT_REQUESTS = 20
_RATE_LIMIT_WINDOW = 60  # seconds

# API key auth (set env var MACE_API_KEY to enable; empty = disabled)
_API_KEY: str = os.environ.get("MACE_API_KEY", "").strip()

# Rate limiter state  {ip: deque of request timestamps}
_rate_counters: dict = {}
_rate_lock = threading.Lock()


def _check_api_key(request: Request) -> Optional[JSONResponse]:
    """Return 401 if API key auth is enabled and the request fails."""
    if not _API_KEY:
        return None
    provided = (
        request.headers.get("X-API-Key", "")
        or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    )
    if provided != _API_KEY:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return None


def _check_rate_limit(request: Request) -> Optional[JSONResponse]:
    """Return 429 if the client IP exceeds the rate limit."""
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_lock:
        from collections import deque as _deque
        if ip not in _rate_counters:
            _rate_counters[ip] = _deque()
        dq = _rate_counters[ip]
        while dq and now - dq[0] > _RATE_LIMIT_WINDOW:
            dq.popleft()
        if len(dq) >= _RATE_LIMIT_REQUESTS:
            return JSONResponse(
                {"error": f"Rate limit exceeded. Max {_RATE_LIMIT_REQUESTS} requests "
                          f"per {_RATE_LIMIT_WINDOW}s."},
                status_code=429,
                headers={"Retry-After": str(_RATE_LIMIT_WINDOW)},
            )
        dq.append(now)
    return None


def _guard(request: Request) -> Optional[JSONResponse]:
    """Run all per-request security checks. Returns error response or None."""
    return _check_api_key(request) or _check_rate_limit(request)

# ---------------------------------------------------------------------------
# MCP import
# ---------------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "The 'mcp' package is required.  Install it with:\n"
        "    pip install mcp"
    ) from exc

# ---------------------------------------------------------------------------
# ASE imports
# ---------------------------------------------------------------------------
try:
    import ase.io
    from ase import Atoms
    from ase.optimize import BFGS, FIRE, LBFGS
    from ase.optimize.sciopt import SciPyFminBFGS
except ImportError as exc:
    raise ImportError(
        "The 'ase' package is required.  Install it with:\n"
        "    pip install ase"
    ) from exc

# ---------------------------------------------------------------------------
# MACE calculator import
# ---------------------------------------------------------------------------
try:
    from mace.calculators import MACECalculator, mace_mp, mace_off
except ImportError as exc:
    raise ImportError(
        "The 'mace-torch' package is required.  Install it with:\n"
        "    pip install mace-torch"
    ) from exc

# ---------------------------------------------------------------------------
# Optional pymatgen for high-quality CIF output
# ---------------------------------------------------------------------------
try:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.cif import CifWriter
    _PYMATGEN_AVAILABLE = True
except ImportError:
    _PYMATGEN_AVAILABLE = False

# ===========================================================================
# Supported optimisers
# ===========================================================================

_OPTIMIZERS = {
    "BFGS": BFGS,
    "FIRE": FIRE,
    "LBFGS": LBFGS,
}

# ===========================================================================
# Global model state (loaded once at startup)
# ===========================================================================

_calculator: Optional[MACECalculator] = None
_config: dict = {}


def _log(msg: str) -> None:
    print(f"[MACE {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_model(config_path: str) -> None:
    """Load MACE model from checkpoint or pretrained weights."""
    global _calculator, _config

    with open(config_path, "r") as f:
        _config = yaml.safe_load(f)

    model_cfg = _config.get("model", {})
    model_path = model_cfg.get("path")
    model_name = model_cfg.get("name", "mace-mp-0")
    device = model_cfg.get("device", "cpu")
    default_dtype = model_cfg.get("default_dtype", "float32")

    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MACE model checkpoint not found: {model_path}\n"
                "Set 'model.path' in config.yaml to a valid .model file, "
                "or set 'model.name' to a pretrained model name."
            )
        _log(f"Loading MACE model from local file: {model_path}")
        _calculator = MACECalculator(
            model_paths=model_path,
            device=device,
            default_dtype=default_dtype,
        )
    else:
        # Use a foundation model from mace-torch
        _log(f"Loading pretrained MACE model: {model_name} (device={device})")
        name_lower = model_name.lower().replace("-", "").replace("_", "")
        if "off" in name_lower:
            _calculator = mace_off(model=model_cfg.get("size", "medium"), device=device,
                                   default_dtype=default_dtype, return_raw_model=False)
        else:
            # Default: MACE-MP-0 (universal potential for periodic solids)
            _calculator = mace_mp(
                model=model_cfg.get("size", "small"),
                device=device,
                default_dtype=default_dtype,
            )

    _log("MACE calculator ready.")


# ===========================================================================
# CIF helpers
# ===========================================================================

def _cif_string_to_atoms(cif_str: str, index: int = 0) -> Atoms:
    """Parse a CIF string into an ASE Atoms object."""
    buf = io.StringIO(cif_str)
    atoms = ase.io.read(buf, format="cif", index=index)
    return atoms


def _atoms_to_cif(atoms: Atoms, structure_id: int = 0) -> str:
    """Convert an ASE Atoms object to a CIF string.

    Uses pymatgen when available for higher-quality output; falls back to
    the ASE built-in CIF writer otherwise.
    """
    if _PYMATGEN_AVAILABLE:
        adaptor = AseAtomsAdaptor()
        structure = adaptor.get_structure(atoms)
        cif_str = str(CifWriter(structure))
        return f"# MACE optimised structure #{structure_id}\n" + cif_str

    # ASE fallback
    buf = io.StringIO()
    ase.io.write(buf, atoms, format="cif")
    return f"# MACE optimised structure #{structure_id}\n" + buf.getvalue()


# ===========================================================================
# Core relaxation logic
# ===========================================================================

def _relax_atoms(
    atoms: Atoms,
    optimizer_name: str = "BFGS",
    fmax: float = 0.05,
    steps: int = 500,
    logfile: str = "-",
) -> tuple[Atoms, dict]:
    """Attach MACE calculator and run geometry optimisation.

    Parameters
    ----------
    atoms : ase.Atoms
        Input structure.
    optimizer_name : str
        Name of the ASE optimiser: "BFGS", "FIRE", or "LBFGS".
    fmax : float
        Convergence criterion (eV/Angstrom).
    steps : int
        Maximum number of optimiser steps.
    logfile : str
        Path for optimiser log or "-" for stdout.

    Returns
    -------
    atoms : ase.Atoms
        Relaxed structure (mutated in place).
    info : dict
        Metadata: converged flag, final energy, final max force, n_steps.
    """
    # Attach a fresh calculator clone so batched calls are independent
    atoms.calc = _calculator

    optimizer_cls = _OPTIMIZERS.get(optimizer_name.upper())
    if optimizer_cls is None:
        raise ValueError(
            f"Unknown optimizer '{optimizer_name}'. "
            f"Supported: {list(_OPTIMIZERS)}"
        )

    opt = optimizer_cls(atoms, logfile=logfile)
    converged = opt.run(fmax=fmax, steps=steps)

    try:
        energy = float(atoms.get_potential_energy())
    except Exception:
        energy = None

    try:
        forces = atoms.get_forces()
        import numpy as np
        max_force = float(np.linalg.norm(forces, axis=1).max())
    except Exception:
        max_force = None

    return atoms, {
        "converged": bool(converged),
        "energy_eV": energy,
        "max_force_eV_Ang": max_force,
        "n_steps": opt.nsteps,
    }


def _relax_cif_list(
    cif_strings: list[str],
    optimizer_name: str = "BFGS",
    fmax: float = 0.05,
    steps: int = 500,
    log_prefix: str = "",
) -> list[tuple[str, dict]]:
    """Relax a batch of CIF strings.

    Returns a list of (optimised_cif_string, info_dict) tuples in the same
    order as the input.
    """
    results: list[tuple[str, dict]] = []
    n = len(cif_strings)

    for i, cif_str in enumerate(cif_strings):
        _log(f"{log_prefix}Relaxing structure {i + 1}/{n} ...")
        t0 = time.time()
        try:
            atoms = _cif_string_to_atoms(cif_str)
            atoms_relaxed, info = _relax_atoms(
                atoms,
                optimizer_name=optimizer_name,
                fmax=fmax,
                steps=steps,
                logfile="-",
            )
            cif_out = _atoms_to_cif(atoms_relaxed, structure_id=i)
            elapsed = time.time() - t0
            _log(
                f"{log_prefix}  structure {i + 1}/{n} done in {elapsed:.1f}s "
                f"| converged={info['converged']} "
                f"E={info['energy_eV']:.4f} eV "
                f"Fmax={info['max_force_eV_Ang']:.4f} eV/Ang"
            )
            results.append((cif_out, info))
        except Exception as exc:
            _log(f"{log_prefix}  structure {i + 1}/{n} FAILED: {exc}")
            results.append(("", {"error": str(exc), "converged": False}))

    return results


# ===========================================================================
# Async task queue (REST API; optimisation is CPU/GPU-bound so we use a
# single worker thread to avoid resource contention)
# ===========================================================================

_tasks: dict[str, dict[str, Any]] = {}
_tasks_lock = threading.Lock()
_task_queue: queue.Queue = queue.Queue()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_task(task_id: str) -> None:
    """Execute one relaxation task and store results in the task record."""
    with _tasks_lock:
        task = _tasks[task_id]
        task["status"] = "running"
        task["started_at"] = _utcnow()

    n = len(task["cif_files"])
    _log(f"Task {task_id[:8]} started | n_structures={n} optimizer={task['optimizer']}")

    try:
        results = _relax_cif_list(
            task["cif_files"],
            optimizer_name=task["optimizer"],
            fmax=task["fmax"],
            steps=task["steps"],
            log_prefix=f"[Task {task_id[:8]}] ",
        )

        optimised_cifs = [cif for cif, _ in results]
        infos = [info for _, info in results]

        with _tasks_lock:
            task["status"] = "completed"
            task["completed_at"] = _utcnow()
            task["optimised_cifs"] = optimised_cifs
            task["infos"] = infos

        _log(f"Task {task_id[:8]} COMPLETED")

    except Exception as exc:
        _log(f"Task {task_id[:8]} FAILED: {exc}")
        with _tasks_lock:
            task["status"] = "failed"
            task["error"] = str(exc)
            task["completed_at"] = _utcnow()


def _worker() -> None:
    """Background worker: processes tasks one at a time."""
    while True:
        task_id = _task_queue.get()
        try:
            _run_task(task_id)
        finally:
            _task_queue.task_done()


def _start_worker() -> None:
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# ===========================================================================
# MCP server definition
# ===========================================================================

mcp = FastMCP(
    name="mace-relax",
    instructions=textwrap.dedent("""\
        MACE crystal structure relaxation service.

        Given one or more crystal structures in CIF format, this service uses
        the MACE machine-learning interatomic potential to relax the structures
        with an ASE geometry optimiser (BFGS, FIRE, or LBFGS) and returns the
        optimised structures as CIF strings.

        Supported optimisers: BFGS (default), FIRE, LBFGS.
        Convergence criterion: maximum force on any atom (fmax, eV/Angstrom).
    """),
)


# ===========================================================================
# REST HTTP endpoints
#
# POST  /api/optimize                        submit task
# GET   /api/tasks/{task_id}                 query status
# GET   /api/tasks/{task_id}/download        download ZIP of optimised CIFs
# ===========================================================================

def _make_task(cif_files: list[str], optimizer: str, fmax: float, steps: int, top_n: int) -> str:
    """Create a task record, enqueue it, and return the task_id."""
    task_id = str(uuid.uuid4())
    with _tasks_lock:
        # Evict oldest completed/failed tasks when the store is full
        if len(_tasks) >= _MAX_TASK_STORE:
            evict_keys = [
                k for k, v in _tasks.items()
                if v["status"] in ("completed", "failed")
            ]
            for k in evict_keys[:max(1, len(_tasks) - _MAX_TASK_STORE + 1)]:
                del _tasks[k]

        _tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "n_structures": len(cif_files),
            "optimizer": optimizer,
            "fmax": fmax,
            "steps": steps,
            "top_n": top_n,
            "created_at": _utcnow(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "cif_files": cif_files,
            "optimised_cifs": [],
            "infos": [],
        }
    _task_queue.put(task_id)
    return task_id


@mcp.custom_route("/api/optimize", methods=["POST"])
async def api_optimize(request: Request) -> Response:
    """Submit a crystal structure relaxation task.

    Request body (JSON)::

        {
            "cif_files": ["<CIF string 1>", "<CIF string 2>", ...],
            "optimizer": "BFGS",   // optional, default "BFGS"
            "fmax": 0.05,          // optional, convergence force threshold (eV/Ang)
            "steps": 500,          // optional, max optimiser steps
            "top_n": 3             // optional, return only N lowest-energy structures on download
        }

    Returns (JSON)::

        {"task_id": "...", "status": "pending", "n_structures": 2}
    """
    err = _guard(request)
    if err:
        return err

    if _calculator is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    cif_files = body.get("cif_files")
    if not cif_files or not isinstance(cif_files, list):
        return JSONResponse(
            {"error": "'cif_files' must be a non-empty list of CIF strings"},
            status_code=422,
        )
    if len(cif_files) > _MAX_STRUCTURES_PER_REQUEST:
        return JSONResponse(
            {"error": f"Maximum {_MAX_STRUCTURES_PER_REQUEST} structures per request"},
            status_code=422,
        )
    for cif in cif_files:
        if not isinstance(cif, str) or len(cif.encode()) > _MAX_CIF_STRING_BYTES:
            return JSONResponse(
                {"error": f"Each CIF string must be under {_MAX_CIF_STRING_BYTES // 1024} KB"},
                status_code=422,
            )

    optimizer = str(body.get("optimizer", "BFGS")).upper()
    if optimizer not in _OPTIMIZERS:
        return JSONResponse(
            {"error": f"Unknown optimizer. Supported: {list(_OPTIMIZERS)}"},
            status_code=422,
        )

    try:
        fmax = float(body.get("fmax", _config.get("optimization", {}).get("fmax", 0.05)))
        steps = int(body.get("steps", _config.get("optimization", {}).get("steps", 500)))
        top_n = int(body.get("top_n", 0))
    except (TypeError, ValueError) as exc:
        return JSONResponse({"error": f"Invalid parameter: {exc}"}, status_code=422)

    if steps < 1 or steps > _MAX_STEPS:
        return JSONResponse(
            {"error": f"'steps' must be between 1 and {_MAX_STEPS}"},
            status_code=422,
        )

    task_id = _make_task(cif_files, optimizer, fmax, steps, top_n)
    return JSONResponse(
        {"task_id": task_id, "status": "pending", "n_structures": len(cif_files)},
        status_code=202,
    )


@mcp.custom_route("/api/optimize/upload", methods=["POST"])
async def api_optimize_upload(request: Request) -> Response:
    """Submit a relaxation task by uploading a ZIP file of CIF files.

    Request: ``multipart/form-data`` with the following fields:

    * ``file``      – required, a ``.zip`` archive containing ``.cif`` files
    * ``optimizer`` – optional, default ``"BFGS"``
    * ``fmax``      – optional, default ``0.05``
    * ``steps``     – optional, default ``500``
    * ``top_n``     – optional, return only N lowest-energy structures on download

    Returns (JSON)::

        {"task_id": "...", "status": "pending", "n_structures": 3}
    """
    err = _guard(request)
    if err:
        return err

    if _calculator is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    # Enforce upload size limit before reading the body
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"error": f"Upload exceeds maximum size of {_MAX_UPLOAD_BYTES // (1024*1024)} MB"},
            status_code=413,
        )

    try:
        form = await request.form()
    except Exception:
        return JSONResponse({"error": "Failed to parse multipart form"}, status_code=400)

    upload = form.get("file")
    if upload is None:
        return JSONResponse({"error": "'file' field is required"}, status_code=422)

    try:
        zip_bytes = await upload.read(_MAX_UPLOAD_BYTES + 1)
    except Exception:
        return JSONResponse({"error": "Failed to read uploaded file"}, status_code=400)

    if len(zip_bytes) > _MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"error": f"Upload exceeds maximum size of {_MAX_UPLOAD_BYTES // (1024*1024)} MB"},
            status_code=413,
        )

    # Extract CIF strings from the ZIP – guard against Zip Slip path traversal
    try:
        cif_files: list[str] = []
        cif_names: list[str] = []
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            candidate_names = sorted(
                name for name in zf.namelist()
                if name.lower().endswith(".cif") and not name.startswith("__MACOSX")
            )
            if not candidate_names:
                return JSONResponse(
                    {"error": "ZIP contains no .cif files"},
                    status_code=422,
                )
            for name in candidate_names:
                # Zip Slip: reject entries with path separators or absolute paths
                if "/" in name or "\\" in name or name.startswith((".", "/")):
                    return JSONResponse(
                        {"error": f"ZIP entry with unsafe path rejected: '{Path(name).name}'"},
                        status_code=422,
                    )
                raw = zf.read(name)
                if len(raw) > _MAX_CIF_STRING_BYTES:
                    return JSONResponse(
                        {"error": f"CIF file '{Path(name).name}' exceeds size limit"},
                        status_code=422,
                    )
                cif_files.append(raw.decode("utf-8", errors="replace"))
                cif_names.append(name)
    except zipfile.BadZipFile:
        return JSONResponse({"error": "Uploaded file is not a valid ZIP archive"}, status_code=422)

    if len(cif_files) > _MAX_STRUCTURES_PER_REQUEST:
        return JSONResponse(
            {"error": f"Maximum {_MAX_STRUCTURES_PER_REQUEST} structures per request"},
            status_code=422,
        )

    optimizer = str(form.get("optimizer", "BFGS")).upper()
    if optimizer not in _OPTIMIZERS:
        return JSONResponse(
            {"error": f"Unknown optimizer. Supported: {list(_OPTIMIZERS)}"},
            status_code=422,
        )

    try:
        fmax = float(form.get("fmax", _config.get("optimization", {}).get("fmax", 0.05)))
        steps = int(form.get("steps", _config.get("optimization", {}).get("steps", 500)))
        top_n = int(form.get("top_n", 0))
    except (TypeError, ValueError) as exc:
        return JSONResponse({"error": f"Invalid parameter: {exc}"}, status_code=422)

    if steps < 1 or steps > _MAX_STEPS:
        return JSONResponse(
            {"error": f"'steps' must be between 1 and {_MAX_STEPS}"},
            status_code=422,
        )

    task_id = _make_task(cif_files, optimizer, fmax, steps, top_n)
    return JSONResponse(
        {"task_id": task_id, "status": "pending", "n_structures": len(cif_files),
         "cif_names": cif_names},
        status_code=202,
    )


@mcp.custom_route("/api/tasks/{task_id}", methods=["GET"])
async def api_task_status(request: Request) -> Response:
    """Query the status of a relaxation task.

    Returns (JSON)::

        {
            "task_id": "...",
            "status": "pending" | "running" | "completed" | "failed",
            "n_structures": 2,
            "optimizer": "BFGS",
            "fmax": 0.05,
            "steps": 500,
            "created_at": "...",
            "started_at": "...",
            "completed_at": "...",
            "error": null,
            "infos": [...]         // only when completed
        }
    """
    err = _guard(request)
    if err:
        return err

    task_id = request.path_params["task_id"]
    if not _VALID_UUID_RE.match(task_id):
        return JSONResponse({"error": "Invalid task_id format"}, status_code=400)
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    response = {k: task[k] for k in (
        "task_id", "status", "n_structures", "optimizer", "fmax", "steps", "top_n",
        "created_at", "started_at", "completed_at", "error",
    )}
    if task["status"] == "completed":
        response["infos"] = task["infos"]
    return JSONResponse(response)


@mcp.custom_route("/api/tasks/{task_id}/download", methods=["GET"])
async def api_download(request: Request) -> Response:
    """Download all optimised CIF files as a ZIP archive.

    Only available when ``status == "completed"``.
    """
    err = _guard(request)
    if err:
        return err

    task_id = request.path_params["task_id"]
    if not _VALID_UUID_RE.match(task_id):
        return JSONResponse({"error": "Invalid task_id format"}, status_code=400)
    with _tasks_lock:
        task = _tasks.get(task_id)

    if task is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    if task["status"] == "failed":
        return JSONResponse({"error": "Task failed"}, status_code=410)
    if task["status"] != "completed":
        return JSONResponse(
            {"error": f"Task not completed yet (status: {task['status']})"},
            status_code=202,
        )

    # Pair each structure with its index and energy, then optionally filter top_n
    paired = list(zip(task["optimised_cifs"], task["infos"]))
    top_n = task.get("top_n", 0)
    if top_n and top_n > 0:
        # Sort by energy_eV ascending (None energies go last)
        paired.sort(key=lambda x: (x[1].get("energy_eV") is None, x[1].get("energy_eV", float("inf"))))
        paired = paired[:top_n]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rank, (cif_str, info) in enumerate(paired):
            energy = info.get("energy_eV")
            energy_tag = f"{energy:.4f}eV" if energy is not None else "noE"
            filename = f"optimised_rank{rank:03d}_{energy_tag}.cif"
            zf.writestr(filename, cif_str)
        # Include a JSON summary of per-structure info
        zf.writestr("optimization_info.json", json.dumps([info for _, info in paired], indent=2))
    buf.seek(0)

    zip_name = f"mace_relaxed_{task_id[:8]}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


# ===========================================================================
# MCP tools
# ===========================================================================

@mcp.tool()
def optimize_crystal_structures(
    cif_strings: list[str],
    optimizer: str = "BFGS",
    fmax: float = 0.05,
    steps: int = 500,
) -> dict:
    """Relax crystal structures using the MACE interatomic potential.

    Parameters
    ----------
    cif_strings : list[str]
        List of CIF-format strings to optimise.  Each string represents one
        crystal structure (as produced by, e.g., pymatgen or ASE).
    optimizer : str, optional
        ASE geometry optimiser to use.  One of ``"BFGS"`` (default),
        ``"FIRE"``, or ``"LBFGS"``.
    fmax : float, optional
        Force convergence threshold in eV/Angstrom.  Defaults to ``0.05``.
    steps : int, optional
        Maximum number of optimiser steps per structure.  Defaults to ``500``.

    Returns
    -------
    dict
        ``{"cif_strings": [...], "infos": [...]}``

        * ``cif_strings`` – list of optimised CIF strings, one per input.
        * ``infos`` – list of per-structure metadata dicts containing
          ``converged``, ``energy_eV``, ``max_force_eV_Ang``, ``n_steps``.
    """
    if _calculator is None:
        raise RuntimeError(
            "MACE calculator is not loaded.  Start the server via 'python mcp_server.py'."
        )
    if not cif_strings:
        raise ValueError("cif_strings must not be empty.")
    if len(cif_strings) > _MAX_STRUCTURES_PER_REQUEST:
        raise ValueError(f"Maximum {_MAX_STRUCTURES_PER_REQUEST} structures per call.")
    for cif in cif_strings:
        if not isinstance(cif, str) or len(cif.encode()) > _MAX_CIF_STRING_BYTES:
            raise ValueError(f"Each CIF string must be under {_MAX_CIF_STRING_BYTES // 1024} KB.")

    optimizer_upper = optimizer.upper()
    if optimizer_upper not in _OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Supported: {list(_OPTIMIZERS)}"
        )
    if steps < 1 or steps > _MAX_STEPS:
        raise ValueError(f"steps must be between 1 and {_MAX_STEPS}.")

    results = _relax_cif_list(
        cif_strings,
        optimizer_name=optimizer_upper,
        fmax=fmax,
        steps=steps,
    )

    return {
        "cif_strings": [cif for cif, _ in results],
        "infos": [info for _, info in results],
    }


@mcp.tool()
def save_optimized_structures(
    cif_strings: list[str],
    output_dir: str,
    optimizer: str = "BFGS",
    fmax: float = 0.05,
    steps: int = 500,
) -> dict:
    """Relax crystal structures and save the results as CIF files on disk.

    Parameters
    ----------
    cif_strings : list[str]
        List of CIF-format strings to optimise.
    output_dir : str
        Directory in which to save the optimised ``.cif`` files.  Created if
        it does not exist.
    optimizer : str, optional
        ASE geometry optimiser: ``"BFGS"`` (default), ``"FIRE"``, ``"LBFGS"``.
    fmax : float, optional
        Force convergence threshold in eV/Angstrom (default ``0.05``).
    steps : int, optional
        Maximum number of optimiser steps (default ``500``).

    Returns
    -------
    dict
        ``{"saved_paths": [...], "infos": [...]}``

        * ``saved_paths`` – absolute paths of the saved CIF files.
        * ``infos`` – per-structure optimisation metadata.
    """
    result = optimize_crystal_structures(
        cif_strings=cif_strings,
        optimizer=optimizer,
        fmax=fmax,
        steps=steps,
    )

    out_dir = Path(output_dir).resolve()

    # Restrict writes to the current working directory subtree
    _allowed_base = Path.cwd().resolve()
    try:
        out_dir.relative_to(_allowed_base)
    except ValueError:
        raise ValueError(
            f"output_dir must be within the working directory '{_allowed_base}'. "
            "Absolute paths outside the working directory are not permitted."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for i, cif_str in enumerate(result["cif_strings"]):
        file_path = out_dir / f"optimised_{i:03d}.cif"
        if not str(file_path.resolve()).startswith(str(out_dir)):
            raise ValueError("Resolved file path escapes the output directory.")
        file_path.write_text(cif_str, encoding="utf-8")
        saved_paths.append(str(file_path.resolve()))

    # Write JSON summary alongside the CIF files
    info_path = out_dir / "optimization_info.json"
    info_path.write_text(json.dumps(result["infos"], indent=2), encoding="utf-8")

    return {"saved_paths": saved_paths, "infos": result["infos"]}


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MACE Crystal Structure Relaxation MCP Server"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--transport", default="stdio",
        choices=["stdio", "sse"],
        help=(
            "Transport mode (default: stdio).\n"
            "  stdio – for Claude Desktop / MCP clients.\n"
            "  sse   – HTTP server mode; enables REST API at /api/optimize, "
            "/api/tasks/... and serves MCP over SSE."
        ),
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind when transport=sse (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to bind when transport=sse (default: 8000)",
    )
    args = parser.parse_args()

    _load_model(args.config)
    _start_worker()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        import uvicorn
        uvicorn.run(mcp.sse_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
