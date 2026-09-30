"""The desktop app's backend: every method is callable from the page as window.pywebview.api.<name>(...).

Nothing listens on a network port. pywebview passes calls straight from the window's JavaScript to this
object. Each method returns {"ok": True, "data": ...} or {"ok": False, "error": "..."}, so the frontend
gets the same error text the old HTTP API sent as `detail`.

pywebview exposes every public attribute to JavaScript, so internal state is kept in `_`-prefixed names.
"""
from __future__ import annotations

import asyncio
import base64
import functools
import json
import os
import shutil
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import webview

from . import media, pdf_export, tools
from . import report as report_mod
from .jobs import job_manager
from .pipeline import QUALITY_PRESETS, Settings, workspace_root

OUTPUT_FILES = {  # file name -> label shown in the Save dialog's type filter
    "model.glb": "glTF binary",
    "model.obj": "Wavefront OBJ",
    "model.ply": "PLY mesh",
    "points.ply": "PLY point cloud",
    "points.glb": "glTF binary",
}
CHUNK_BYTES = 4 * 1024 * 1024  # models reach the viewer in pieces this size (base64 through the bridge)
SETTINGS_FIELDS = ("inputs", "name", "quality", "frames_per_video", "matcher", "dense", "mesher",
                   "crop", "keep_largest", "keep_workspace")
CHOICES = {"matcher": ("auto", "exhaustive", "sequential"), "dense": ("auto", "on", "off"),
           "mesher": ("poisson", "delaunay")}

# pywebview renamed its dialog constants in 5.x; support both spellings.
_FD = getattr(webview, "FileDialog", None)
OPEN_DIALOG = _FD.OPEN if _FD else webview.OPEN_DIALOG
FOLDER_DIALOG = _FD.FOLDER if _FD else webview.FOLDER_DIALOG
SAVE_DIALOG = _FD.SAVE if _FD else webview.SAVE_DIALOG


def _patterns(exts) -> str:
    return ";".join(f"*{ext}" for ext in sorted(exts))


MEDIA_FILE_TYPES = (
    f"Photos and videos ({_patterns(media.MEDIA_EXTENSIONS)})",
    f"Photos ({_patterns(media.IMAGE_EXTENSIONS)})",
    f"Videos ({_patterns(media.VIDEO_EXTENSIONS)})",
    "All files (*.*)",
)


class ApiError(Exception):
    """An error whose message is shown to the user as-is."""


def _result(fn):
    """Wrap an API method so it always returns {"ok", "data"|"error"} instead of raising."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            # round-trip through JSON so anything the old API serialised (paths, tuples) arrives the same way
            return {"ok": True, "data": json.loads(json.dumps(fn(self, *args, **kwargs), default=str))}
        except ApiError as exc:
            return {"ok": False, "error": str(exc)}
        except tools.ToolError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return wrapper


def _start_dir(path: str) -> str:
    candidate = path if os.path.isdir(path) else os.path.dirname(path or "")
    return candidate if os.path.isdir(candidate) else ""


def _paths(result) -> list[str]:
    """create_file_dialog returns a tuple, a list or a plain string depending on platform and dialog."""
    if not result:
        return []
    items = [result] if isinstance(result, str) else list(result)
    return [os.path.normpath(p) for p in items if p]


class Api:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop  # background event loop that runs jobs and tools (jobs.py is asyncio-based)
        self._window = None
        self._thumbs: OrderedDict[tuple, str] = OrderedDict()
        self._thumbs_lock = threading.Lock()

    def _attach(self, window) -> None:
        self._window = window

    def _run(self, coro, timeout: float | None = None):
        """Run a coroutine on the event loop thread and wait for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    def _on_loop(self, fn, *args):
        """Run a job-manager call on the event loop thread (asyncio.Event isn't thread-safe) and wait."""

        async def call():
            return fn(*args)

        return self._run(call(), timeout=10)

    def _job(self, job_id: str):
        job = job_manager.get(job_id)
        if job is None:
            raise ApiError("Job not found")
        return job

    def _finished_job(self, job_id: str):
        job = self._job(job_id)
        if job.status != "done":
            raise ApiError(f"Job is not finished (status: {job.status})")
        return job

    def _output_path(self, job_id: str, name: str) -> Path:
        job = self._finished_job(job_id)
        if name not in OUTPUT_FILES:
            raise ApiError("Unknown file")
        path = Path(job.workspace) / "output" / name
        if not path.is_file():
            raise ApiError("This run did not produce that file")
        return path

    def _ask_save_path(self, filename: str, file_type: str) -> str:
        chosen = self._window.create_file_dialog(
            SAVE_DIALOG,
            save_filename=filename,
            file_types=(file_type, "All files (*.*)"),
        )
        return next(iter(_paths(chosen)), "")

    # ---------- status / options ----------
    @_result
    def health(self):
        binaries = tools.find_binaries()
        colmap = tools.colmap_info() if binaries["colmap"] else {"found": False}
        return {"ok": True, "binaries": binaries, "colmap": colmap, "workspace": str(workspace_root())}

    @_result
    def options(self):
        return {"quality": list(QUALITY_PRESETS.keys())}

    # ---------- picking files ----------
    @_result
    def pick_files(self, start: str = ""):
        """The operating system's own Open dialog; returns {"paths": []} if cancelled."""
        chosen = self._window.create_file_dialog(
            OPEN_DIALOG, directory=_start_dir(start), allow_multiple=True, file_types=MEDIA_FILE_TYPES,
        )
        return {"paths": _paths(chosen)}

    @_result
    def pick_folder(self, start: str = ""):
        chosen = self._window.create_file_dialog(FOLDER_DIALOG, directory=_start_dir(start))
        return {"path": next(iter(_paths(chosen)), "")}

    @_result
    def inspect(self, paths: list):
        async def one(path: str):
            try:
                return await media.inspect(path)
            except tools.ToolError as exc:
                return {"path": path, "name": Path(path).name or path, "kind": "error", "error": str(exc)}

        async def all_items():
            return await asyncio.gather(*(one(str(p)) for p in (paths or [])[:500]))

        return {"items": self._run(all_items())}

    @_result
    def thumb(self, path: str):
        """A small JPEG preview as a data: URL (there's no server for an <img src> to point at)."""
        try:
            key = (path, os.path.getmtime(path))
        except OSError as exc:
            raise ApiError("File not found") from exc
        with self._thumbs_lock:
            src = self._thumbs.get(key)
        if src is None:
            data = self._run(media.thumbnail(path), timeout=60)
            src = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
            with self._thumbs_lock:
                self._thumbs[key] = src
                while len(self._thumbs) > 400:
                    self._thumbs.popitem(last=False)
        return {"src": src}

    # ---------- reconstruction ----------
    @_result
    def reconstruct(self, body: dict):
        req = {k: v for k, v in (body or {}).items() if k in SETTINGS_FIELDS}
        req["inputs"] = [str(p) for p in req.get("inputs") or []]
        if not tools.find_colmap():
            raise ApiError("COLMAP was not found. Install it and put it on PATH, or set COLMAP_BIN.")
        if not req["inputs"]:
            raise ApiError("Add some photos or videos first.")
        missing = [p for p in req["inputs"] if not Path(p).exists()]
        if missing:
            raise ApiError(f"Not found: {missing[0]}")
        if req.setdefault("quality", "standard") not in QUALITY_PRESETS:
            raise ApiError(f"Unknown quality: {req['quality']}")
        for field, allowed in CHOICES.items():
            if req.setdefault(field, allowed[0]) not in allowed:
                raise ApiError(f"Unknown {field}: {req[field]}")
        frames = int(req.get("frames_per_video", 60))
        if not 5 <= frames <= 600:
            raise ApiError("Frames per video must be between 5 and 600")
        req["frames_per_video"] = frames
        req["name"] = str(req.get("name") or "")
        for flag in ("crop", "keep_largest", "keep_workspace"):
            if flag in req:
                req[flag] = bool(req[flag])

        job = self._on_loop(job_manager.create, Settings(**req))
        asyncio.run_coroutine_threadsafe(job_manager.run(job.id), self._loop)
        return {"job_id": job.id}

    @_result
    def job(self, job_id: str, log: int = 0):
        return self._job(job_id).public_dict(log_lines=max(0, min(int(log or 0), 400)))

    @_result
    def cancel(self, job_id: str):
        if not self._on_loop(job_manager.cancel, job_id):
            raise ApiError("Job cannot be cancelled")
        return {"cancelled": True}

    # ---------- results ----------
    @_result
    def model_chunk(self, job_id: str, name: str, offset: int = 0):
        """One piece of an output file, base64-encoded, for the 3D viewer to reassemble."""
        path = self._output_path(job_id, name)
        size = path.stat().st_size
        offset = max(0, int(offset))
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(CHUNK_BYTES)
        return {"data": base64.b64encode(data).decode("ascii"), "size": size, "end": offset + len(data)}

    @_result
    def save_output(self, job_id: str, name: str):
        """Ask where to save one of the model files, then copy it there. Returns {"path": ""} if cancelled."""
        src = self._output_path(job_id, name)
        job = self._finished_job(job_id)
        ext = src.suffix.lstrip(".")
        dest = self._ask_save_path(f"{report_mod.file_stem(job)}-{name}", f"{OUTPUT_FILES[name]} (*.{ext})")
        if dest:
            try:
                shutil.copyfile(src, dest)
            except OSError as exc:
                raise ApiError(f"Could not save the file: {exc}") from exc
        return {"path": dest}

    @_result
    def view_report(self, job_id: str, print_it: bool = False):
        job = self._finished_job(job_id)
        html = report_mod.generate_report_html(job)
        if print_it:
            html = html.replace("</body>", "<script>window.addEventListener('load', () => window.print());</script></body>")
        webview.create_window(
            f"Model Forge Report — {job.settings.name or job.id}",
            html=html, width=1000, height=900, background_color="#1c2023",
        )
        return {"opened": True}

    @_result
    def save_report(self, job_id: str, kind: str = "pdf"):
        """Ask where to save the PDF, HTML or JSON report, then write it. Returns {"path": ""} if cancelled.

        With no Edge/Chrome to render a PDF, the report opens with the print dialog instead ("printed": True),
        where "Save as PDF" works."""
        job = self._finished_job(job_id)
        if kind == "json":
            content: str | bytes = json.dumps(report_mod.generate_report_json(job), indent=2)
        elif kind == "html":
            content = report_mod.generate_report_html(job)
        else:
            kind = "pdf"
            try:
                content = self._run(pdf_export.html_to_pdf(report_mod.generate_report_html(job)))
            except pdf_export.PdfUnavailable:
                self.view_report(job_id, True)
                return {"path": "", "printed": True}
            except pdf_export.PdfError as exc:
                raise ApiError(str(exc)) from exc
        dest = self._ask_save_path(f"{report_mod.file_stem(job)}-report.{kind}", f"{kind.upper()} file (*.{kind})")
        if dest:
            try:
                if isinstance(content, bytes):
                    Path(dest).write_bytes(content)
                else:
                    Path(dest).write_text(content, encoding="utf-8")
            except OSError as exc:
                raise ApiError(f"Could not save the report: {exc}") from exc
        return {"path": dest}

    @_result
    def open_folder(self, job_id: str):
        folder = Path(self._job(job_id).workspace or "") / "output"
        if not folder.is_dir():
            raise ApiError("Output folder not found")
        try:
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as exc:
            raise ApiError(f"Could not open the folder: {exc}") from exc
        return {"path": str(folder)}
