from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import file_browser, media, native_dialog, pdf_export, tools
from . import report as report_mod
from .jobs import job_manager
from .pipeline import QUALITY_PRESETS, Settings, workspace_root

app = FastAPI(title="Model Forge")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

OUTPUT_TYPES = {
    "model.glb": "model/gltf-binary",
    "model.obj": "text/plain",
    "model.ply": "application/octet-stream",
    "points.ply": "application/octet-stream",
    "points.glb": "model/gltf-binary",
}


@app.get("/api/health")
async def health():
    binaries = tools.find_binaries()
    colmap = await asyncio.to_thread(tools.colmap_info) if binaries["colmap"] else {"found": False}
    return {"ok": True, "binaries": binaries, "colmap": colmap, "workspace": str(workspace_root())}


# ---------------------------------------------------------------- picking files

@app.get("/api/browse")
async def api_browse(path: str = Query(default="")):
    try:
        return file_browser.browse(path or None)
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class PickRequest(BaseModel):
    initial_path: str = ""


def _dialog_errors(exc: Exception):
    if isinstance(exc, native_dialog.DialogUnavailable):
        return HTTPException(status_code=501, detail=f"Native file dialog unavailable: {exc}")
    return HTTPException(status_code=409, detail=str(exc))


@app.post("/api/pick-files")
async def api_pick_files(req: PickRequest):
    try:
        paths = await asyncio.to_thread(native_dialog.pick_files, "Add photos or videos", req.initial_path)
    except (native_dialog.DialogUnavailable, native_dialog.DialogBusy) as exc:
        raise _dialog_errors(exc) from exc
    return {"paths": paths}


@app.post("/api/pick-folder")
async def api_pick_folder(req: PickRequest):
    try:
        path = await asyncio.to_thread(native_dialog.pick_folder, "Add a folder of photos", req.initial_path)
    except (native_dialog.DialogUnavailable, native_dialog.DialogBusy) as exc:
        raise _dialog_errors(exc) from exc
    return {"path": path}


class InspectRequest(BaseModel):
    paths: list[str]


@app.post("/api/inspect")
async def api_inspect(req: InspectRequest):
    async def one(path: str):
        try:
            return await media.inspect(path)
        except tools.ToolError as exc:
            return {"path": path, "name": Path(path).name or path, "kind": "error", "error": str(exc)}
    return {"items": await asyncio.gather(*(one(p) for p in req.paths[:500]))}


_thumb_cache: OrderedDict[tuple, bytes] = OrderedDict()


@app.get("/api/thumb")
async def api_thumb(path: str = Query(...)):
    try:
        key = (path, os.path.getmtime(path))
    except OSError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    data = _thumb_cache.get(key)
    if data is None:
        try:
            data = await media.thumbnail(path)
        except tools.ToolError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _thumb_cache[key] = data
        while len(_thumb_cache) > 400:
            _thumb_cache.popitem(last=False)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


# ---------------------------------------------------------------- reconstruction

class ReconstructRequest(BaseModel):
    inputs: list[str]
    name: str = ""
    quality: str = "standard"
    frames_per_video: int = Field(default=60, ge=5, le=600)
    matcher: str = "auto"
    dense: str = "auto"
    mesher: str = "poisson"
    crop: bool = True
    keep_largest: bool = True
    keep_workspace: bool = False


@app.get("/api/options")
async def api_options():
    return {"quality": list(QUALITY_PRESETS.keys())}


@app.post("/api/reconstruct")
async def api_reconstruct(req: ReconstructRequest):
    if not tools.find_colmap():
        raise HTTPException(status_code=400, detail="COLMAP was not found. Install it and put it on PATH, or set COLMAP_BIN.")
    if not req.inputs:
        raise HTTPException(status_code=400, detail="Add some photos or videos first.")
    missing = [p for p in req.inputs if not Path(p).exists()]
    if missing:
        raise HTTPException(status_code=400, detail=f"Not found: {missing[0]}")
    if req.quality not in QUALITY_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown quality: {req.quality}")
    for field, allowed in (("matcher", ("auto", "exhaustive", "sequential")), ("dense", ("auto", "on", "off")),
                           ("mesher", ("poisson", "delaunay"))):
        if getattr(req, field) not in allowed:
            raise HTTPException(status_code=400, detail=f"Unknown {field}: {getattr(req, field)}")

    job = job_manager.create(Settings(**req.model_dump()))
    asyncio.create_task(job_manager.run(job.id))
    return {"job_id": job.id}


def _get_job(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _get_finished_job(job_id: str):
    job = _get_job(job_id)
    if job.status != "done":
        raise HTTPException(status_code=400, detail=f"Job is not finished (status: {job.status})")
    return job


@app.get("/api/jobs/{job_id}")
async def api_job_status(job_id: str, log: int = Query(default=0, ge=0, le=400)):
    return _get_job(job_id).public_dict(log_lines=log)


@app.post("/api/jobs/{job_id}/cancel")
async def api_job_cancel(job_id: str):
    if not job_manager.cancel(job_id):
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    return {"cancelled": True}


@app.get("/api/jobs/{job_id}/files/{name}")
async def api_job_file(job_id: str, name: str, download: bool = False):
    job = _get_finished_job(job_id)
    if name not in OUTPUT_TYPES:
        raise HTTPException(status_code=404, detail="Unknown file")
    path = Path(job.workspace) / "output" / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="This run did not produce that file")
    stem = report_mod.file_stem(job)
    return FileResponse(
        str(path), media_type=OUTPUT_TYPES[name],
        filename=f"{stem}-{name}" if download else None,
    )


@app.post("/api/jobs/{job_id}/open-folder")
async def api_open_folder(job_id: str):
    job = _get_job(job_id)
    folder = Path(job.workspace or "") / "output"
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail="Output folder not found")
    try:
        if os.name == "nt":
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except OSError as exc:
        raise HTTPException(status_code=501, detail=f"Could not open the folder: {exc}") from exc
    return {"path": str(folder)}


@app.get("/api/jobs/{job_id}/report.html")
async def api_job_report_html(job_id: str):
    return HTMLResponse(report_mod.generate_report_html(_get_finished_job(job_id)))


@app.get("/api/jobs/{job_id}/report.pdf")
async def api_job_report_pdf(job_id: str):
    job = _get_finished_job(job_id)
    try:
        pdf = await pdf_export.html_to_pdf(report_mod.generate_report_html(job))
    except pdf_export.PdfUnavailable as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except pdf_export.PdfError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{report_mod.file_stem(job)}-report.pdf"'},
    )


@app.get("/api/jobs/{job_id}/report.json")
async def api_job_report_json(job_id: str):
    job = _get_finished_job(job_id)
    return JSONResponse(
        report_mod.generate_report_json(job),
        headers={"Content-Disposition": f'attachment; filename="{report_mod.file_stem(job)}-report.json"'},
    )


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
