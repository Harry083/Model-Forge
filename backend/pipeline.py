"""The reconstruction pipeline: photos/videos -> COLMAP structure-from-motion -> (dense) mesh -> GLB/OBJ/PLY.

Stages:
  prepare   copy photos, pull frames out of videos (ffmpeg)
  features  SIFT features in every image                          colmap feature_extractor
  matching  which images overlap                                   colmap exhaustive/sequential_matcher
  mapping   camera positions + sparse point cloud                  colmap mapper
  -- with a CUDA build of COLMAP --
  undistort                                                        colmap image_undistorter
  stereo    per-image depth maps                                   colmap patch_match_stereo
  fusion    depth maps -> dense coloured point cloud               colmap stereo_fusion
  --
  meshing   surface from the points                                colmap poisson_mesher / delaunay_mesher
  export    crop to the subject, stand it upright, write files     geometry.py

Without CUDA the mesh is built from the sparse points (normals estimated here), which is coarser
but still gives a recognisable, watertight shape.
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import geometry as geo
from . import media
from . import sparse_model as sm
from . import tools

QUALITY_PRESETS = {
    #            feature image size, max features, dense image size, poisson depth dense/sparse, video frame size
    "draft":    {"sift_size": 1000, "features": 4096,  "dense_size": 1000, "depth_dense": 9,  "depth_sparse": 8,  "frame_size": 1280},
    "standard": {"sift_size": 1600, "features": 8192,  "dense_size": 1600, "depth_dense": 10, "depth_sparse": 9,  "frame_size": 1920},
    "high":     {"sift_size": 3200, "features": 16384, "dense_size": 2400, "depth_dense": 11, "depth_sparse": 10, "frame_size": 3840},
}

STAGES = {
    "prepare": ("Preparing photos & frames", 4),
    "features": ("Finding features", 14),
    "matching": ("Matching images", 20),
    "mapping": ("Placing cameras", 22),
    "undistort": ("Undistorting images", 3),
    "stereo": ("Computing depth maps", 26),
    "fusion": ("Fusing depth maps", 5),
    "meshing": ("Building the mesh", 4),
    "export": ("Cleaning up & exporting", 2),
}
DENSE_STAGES = ("undistort", "stereo", "fusion")

MAX_VIEWER_POINTS = 1_500_000
WORKSPACE_ROOT = Path.home() / "ModelForge"


@dataclass
class Settings:
    inputs: list[str]
    name: str = ""
    quality: str = "standard"        # draft | standard | high
    frames_per_video: int = 60
    matcher: str = "auto"            # auto | exhaustive | sequential
    dense: str = "auto"              # auto | on | off
    mesher: str = "poisson"          # poisson | delaunay (dense only)
    crop: bool = True                # keep only the thing the cameras were pointed at
    keep_largest: bool = True        # drop floating bits of mesh
    keep_workspace: bool = False     # keep images, database and depth maps afterwards


@dataclass
class Progress:
    stage: str = "queued"
    stages: list[str] = field(default_factory=list)
    stage_fraction: float = 0.0
    detail: str = ""

    @property
    def percent(self) -> float:
        if self.stage not in self.stages:
            return 100.0 if self.stage == "done" else 0.0
        weights = [STAGES[s][1] for s in self.stages]
        idx = self.stages.index(self.stage)
        done = sum(weights[:idx]) + weights[idx] * max(0.0, min(1.0, self.stage_fraction))
        return 100.0 * done / sum(weights)


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug[:40] or "model"


def workspace_root() -> Path:
    env = os.environ.get("MODEL_FORGE_WORKSPACE")
    return Path(env).expanduser() if env else WORKSPACE_ROOT


class Reconstruction:
    def __init__(self, job_id: str, settings: Settings, on_update: Callable[[], None],
                 cancel_event: asyncio.Event) -> None:
        self.job_id = job_id
        self.s = settings
        self.preset = QUALITY_PRESETS.get(settings.quality, QUALITY_PRESETS["standard"])
        self.on_update = on_update
        self.cancel_event = cancel_event
        self.progress = Progress()
        self.log_lines: list[str] = []
        self.warnings: list[str] = []
        self.stage_times: dict[str, float] = {}

        name = settings.name.strip() or "model"
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        self.workspace = workspace_root() / f"{stamp}_{slugify(name)}_{job_id[:6]}"
        self.image_dir = self.workspace / "images"
        self.database = self.workspace / "database.db"
        self.sparse_dir = self.workspace / "sparse"
        self.dense_dir = self.workspace / "dense"
        self.output_dir = self.workspace / "output"
        self.log_path = self.workspace / "logs" / "pipeline.log"

        info = tools.colmap_info()
        self.cuda = bool(info.get("cuda"))
        self.use_dense = settings.dense == "on" or (settings.dense == "auto" and self.cuda)
        if self.use_dense and not self.cuda:
            raise tools.ToolError(
                "Dense reconstruction needs a CUDA build of COLMAP and an NVIDIA GPU. "
                "Set 'Dense reconstruction' to Auto or Off to build the model from the sparse points."
            )
        self.progress.stages = [s for s in STAGES if self.use_dense or s not in DENSE_STAGES]

    # ------------------------------------------------------------------ helpers

    def _log(self, line: str) -> None:
        self.log_lines.append(line)
        if len(self.log_lines) > 400:
            del self.log_lines[:100]

    def _enter(self, stage: str, detail: str = "") -> None:
        now = time.time()
        if self.progress.stage in STAGES and hasattr(self, "_stage_start"):
            self.stage_times[self.progress.stage] = round(now - self._stage_start, 1)
        self._stage_start = now
        self.progress.stage = stage
        self.progress.stage_fraction = 0.0
        self.progress.detail = detail
        self._log(f"== {STAGES[stage][0]}")
        self.on_update()

    def _set(self, fraction: float, detail: Optional[str] = None) -> None:
        self.progress.stage_fraction = max(self.progress.stage_fraction, fraction)
        if detail is not None:
            self.progress.detail = detail
        self.on_update()

    def _counter(self, pattern: str, label: str, passes: int = 1) -> Callable[[str], None]:
        """on_line callback that turns COLMAP's '[i/N]'-style log lines into stage progress."""
        regex = re.compile(pattern)
        state = {"pass": 0, "last": 0}

        def on_line(line: str) -> None:
            self._log(line)
            m = regex.search(line)
            if not m:
                return
            i, n = int(m.group(1)), int(m.group(2))
            if i < state["last"]:
                state["pass"] = min(passes - 1, state["pass"] + 1)
            state["last"] = i
            if n:
                self._set((state["pass"] + i / n) / passes, f"{label} {i} of {n}")
        return on_line

    async def _colmap(self, command: str, options: dict, on_line: Optional[Callable[[str], None]] = None) -> str:
        return await tools.colmap(
            command, options,
            on_line=on_line or self._log,
            cancel_event=self.cancel_event,
            log_path=self.log_path,
        )

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise tools.Cancelled()

    # ------------------------------------------------------------------ stages

    async def run(self) -> dict:
        started = time.time()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sources = media.expand(self.s.inputs)
        if not sources:
            raise tools.ToolError("None of the chosen items are photos or videos Model Forge can read.")

        self._enter("prepare", "Copying photos")
        staged, video_folders = await media.stage_images(
            sources, self.image_dir,
            frames_per_video=self.s.frames_per_video,
            max_size=self.preset["frame_size"],
            on_progress=lambda i, n, name: self._set(i / max(n, 1), f"{name} ({i + 1} of {n})" if name else ""),
            cancel_event=self.cancel_event,
            log_path=self.log_path,
        )
        if len(staged) < 3:
            raise tools.ToolError(f"At least 3 images are needed; got {len(staged)}. Add more photos or a longer video.")

        await self._features(staged, video_folders)
        await self._matching(len(staged), bool(video_folders) and all(not im.name.startswith("photos/") for im in staged))
        model_path, analysis = await self._mapping(len(staged))

        text_dir = self.workspace / "sparse_text"
        text_dir.mkdir(exist_ok=True)
        await self._colmap("model_converter", {"input_path": model_path, "output_path": text_dir, "output_type": "TXT"})
        model = sm.read_text_model(text_dir)
        framing = sm.compute_framing(model, crop=self.s.crop)

        if self.use_dense:
            points, mesh = await self._dense(model_path)
        else:
            points, mesh = await self._sparse_mesh(model)

        self._enter("export", "Cropping and orienting")
        outputs, geo_stats = await asyncio.to_thread(self._export, points, mesh, framing)

        if not self.s.keep_workspace:
            for path in (self.image_dir, self.dense_dir, self.database):
                shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)

        self.stage_times["export"] = round(time.time() - self._stage_start, 1)
        self.progress.stage = "done"
        self.progress.detail = ""

        registered = set(model.image_names)
        cameras = []
        for c, f in zip(model.centers, model.forwards):
            pc = framing.rotation @ (c - framing.center) * framing.scale
            pf = framing.rotation @ f
            cameras.append([round(float(v), 4) for v in (*pc, *pf)])

        n_images = len(staged)
        n_registered = len(registered)
        if n_registered < 0.6 * n_images:
            self.warnings.append(
                f"Only {n_registered} of {n_images} images could be placed. Images with little overlap, "
                "motion blur, or plain/shiny surfaces are the usual cause — try more photos taken closer together."
            )

        return {
            "workspace": str(self.workspace),
            "outputs": outputs,
            "dense": self.use_dense,
            "stats": {
                "input_files": len(sources),
                "photos": sum(1 for s in sources if media.kind_of(s) == "image"),
                "videos": len(video_folders),
                "images": n_images,
                "registered_images": n_registered,
                "sparse_points": analysis.get("points"),
                "mean_reprojection_error": analysis.get("mean_reprojection_error"),
                "mean_track_length": analysis.get("mean_track_length"),
                "models_found": analysis.get("models_found"),
                **geo_stats,
                "elapsed_seconds": round(time.time() - started, 1),
                "stage_seconds": self.stage_times,
            },
            "settings": {
                "quality": self.s.quality,
                "frames_per_video": self.s.frames_per_video,
                "matcher": self._matcher_used,
                "dense": self.use_dense,
                "mesher": self.s.mesher if self.use_dense else "poisson (sparse)",
                "crop": self.s.crop,
                "keep_largest": self.s.keep_largest,
            },
            "colmap": tools.colmap_info(),
            "images": [
                {"name": im.name, "source": im.source, "time": im.frame_time, "registered": im.name in registered}
                for im in staged
            ],
            "cameras": cameras[:600],
            "warnings": self.warnings,
        }

    async def _features(self, staged: list[media.StagedImage], video_folders: list[str]) -> None:
        self._enter("features", "Starting")
        opt = tools.colmap_option
        use_gpu = opt("feature_extractor", "FeatureExtraction.use_gpu", "SiftExtraction.use_gpu")
        max_size = opt("feature_extractor", "FeatureExtraction.max_image_size", "SiftExtraction.max_image_size")
        max_feat = opt("feature_extractor", "SiftExtraction.max_num_features", "FeatureExtraction.max_num_features")
        per_folder = opt("feature_extractor", "ImageReader.single_camera_per_folder")

        photos = [im.name for im in staged if im.name.startswith("photos/")]
        frames = [im.name for im in staged if not im.name.startswith("photos/")]
        groups = [(photos, {}), (frames, {per_folder: 1})]
        total = len(staged)
        done_before = 0
        for names, extra in groups:
            if not names:
                continue
            list_path = self.workspace / f"image_list_{done_before}.txt"
            list_path.write_text("\n".join(names) + "\n", encoding="utf-8")
            offset, count = done_before, len(names)
            regex = re.compile(r"Processed file \[(\d+)/(\d+)\]")

            def on_line(line: str, offset=offset) -> None:
                self._log(line)
                m = regex.search(line)
                if m:
                    i = offset + int(m.group(1))
                    self._set(i / total, f"Image {i} of {total}")

            await self._colmap("feature_extractor", {
                "database_path": self.database,
                "image_path": self.image_dir,
                "image_list_path": list_path,
                use_gpu: self.cuda,
                max_size: self.preset["sift_size"],
                max_feat: self.preset["features"],
                **extra,
            }, on_line)
            done_before += count

    async def _matching(self, n_images: int, video_only: bool) -> None:
        self._enter("matching", "Starting")
        matcher = self.s.matcher
        if matcher == "auto":
            matcher = "sequential" if (n_images > 150 and video_only) or n_images > 300 else "exhaustive"
        self._matcher_used = matcher
        cmd = f"{matcher}_matcher"
        use_gpu = tools.colmap_option(cmd, "FeatureMatching.use_gpu", "SiftMatching.use_gpu")
        options = {"database_path": self.database, use_gpu: self.cuda}

        if matcher == "exhaustive":
            block = min(50, max(10, math.ceil(n_images / 5)))
            options["ExhaustiveMatching.block_size"] = block
            regex = re.compile(r"Matching block \[(\d+)/(\d+),\s*(\d+)/(\d+)\]")

            def on_line(line: str) -> None:
                self._log(line)
                m = regex.search(line)
                if m:
                    i, n, j, n2 = map(int, m.groups())
                    step, steps = (i - 1) * n2 + j, n * n2
                    self._set((step - 1) / steps, f"Block {step} of {steps}")
            await self._colmap(cmd, options, on_line)
        else:
            options["SequentialMatching.overlap"] = 20
            await self._colmap(cmd, options, self._counter(r"Matching image \[(\d+)/(\d+)\]", "Image"))

    async def _mapping(self, n_images: int) -> tuple[Path, dict]:
        self._enter("mapping", "Finding a starting pair")
        self.sparse_dir.mkdir(exist_ok=True)
        regex = re.compile(r"Registering image #\d+ \((\d+)\)")

        def on_line(line: str) -> None:
            self._log(line)
            m = regex.search(line)
            if m:
                i = int(m.group(1))
                self._set(0.95 * i / n_images, f"{i} of {n_images} images placed")
            elif "Initializing with image pair" in line:
                self._set(0.02, "Found a starting pair")

        await self._colmap("mapper", {
            "database_path": self.database,
            "image_path": self.image_dir,
            "output_path": self.sparse_dir,
        }, on_line)

        models = sorted(p for p in self.sparse_dir.iterdir() if (p / "images.bin").exists())
        if not models:
            raise tools.ToolError(
                "COLMAP could not work out where the cameras were. The images probably don't overlap enough, "
                "or the subject has too little texture. Take more photos from closer angles (60–80% overlap)."
            )
        best, best_stats = None, {}
        for path in models:
            out = await self._colmap("model_analyzer", {"path": path})
            stats = _parse_analyzer(out)
            if best is None or stats.get("registered_images", 0) > best_stats.get("registered_images", 0):
                best, best_stats = path, stats
        best_stats["models_found"] = len(models)
        if len(models) > 1:
            self.warnings.append(
                f"The images split into {len(models)} separate groups that couldn't be joined; "
                f"the largest ({best_stats.get('registered_images')} images) was used. Add photos that bridge the gaps."
            )

        return best, best_stats

    async def _dense(self, model_path: Path) -> tuple[geo.Geometry, Optional[geo.Geometry]]:
        size = self.preset["dense_size"]
        self._enter("undistort", "Starting")
        await self._colmap("image_undistorter", {
            "image_path": self.image_dir,
            "input_path": model_path,
            "output_path": self.dense_dir,
            "output_type": "COLMAP",
            "max_image_size": size,
        }, self._counter(r"Undistorting image \[(\d+)/(\d+)\]", "Image"))

        self._enter("stereo", "Starting")
        await self._colmap("patch_match_stereo", {
            "workspace_path": self.dense_dir,
            "workspace_format": "COLMAP",
            "PatchMatchStereo.geom_consistency": "true",
            "PatchMatchStereo.max_image_size": size,
        }, self._counter(r"Processing view (\d+)\s*/\s*(\d+)", "View", passes=2))

        self._enter("fusion", "Starting")
        fused = self.dense_dir / "fused.ply"
        await self._colmap("stereo_fusion", {
            "workspace_path": self.dense_dir,
            "workspace_format": "COLMAP",
            "input_type": "geometric",
            "output_path": fused,
        }, self._counter(r"Fusing image \[(\d+)/(\d+)\]", "Image"))
        if not fused.exists():
            raise tools.ToolError("Depth-map fusion produced no points.")

        self._enter("meshing", "Running Poisson reconstruction" if self.s.mesher == "poisson" else "Running Delaunay meshing")
        if self.s.mesher == "delaunay":
            mesh_path = self.dense_dir / "meshed-delaunay.ply"
            await self._colmap("delaunay_mesher", {
                "input_path": self.dense_dir, "input_type": "dense", "output_path": mesh_path,
            })
        else:
            mesh_path = self.dense_dir / "meshed-poisson.ply"
            await self._colmap("poisson_mesher", {
                "input_path": fused,
                "output_path": mesh_path,
                "PoissonMeshing.depth": self.preset["depth_dense"],
            })
        points = await asyncio.to_thread(geo.read_ply, fused)
        mesh = await asyncio.to_thread(geo.read_ply, mesh_path) if mesh_path.exists() else None
        return points, mesh

    async def _sparse_mesh(self, model: sm.SparseModel) -> tuple[geo.Geometry, Optional[geo.Geometry]]:
        self._enter("meshing", "Estimating surface normals")
        keep = (model.track_lengths >= 3) & (model.errors < 2.0)
        if keep.sum() < 1000:  # small captures: every triangulated point counts
            keep = model.errors < 4.0

        def prepare():
            pts = model.points[keep]
            cols = model.colors[keep]
            views = model.view_points[keep]
            inliers = geo.statistical_outliers(pts)
            pts, cols, views = pts[inliers], cols[inliers], views[inliers]
            normals = geo.estimate_normals(pts, views)
            cloud = geo.Geometry(pts, cols, normals)
            geo.write_ply(self.workspace / "sparse_points.ply", cloud)
            return cloud

        cloud = await asyncio.to_thread(prepare)
        if len(cloud.points) < 50:
            self.warnings.append("Too few points for a mesh; only the point cloud was exported.")
            return cloud, None
        self._set(0.4, "Running Poisson reconstruction")
        mesh_path = self.workspace / "sparse_mesh.ply"
        await self._colmap("poisson_mesher", {
            "input_path": self.workspace / "sparse_points.ply",
            "output_path": mesh_path,
            "PoissonMeshing.depth": self.preset["depth_sparse"],
            "PoissonMeshing.trim": 5,
        })
        mesh = await asyncio.to_thread(geo.read_ply, mesh_path) if mesh_path.exists() else None
        self.warnings.append(
            "Built from sparse points because this COLMAP has no CUDA support — fine for shape, coarse on detail. "
            "A CUDA build of COLMAP with an NVIDIA GPU enables dense reconstruction."
        )
        return cloud, mesh

    def _export(self, points: geo.Geometry, mesh: Optional[geo.Geometry], framing: sm.Framing) -> tuple[dict, dict]:
        stats: dict = {}
        outputs: dict = {}

        def crop(g: geo.Geometry) -> geo.Geometry:
            if math.isinf(framing.radius):
                return g
            inside = np.linalg.norm(g.points - framing.focus, axis=1) <= framing.radius
            return geo.filter_vertices(g, inside)

        points = crop(points)
        if not self.use_dense or len(points.points) <= 3_000_000:
            if len(points.points) > 20:
                points = geo.filter_vertices(points, geo.statistical_outliers(points.points))
        points = geo.transform(points, framing.rotation, framing.center, framing.scale)
        stats["points"] = int(len(points.points))
        self._set(0.3, "Writing point cloud")
        geo.write_ply(self.output_dir / "points.ply", points)
        outputs["points_ply"] = "points.ply"
        viewer_points = points
        if len(points.points) > MAX_VIEWER_POINTS:
            pick = np.random.default_rng(0).choice(len(points.points), MAX_VIEWER_POINTS, replace=False)
            viewer_points = geo.filter_vertices(points, np.isin(np.arange(len(points.points)), pick))
        geo.write_glb(self.output_dir / "points.glb", geo.Geometry(viewer_points.points, viewer_points.colors), "points")
        outputs["points_glb"] = "points.glb"

        if mesh is not None and mesh.is_mesh:
            self._set(0.5, "Cleaning mesh")
            mesh = crop(mesh)
            if self.s.keep_largest:
                mesh = geo.keep_largest_component(mesh)
            mesh = geo.remove_unreferenced(mesh)
            if mesh.is_mesh:
                mesh = geo.transform(mesh, framing.rotation, framing.center, framing.scale)
                mesh.normals = geo.vertex_normals(mesh.points, mesh.faces)
                stats["mesh_vertices"] = int(len(mesh.points))
                stats["mesh_faces"] = int(len(mesh.faces))
                self._set(0.7, "Writing mesh")
                geo.write_glb(self.output_dir / "model.glb", mesh, "model")
                geo.write_ply(self.output_dir / "model.ply", mesh)
                geo.write_obj(self.output_dir / "model.obj", mesh)
                outputs.update({"mesh_glb": "model.glb", "mesh_ply": "model.ply", "mesh_obj": "model.obj"})
            else:
                self.warnings.append("The mesh was empty after cropping; try turning off 'Isolate the object'.")
        self._set(1.0, "")
        return outputs, stats


def _parse_analyzer(text: str) -> dict:
    fields = {
        "cameras": r"Cameras:\s*(\d+)",
        "images": r"\]\s*Images:\s*(\d+)",
        "registered_images": r"Registered images:\s*(\d+)",
        "points": r"Points:\s*(\d+)",
        "observations": r"Observations:\s*(\d+)",
        "mean_track_length": r"Mean track length:\s*([\d.]+)",
        "mean_reprojection_error": r"Mean reprojection error:\s*([\d.]+)",
    }
    out = {}
    for key, pattern in fields.items():
        m = re.search(pattern, text)
        if m:
            val = m.group(1)
            out[key] = float(val) if "." in val else int(val)
    for key in ("mean_track_length", "mean_reprojection_error"):
        if key in out:
            out[key] = round(float(out[key]), 3)
    return out

