"""Turning the user's photos and videos into one folder of still images for COLMAP."""
from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import tools

# formats COLMAP reads directly (through FreeImage)
NATIVE_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# formats converted to JPEG with ffmpeg first
CONVERTED_IMAGE_EXTENSIONS = {".webp", ".heic", ".heif", ".avif", ".jfif"}
IMAGE_EXTENSIONS = NATIVE_IMAGE_EXTENSIONS | CONVERTED_IMAGE_EXTENSIONS
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mts", ".m2ts", ".ts", ".wmv", ".mpg", ".mpeg", ".3gp",
}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def kind_of(path: str | Path) -> Optional[str]:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def folder_media(folder: Path) -> list[Path]:
    """Photos and videos directly inside a folder (not sub-folders), in name order."""
    try:
        return sorted(
            (p for p in folder.iterdir() if p.is_file() and kind_of(p)),
            key=lambda p: p.name.lower(),
        )
    except OSError:
        return []


def expand(paths: list[str]) -> list[Path]:
    """Files as given, folders replaced by the media inside them; duplicates dropped."""
    seen: set[str] = set()
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        items = folder_media(p) if p.is_dir() else [p]
        for item in items:
            key = os.path.normcase(str(item.resolve()))
            if key not in seen and kind_of(item):
                seen.add(key)
                out.append(item)
    return out


async def video_duration(path: Path) -> float:
    data = await tools.probe(str(path))
    duration = float(data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and s.get("duration"):
                duration = float(s["duration"])
    return duration


async def inspect(path: str) -> dict:
    """What the UI shows for one entry in the capture list."""
    p = Path(path)
    if p.is_dir():
        items = folder_media(p)
        if not items:
            raise tools.ToolError(f"No photos or videos directly inside {p.name or p}")
        return {
            "path": str(p),
            "name": p.name or str(p),
            "kind": "folder",
            "images": sum(1 for i in items if kind_of(i) == "image"),
            "videos": sum(1 for i in items if kind_of(i) == "video"),
        }
    if not p.is_file():
        raise tools.ToolError(f"Not found: {path}")
    kind = kind_of(p)
    if kind is None:
        raise tools.ToolError(f"Not a supported photo or video: {p.name}")
    info = {"path": str(p), "name": p.name, "kind": kind, "size": p.stat().st_size}
    try:
        data = await tools.probe(str(p))
        stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        info["width"], info["height"] = stream.get("width"), stream.get("height")
        if kind == "video":
            info["duration"] = float(data.get("format", {}).get("duration") or stream.get("duration") or 0)
    except tools.ToolError:
        pass  # dimensions are a nicety; COLMAP will still try to read the file
    return info


async def thumbnail(path: str, size: int = 240) -> bytes:
    """A small JPEG preview (first frame for videos)."""
    p = Path(path)
    if not p.is_file() or kind_of(p) is None:
        raise tools.ToolError("Not a photo or video")
    args = [tools.FFMPEG_BIN, "-v", "error"]
    if kind_of(p) == "video":
        args += ["-ss", "0.5"]
    args += ["-i", str(p), "-frames:v", "1",
             "-vf", f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}",
             "-q:v", "5", "-f", "image2", "-c:v", "mjpeg", "pipe:1"]
    proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                stdin=asyncio.subprocess.DEVNULL, **tools.NO_WINDOW)
    out, err = await proc.communicate()
    if proc.returncode != 0 or not out:
        raise tools.ToolError(err.decode("utf-8", "replace").strip() or "Could not make a thumbnail")
    return out


@dataclass
class StagedImage:
    name: str      # path relative to the COLMAP image folder, e.g. "photos/0003_IMG_1044.jpg"
    source: str    # the original photo or video
    frame_time: Optional[float] = None


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


async def stage_images(
    sources: list[Path],
    image_dir: Path,
    *,
    frames_per_video: int,
    max_size: int,
    on_progress: Callable[[int, int, str], None],
    cancel_event: asyncio.Event,
    log_path: Path,
) -> tuple[list[StagedImage], list[str]]:
    """Photos go into photos/, each video's frames into their own video_NN/ folder (so COLMAP
    can treat each video as a single camera). Returns the staged images and the video folders."""
    photos_dir = image_dir / "photos"
    staged: list[StagedImage] = []
    video_folders: list[str] = []
    total = len(sources)

    for i, src in enumerate(sources):
        on_progress(i, total, src.name)
        kind = kind_of(src)
        if kind == "image":
            photos_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{len(staged):04d}_{src.stem}"
            if src.suffix.lower() in NATIVE_IMAGE_EXTENSIONS:
                dst = photos_dir / f"{stem}{src.suffix.lower()}"
                _link_or_copy(src, dst)
            else:
                dst = photos_dir / f"{stem}.jpg"
                await tools.run_process(
                    [tools.FFMPEG_BIN, "-v", "error", "-y", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(dst)],
                    cancel_event=cancel_event, log_path=log_path,
                )
            staged.append(StagedImage(f"photos/{dst.name}", str(src)))
        elif kind == "video":
            folder = f"video_{len(video_folders) + 1:02d}"
            out_dir = image_dir / folder
            out_dir.mkdir(parents=True, exist_ok=True)
            duration = await video_duration(src)
            rate = frames_per_video / duration if duration > 0 else 2.0
            scale = f"scale='if(gt(iw,ih),min({max_size},iw),-2)':'if(gt(iw,ih),-2,min({max_size},ih))'"
            await tools.run_process(
                [tools.FFMPEG_BIN, "-v", "error", "-y", "-i", str(src),
                 "-vf", f"fps={rate:.6f},{scale}", "-frames:v", str(frames_per_video),
                 "-q:v", "2", str(out_dir / "frame_%04d.jpg")],
                cancel_event=cancel_event, log_path=log_path,
            )
            frames = sorted(out_dir.glob("frame_*.jpg"))
            if not frames:
                raise tools.ToolError(f"No frames could be read from {src.name}")
            video_folders.append(folder)
            for n, frame in enumerate(frames):
                staged.append(StagedImage(f"{folder}/{frame.name}", str(src), round(n / rate, 2)))
    on_progress(total, total, "")
    return staged, video_folders
