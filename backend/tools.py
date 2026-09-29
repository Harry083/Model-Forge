"""Finding and running the external programs: COLMAP (reconstruction) and ffmpeg/ffprobe (media)."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
from collections import deque
from pathlib import Path
from typing import Callable, Optional

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")


class ToolError(RuntimeError):
    pass


class Cancelled(Exception):
    pass


def find_colmap() -> Optional[str]:
    """COLMAP_BIN wins; otherwise `colmap` on PATH. The Windows release zip ships COLMAP.bat."""
    env = os.environ.get("COLMAP_BIN")
    if env:
        return env if os.path.isfile(env) else shutil.which(env)
    for name in ("colmap", "COLMAP.bat", "colmap.bat"):
        found = shutil.which(name)
        if found:
            return found
    return None


def find_binaries() -> dict:
    return {
        "colmap": find_colmap(),
        "ffmpeg": shutil.which(FFMPEG_BIN),
        "ffprobe": shutil.which(FFPROBE_BIN),
    }


_info_cache: dict = {}
_help_cache: dict[str, str] = {}


def _run_sync(args: list[str], timeout: float = 30) -> str:
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolError(f"Could not run {args[0]}: {exc}") from exc
    return (proc.stdout + proc.stderr).decode("utf-8", "replace")


def colmap_info() -> dict:
    """Version and whether this COLMAP build has CUDA (needed for dense depth maps)."""
    path = find_colmap()
    if not path:
        return {"found": False}
    if _info_cache.get("path") != path:
        text = _run_sync([path, "-h"])
        version = re.search(r"COLMAP\s+([\w.\-]+)", text)
        _info_cache.clear()
        _info_cache.update({
            "path": path,
            "found": True,
            "version": version.group(1) if version else None,
            "cuda": "with CUDA" in text,
        })
    return {k: v for k, v in _info_cache.items() if k != "path"}


def colmap_option(command: str, *candidates: str) -> Optional[str]:
    """Returns whichever option name this COLMAP version accepts (names moved between releases,
    e.g. SiftExtraction.use_gpu became FeatureExtraction.use_gpu)."""
    if command not in _help_cache:
        _help_cache[command] = _run_sync([find_colmap() or "colmap", command, "-h"])
    text = _help_cache[command]
    for name in candidates:
        if f"--{name} " in text or f"--{name}\n" in text:
            return name
    return None


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        if os.name == "nt":  # COLMAP.bat runs colmap.exe as a child of cmd.exe
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


async def run_process(
    args: list[str],
    *,
    on_line: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[asyncio.Event] = None,
    log_path: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> str:
    """Runs a program, feeding each output line to on_line. Returns the tail of its output.
    Raises ToolError on a non-zero exit and Cancelled if cancel_event is set meanwhile."""
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
            cwd=str(cwd) if cwd else None,
            **kwargs,
        )
    except OSError as exc:
        raise ToolError(f"Could not start {args[0]}: {exc}") from exc

    tail: deque[str] = deque(maxlen=60)
    log = open(log_path, "a", encoding="utf-8") if log_path else None
    if log:
        log.write("$ " + " ".join(f'"{a}"' if " " in a else a for a in args) + "\n")

    async def pump():
        assert proc.stdout is not None
        buf = b""
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            *lines, buf = re.split(rb"\r\n|\r|\n", buf)
            for raw in lines:
                line = raw.decode("utf-8", "replace")
                tail.append(line)
                if log:
                    log.write(line + "\n")
                if on_line and line:
                    on_line(line)
        if buf:
            tail.append(buf.decode("utf-8", "replace"))

    pump_task = asyncio.create_task(pump())
    try:
        waiters = {pump_task}
        cancel_task = None
        if cancel_event is not None:
            cancel_task = asyncio.create_task(cancel_event.wait())
            waiters.add(cancel_task)
        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        if cancel_event is not None and cancel_event.is_set():
            _kill_tree(proc)
            await proc.wait()
            pump_task.cancel()
            raise Cancelled()
        if cancel_task:
            cancel_task.cancel()
        await proc.wait()
    except asyncio.CancelledError:
        _kill_tree(proc)
        raise
    finally:
        if log:
            log.close()

    output = "\n".join(tail)
    if proc.returncode != 0:
        name = Path(args[0]).stem
        cmd = args[1] if name.lower() == "colmap" and len(args) > 1 else name
        raise ToolError(f"{cmd} failed (exit code {proc.returncode}):\n{output[-3000:]}")
    return output


async def colmap(command: str, options: dict, **kwargs) -> str:
    path = find_colmap()
    if not path:
        raise ToolError("COLMAP was not found. Install it and put it on PATH, or set COLMAP_BIN.")
    args = [path, command]
    for key, value in options.items():
        if key is None or value is None:
            continue
        if isinstance(value, bool):
            value = 1 if value else 0
        args += [f"--{key}", str(value)]
    return await run_process(args, **kwargs)


async def probe(path: str) -> dict:
    """ffprobe summary: duration and first video stream's size/frame rate."""
    args = [FFPROBE_BIN, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except OSError as exc:
        raise ToolError(f"Could not run ffprobe: {exc}") from exc
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise ToolError(err.decode("utf-8", "replace").strip() or "ffprobe failed")
    try:
        return json.loads(out.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise ToolError(f"Could not parse ffprobe output: {exc}") from exc
