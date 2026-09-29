"""Server-side directory listing so the browser UI can navigate the local filesystem
without relying on <input type=file>, which hides the real path from web pages."""
from __future__ import annotations

import os
import string
from pathlib import Path

from .media import kind_of


def list_drives() -> list[str]:
    if os.name != "nt":
        return ["/"]
    drives = []
    for letter in string.ascii_uppercase:
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            drives.append(drive)
    return drives


def browse(path: str | None) -> dict:
    if not path:
        entries = [{"name": d, "path": d, "type": "dir"} for d in list_drives()]
        return {"path": "", "parent": None, "entries": entries, "media_count": 0}

    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    entries = []
    try:
        children = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        children = []

    media_count = 0
    for child in children:
        try:
            if child.is_dir():
                entries.append({"name": child.name, "path": str(child), "type": "dir"})
            elif kind_of(child):
                media_count += 1
                entries.append({
                    "name": child.name,
                    "path": str(child),
                    "type": "file",
                    "kind": kind_of(child),
                    "size": child.stat().st_size,
                })
        except (PermissionError, OSError):
            continue

    parent = str(p.parent) if p.parent != p else None
    if os.name == "nt" and len(str(p)) <= 3:
        parent = ""  # at a drive root: up goes to the drive list

    return {"path": str(p), "parent": parent, "entries": entries, "media_count": media_count}
