"""Opens the operating system's own "Open" dialogs on the machine running the server.

Browsers never reveal a real filesystem path from <input type=file>, but COLMAP and ffmpeg need
one. Because Model Forge's server runs on the user's own machine, it can show the native dialog
itself (via Tk, which ships with Python) and hand the chosen paths back to the page.
"""
from __future__ import annotations

import os
import threading

from .media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


class DialogUnavailable(RuntimeError):
    """No native dialog can be shown here (tkinter missing, or no desktop session)."""


class DialogBusy(RuntimeError):
    """A dialog is already open."""


_lock = threading.Lock()


def _pattern(exts) -> str:
    return " ".join(f"*{ext} *{ext.upper()}" for ext in sorted(exts))


def _run_dialog(show, initial_path: str):
    """Blocks until the dialog closes. Call it from a worker thread: Tk creates and destroys its
    window in the calling thread."""
    if not _lock.acquire(blocking=False):
        raise DialogBusy("A file dialog is already open.")
    try:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as exc:
            raise DialogUnavailable("tkinter is not installed") from exc

        try:
            root = tk.Tk()
        except tk.TclError as exc:  # e.g. no display on a headless machine
            raise DialogUnavailable(str(exc)) from exc

        try:
            # an invisible, topmost owner window so the dialog opens in front of the browser
            root.withdraw()
            root.attributes("-topmost", True)
            root.lift()
            root.focus_force()

            initial_dir = None
            if initial_path:
                candidate = initial_path if os.path.isdir(initial_path) else os.path.dirname(initial_path)
                if os.path.isdir(candidate):
                    initial_dir = candidate
            return show(filedialog, root, initial_dir)
        finally:
            root.destroy()
    finally:
        _lock.release()


def pick_files(title: str, initial_path: str = "") -> list[str]:
    """Photos and/or videos; returns [] when cancelled."""
    def show(filedialog, root, initial_dir):
        return filedialog.askopenfilenames(
            parent=root,
            title=title,
            initialdir=initial_dir,
            filetypes=[
                ("Photos and videos", _pattern(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)),
                ("Photos", _pattern(IMAGE_EXTENSIONS)),
                ("Videos", _pattern(VIDEO_EXTENSIONS)),
                ("All files", "*.*"),
            ],
        )
    paths = _run_dialog(show, initial_path) or ()
    return [os.path.normpath(p) for p in paths]


def pick_folder(title: str, initial_path: str = "") -> str | None:
    def show(filedialog, root, initial_dir):
        return filedialog.askdirectory(parent=root, title=title, initialdir=initial_dir, mustexist=True)
    path = _run_dialog(show, initial_path)
    return os.path.normpath(path) if path else None
