"""Turns the HTML report into a PDF using a Chromium-based browser already on the machine.

Microsoft Edge ships with Windows 10/11, so on Windows this needs nothing extra installed. Chrome,
Chromium or Brave work too, and MODEL_FORGE_BROWSER can point at any Chromium-based executable.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

from .tools import NO_WINDOW


class PdfUnavailable(RuntimeError):
    """No Chromium-based browser was found to render the PDF."""


class PdfError(RuntimeError):
    """The browser ran but did not produce a PDF."""


def _candidate_paths() -> list[str]:
    candidates = []
    env = os.environ.get("MODEL_FORGE_BROWSER")
    if env:
        candidates.append(env)

    if os.name == "nt":
        roots = [
            os.environ.get("PROGRAMFILES(X86)"),
            os.environ.get("PROGRAMFILES"),
            os.environ.get("LOCALAPPDATA"),
        ]
        for root in filter(None, roots):
            candidates += [
                os.path.join(root, "Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join(root, "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(root, "BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
            ]
    else:
        candidates += [
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]

    for name in ("msedge", "microsoft-edge", "google-chrome", "chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    return candidates


def find_browser() -> str | None:
    for path in _candidate_paths():
        if path and os.path.isfile(path):
            return path
    return None


async def html_to_pdf(html: str, timeout: float = 90.0) -> bytes:
    browser = find_browser()
    if browser is None:
        raise PdfUnavailable(
            "No Microsoft Edge or Chrome found to create the PDF. "
            "Set MODEL_FORGE_BROWSER to a Chromium-based browser's path."
        )

    with tempfile.TemporaryDirectory(prefix="model-forge-pdf-") as tmp:
        tmp_dir = Path(tmp)
        html_path = tmp_dir / "report.html"
        pdf_path = tmp_dir / "report.pdf"
        html_path.write_text(html, encoding="utf-8")

        args = [
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-pdf-header-footer",
            # a throwaway profile, so this never attaches to a browser window the user already has open
            f"--user-data-dir={tmp_dir / 'profile'}",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ]
        if os.name != "nt" and os.geteuid() == 0:
            args.insert(0, "--no-sandbox")  # Chromium refuses to start as root otherwise

        proc = await asyncio.create_subprocess_exec(
            browser,
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            **NO_WINDOW,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise PdfError("The browser took too long to create the PDF.") from exc

        if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
            detail = stderr.decode("utf-8", "replace").strip()[-500:]
            raise PdfError(f"The browser did not produce a PDF (exit code {proc.returncode}). {detail}".strip())
        return pdf_path.read_bytes()
