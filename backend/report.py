"""Generates a self-contained HTML report (and a JSON export) for a finished reconstruction."""
from __future__ import annotations

import base64
import html
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import geometry as geo
from .pipeline import STAGES, slugify

FONT_PATH = Path(__file__).resolve().parent.parent / "frontend" / "fonts" / "manrope-variable.woff2"
PREVIEW_POINTS = 6000


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _fmt_int(value) -> str:
    return "-" if value is None else f"{int(value):,}"


def _fmt_secs(value) -> str:
    if value is None:
        return "-"
    value = float(value)
    if value < 60:
        return f"{value:.0f}s"
    m, s = divmod(int(round(value)), 60)
    return f"{m}m {s:02d}s" if m < 60 else f"{m // 60}h {m % 60:02d}m"


def _plural(n, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def file_stem(job) -> str:
    return f"model-forge-{slugify(job.settings.name or 'model')}-{job.id[:6]}"


def build_report_context(job) -> dict:
    """Plain-dict snapshot of a finished job, used for both the HTML and JSON report."""
    result = job.result or {}
    return {
        "job_id": job.id,
        "name": job.settings.name or "Untitled model",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": job.settings.inputs,
        "workspace": result.get("workspace"),
        "outputs": result.get("outputs", {}),
        "settings": result.get("settings", {}),
        "stats": result.get("stats", {}),
        "colmap": result.get("colmap", {}),
        "warnings": result.get("warnings", []),
        "images": result.get("images", []),
    }


def _preview_svg(points_path: Path, axis: str, size: int = 340) -> str:
    """Orthographic dot render of the point cloud (front: X/Y, side: Z/Y), nearest points on top."""
    try:
        cloud = geo.read_ply(points_path)
    except (OSError, geo.GeometryError):
        return "<p class='muted'>Preview unavailable.</p>"
    pts = cloud.points
    if len(pts) == 0:
        return "<p class='muted'>No points.</p>"
    cols = cloud.colors if cloud.colors is not None else np.full((len(pts), 3), 200, dtype=np.uint8)
    if len(pts) > PREVIEW_POINTS:
        pick = np.random.default_rng(1).choice(len(pts), PREVIEW_POINTS, replace=False)
        pts, cols = pts[pick], cols[pick]
    h_idx, depth_idx = (0, 2) if axis == "front" else (2, 0)
    order = np.argsort(pts[:, depth_idx] * (1 if axis == "front" else -1))
    pts, cols = pts[order], cols[order]
    hx, vy = pts[:, h_idx] * (1 if axis == "front" else -1), pts[:, 1]
    lo_x, hi_x = np.percentile(hx, [0.5, 99.5])
    lo_y, hi_y = np.percentile(vy, [0.5, 99.5])
    span = max(hi_x - lo_x, hi_y - lo_y) or 1.0
    pad = 14
    scale = (size - 2 * pad) / span
    cx, cy = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2
    dots = []
    for (x, y), (r, g, b) in zip(np.stack([hx, vy], axis=1), cols):
        px = size / 2 + (x - cx) * scale
        py = size / 2 - (y - cy) * scale
        if 0 <= px <= size and 0 <= py <= size:
            dots.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='1.6' fill='#{r:02x}{g:02x}{b:02x}'/>")
    return (f"<svg viewBox='0 0 {size} {size}' width='100%' role='img' aria-label='{axis} view of the point cloud'>"
            f"{''.join(dots)}</svg>")


def _stage_bars(stage_seconds: dict) -> str:
    items = [(STAGES[k][0], v) for k, v in stage_seconds.items() if k in STAGES and v is not None]
    if not items:
        return "<p class='muted'>No timings recorded.</p>"
    longest = max(v for _, v in items) or 1
    rows = ""
    for label, secs in items:
        pct = max(0.6, 100 * secs / longest)
        rows += (f"<div class='bar-row'><div class='bar-label'>{_esc(label)}</div>"
                 f"<div class='bar-track'><div class='bar' style='width:{pct:.1f}%'></div></div>"
                 f"<div class='bar-value'>{_fmt_secs(secs)}</div></div>")
    return rows


def _card(label: str, value: str, sub: str = "") -> str:
    sub_html = f"<div class='card-sub'>{_esc(sub)}</div>" if sub else ""
    return (f"<div class='card'><div class='card-label'>{_esc(label)}</div>"
            f"<div class='card-value'>{_esc(value)}</div>{sub_html}</div>")


@lru_cache(maxsize=1)
def _font_face_css() -> str:
    """Manrope embedded as a data URI so the saved report stays a single file."""
    try:
        data = base64.b64encode(FONT_PATH.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return (
        '@font-face { font-family: "Manrope"; font-weight: 200 800; font-style: normal; '
        f'font-display: swap; src: url(data:font/woff2;base64,{data}) format("woff2"); }}'
    )


def generate_report_html(job) -> str:
    ctx = build_report_context(job)
    st = ctx["stats"]
    settings = ctx["settings"]
    generated = datetime.fromisoformat(ctx["generated_at"]).strftime("%Y-%m-%d %H:%M UTC")
    points_path = Path(ctx["workspace"] or "") / "output" / "points.ply"

    cards = "".join([
        _card("Images placed", f"{_fmt_int(st.get('registered_images'))} / {_fmt_int(st.get('images'))}",
              f"{_plural(st.get('photos', 0), 'photo')} · {_plural(st.get('videos', 0), 'video')}"),
        _card("Points", _fmt_int(st.get("points")), "dense cloud" if ctx["settings"].get("dense") else "sparse cloud"),
        _card("Mesh triangles", _fmt_int(st.get("mesh_faces")), f"{_fmt_int(st.get('mesh_vertices'))} vertices"),
        _card("Reprojection error", f"{st['mean_reprojection_error']:.2f} px" if st.get("mean_reprojection_error") is not None else "-",
              f"mean track length {st.get('mean_track_length', '-')}"),
        _card("Time", _fmt_secs(st.get("elapsed_seconds")), f"COLMAP {ctx['colmap'].get('version') or ''}".strip()),
    ])

    warnings = "".join(f"<div class='warning'>{_esc(w)}</div>" for w in ctx["warnings"])

    setting_rows = [
        ("Quality", settings.get("quality")),
        ("Frames per video", settings.get("frames_per_video")),
        ("Matching", settings.get("matcher")),
        ("Dense reconstruction", "yes (CUDA)" if settings.get("dense") else "no (sparse points)"),
        ("Mesh method", settings.get("mesher")),
        ("Isolate the object", "yes" if settings.get("crop") else "no"),
        ("Keep largest piece", "yes" if settings.get("keep_largest") else "no"),
    ]
    settings_table = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in setting_rows if v is not None)

    files_table = "".join(
        f"<tr><td>{_esc(name)}</td><td>{_esc(str(Path(ctx['workspace']) / 'output' / name))}</td></tr>"
        for name in ctx["outputs"].values()
    )

    image_rows = ""
    for im in ctx["images"]:
        placed = "<span class='ok'>placed</span>" if im.get("registered") else "<span class='bad'>not placed</span>"
        source = Path(im.get("source", "")).name
        if im.get("time") is not None:
            source += f" @ {im['time']:.1f}s"
        image_rows += f"<tr><td>{_esc(im['name'])}</td><td>{_esc(source)}</td><td>{placed}</td></tr>"

    inputs = "".join(f"<tr><td>{i + 1}</td><td>{_esc(p)}</td></tr>" for i, p in enumerate(ctx["inputs"]))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Model Forge Report — {_esc(ctx['name'])}</title>
<style>
  {_font_face_css()}
  :root {{
    --bg: #1c2023; --panel: #262c30; --panel-2: #30373c;
    --border: rgba(255, 255, 255, 0.08); --border-strong: rgba(255, 255, 255, 0.14);
    --text: #eef1f2; --text-dim: #a9b3b8; --text-faint: #7f8b91;
    --accent: #e8793b; --accent-2: #f39a63; --good: #3ecf8e; --bad: #f05a5a; --warn: #f5b942;
    --mono: Consolas, "Cascadia Mono", "SFMono-Regular", monospace;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; color: var(--text); font-size: 15px; line-height: 1.6;
          background: radial-gradient(90rem 40rem at 50% -12rem, rgba(154, 170, 178, 0.14), transparent 70%), var(--bg);
          font-family: "Manrope", "Segoe UI", system-ui, -apple-system, sans-serif;
          -webkit-font-smoothing: antialiased; }}
  .wrap {{ max-width: 1040px; margin: 0 auto; padding: 56px 24px 64px; }}
  .brand {{ display: flex; align-items: center; gap: 10px; margin-bottom: 40px; font-weight: 600; font-size: 15px; }}
  .brand-mark {{ display: inline-grid; place-items: center; width: 32px; height: 32px; border-radius: 10px;
                 background: var(--accent); color: #16191b; font-weight: 800; font-size: 11.5px; }}
  .eyebrow {{ display: flex; align-items: center; gap: 10px; font-family: var(--mono); font-size: 12.5px;
              color: var(--text-faint); margin-bottom: 18px; }}
  .eyebrow::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
                      box-shadow: 0 0 0 4px rgba(232, 121, 59, 0.18); flex-shrink: 0; }}
  h1 {{ font-size: 52px; font-weight: 500; letter-spacing: -0.035em; line-height: 1.05; margin: 0 0 36px; }}
  h2 {{ font-size: 22px; font-weight: 500; letter-spacing: -0.025em; margin: 0 0 18px; }}
  h3 {{ font-family: var(--mono); font-size: 11.5px; font-weight: 400; margin: 0 0 12px; color: var(--accent-2);
        text-transform: uppercase; letter-spacing: .08em; }}
  .muted {{ color: var(--text-faint); font-family: var(--mono); font-size: 13px; }}
  .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 28px; padding: 28px 30px; margin-bottom: 14px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
  .card {{ background: var(--panel-2); border: 1px solid var(--border); border-radius: 18px; padding: 18px 20px; }}
  .card-label {{ font-family: var(--mono); color: var(--accent-2); font-size: 11.5px; text-transform: uppercase;
                 letter-spacing: .08em; margin-bottom: 10px; }}
  .card-value {{ font-size: 30px; font-weight: 500; letter-spacing: -0.04em; line-height: 1; font-variant-numeric: tabular-nums; }}
  .card-sub {{ margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--border-strong);
               font-family: var(--mono); font-size: 11.5px; color: var(--text-faint); }}
  .warning {{ background: rgba(245, 185, 66, 0.08); border: 1px solid rgba(245, 185, 66, 0.35);
              border-left: 3px solid var(--warn); border-radius: 14px; padding: 12px 16px; margin-top: 12px;
              color: #f8dca0; font-size: 13.5px; }}
  .views {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .view {{ background: #15181a; border: 1px solid var(--border); border-radius: 18px; padding: 10px; }}
  .view-label {{ font-family: var(--mono); font-size: 11.5px; color: var(--text-faint); margin: 2px 6px 4px; }}
  .bar-row {{ display: grid; grid-template-columns: 200px 1fr 70px; align-items: center; gap: 12px; margin-bottom: 8px; }}
  .bar-label {{ font-size: 13.5px; color: var(--text-dim); }}
  .bar-track {{ height: 10px; background: var(--panel-2); border-radius: 999px; overflow: hidden; }}
  .bar {{ height: 100%; background: var(--accent-2); border-radius: 999px; }}
  .bar-value {{ font-family: var(--mono); font-size: 12.5px; color: var(--text-dim); text-align: right; font-variant-numeric: tabular-nums; }}
  table {{ width: 100%; border-collapse: collapse; }}
  td {{ padding: 5px 8px 5px 0; border-bottom: 1px solid var(--border); word-break: break-word; font-size: 13.5px; }}
  tr:last-child td {{ border-bottom: none; }}
  td:first-child {{ color: var(--text-faint); width: 34%; font-family: var(--mono); font-size: 12.5px; }}
  .images td:first-child {{ width: 45%; }}
  .ok {{ color: var(--good); font-family: var(--mono); font-size: 12.5px; }}
  .bad {{ color: var(--bad); font-family: var(--mono); font-size: 12.5px; }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .col {{ background: var(--panel-2); border: 1px solid var(--border); border-radius: 18px; padding: 18px 20px; }}
  .footer {{ margin-top: 28px; padding-top: 20px; border-top: 1px solid var(--border);
             color: var(--text-faint); font-size: 13px; }}
  .footer a {{ color: var(--text-dim); text-decoration: none; }}
  @media (max-width: 700px) {{ .cols, .views {{ grid-template-columns: 1fr; }} h1 {{ font-size: 38px; }}
                                .bar-row {{ grid-template-columns: 1fr 60px; }} .bar-track {{ display: none; }} }}
  @page {{ size: A4; margin: 14mm 12mm; }}
  @media print {{
    body {{ background: white; color: #16191b; font-size: 12.5px; }}
    .wrap {{ max-width: none; padding: 0; }}
    .brand {{ margin-bottom: 18px; }}
    h1 {{ font-size: 34px; margin-bottom: 20px; }}
    .panel {{ background: white; border-color: #d9dee1; border-radius: 16px; padding: 18px 20px; }}
    .col, .card {{ background: #f6f7f8; border-color: #e3e7e9; }}
    .card-label, h3 {{ color: #b3561f; }}
    .eyebrow, .muted, td:first-child, .card-sub, .footer, .bar-label, .bar-value, .view-label {{ color: #5b666c; }}
    .card-sub {{ border-top-color: #d9dee1; }}
    td {{ border-bottom-color: #e3e7e9; }}
    .bar-track {{ background: #e9ecee; }}
    .bar {{ background: #d9652a; }}
    .warning {{ background: #fff7e6; color: #7a5200; border-color: #f0d9a8; }}
    .ok {{ color: #1f9d68; }} .bad {{ color: #c03a3a; }}
    .card, .block, tr, .views, .bar-row {{ break-inside: avoid; }}
    h2 {{ break-after: avoid; }}
    .page-start {{ break-before: page; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="brand"><span class="brand-mark">MF</span><span>Model Forge</span></div>
  <div class="eyebrow">Generated {_esc(generated)} · job {_esc(ctx['job_id'])} · {_esc(settings.get('quality', ''))} quality</div>
  <h1>{_esc(ctx['name'])}</h1>

  <div class="panel">
    <h2>Reconstruction</h2>
    <div class="cards">{cards}</div>
    {warnings}
  </div>

  <div class="panel">
    <h2>Preview</h2>
    <div class="views">
      <div class="view"><div class="view-label">Front</div>{_preview_svg(points_path, 'front')}</div>
      <div class="view"><div class="view-label">Side</div>{_preview_svg(points_path, 'side')}</div>
    </div>
  </div>

  <div class="panel">
    <h2>Where the time went</h2>
    {_stage_bars(st.get('stage_seconds', {}))}
  </div>

  <div class="panel">
    <div class="cols">
      <div class="col"><h3>Settings</h3><table>{settings_table}</table></div>
      <div class="col"><h3>Files</h3><table>{files_table or "<tr><td>none</td><td></td></tr>"}</table></div>
    </div>
  </div>

  <div class="panel page-start">
    <h2>Inputs</h2>
    <table>{inputs}</table>
  </div>

  <div class="panel">
    <h2>Images</h2>
    <table class="images">{image_rows or "<tr><td class='muted'>No images recorded.</td></tr>"}</table>
  </div>

  <div class="footer">Made with Model Forge · COLMAP {_esc(ctx['colmap'].get('version') or '')}
    {'with CUDA' if ctx['colmap'].get('cuda') else 'without CUDA'} ·
    <a href="https://github.com/Harry083/Frame-Guard">github.com/Harry083/Frame-Guard</a></div>
</div>
<script>
  // ?print=1 opens the print dialog straight away (the PDF fallback when no browser is found for export)
  if (new URLSearchParams(location.search).has("print")) window.addEventListener("load", () => window.print());
</script>
</body>
</html>
"""


def generate_report_json(job) -> dict:
    return build_report_context(job)
