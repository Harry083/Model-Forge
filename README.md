# Model Forge

A local web app that turns photos and/or videos of an object into a coloured 3D model. It uses
[COLMAP](https://colmap.github.io/) for photogrammetry (working out where every shot was taken, then building a
point cloud and a mesh) and `ffmpeg` to pull frames out of videos. Results can be viewed in the browser,
exported as **GLB, OBJ or PLY**, and summarised in an HTML/PDF report.

It's the sibling of [Frame Guard](../README.md): the same local FastAPI + vanilla JS setup, the same look.

## Requirements

- Python 3.10+
- **COLMAP 3.8 or newer**, on your `PATH` or pointed to by `COLMAP_BIN`.
  On Windows, download the release zip from <https://github.com/colmap/colmap/releases> (pick the `cuda`
  build if you have an NVIDIA GPU), unzip it, and set `COLMAP_BIN` to the `COLMAP.bat` inside:
  ```powershell
  $env:COLMAP_BIN = "C:\Tools\COLMAP\COLMAP.bat"
  ```
  On Linux `apt install colmap`, on macOS `brew install colmap`.
- `ffmpeg` and `ffprobe` on your `PATH` (for videos and thumbnails; photos alone work without it, minus previews).

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```bash
.venv\Scripts\python.exe run.py
```

Then open http://localhost:8757 (Frame Guard uses 8756, so both can run at once).

## Taking good photos

- 20–80 photos, each overlapping the last by 60–80%. Go all the way round, ideally in two or three loops at
  different heights. A slow 20–60 s video walking around the object works too.
- Keep the object still and move yourself, fill the frame, keep it sharp, and use even light without a flash.
- Shiny, transparent and plain single-colour surfaces are hard for any photogrammetry tool. A textured surface
  under a small object (newspaper, patterned cloth) helps.

## How it works

`backend/pipeline.py` runs these stages as a background job; each one's COLMAP log output is parsed for the
progress bar (`[i/N]` counters, "Registering image #x (n)" and so on).

| Stage | What runs |
|---|---|
| Preparing photos & frames | Photos are hard-linked (or copied) into the workspace; each video's frames are extracted with `ffmpeg -vf fps=…` into their own folder. HEIC/WebP/AVIF photos are converted to JPEG. |
| Finding features | `colmap feature_extractor`: photos first (cameras detected from EXIF), then video frames with `single_camera_per_folder` so each video is one camera. |
| Matching images | `colmap exhaustive_matcher` (every pair: the most robust), or `sequential_matcher` for long videos (Auto picks sequential over 150 frames of video, or 300 images). |
| Placing cameras | `colmap mapper`, incremental structure-from-motion. If the images split into several groups, the largest is used and the rest reported. |
| Undistorting / depth maps / fusing | **CUDA builds only:** `image_undistorter`, `patch_match_stereo` and `stereo_fusion` make a dense point cloud. |
| Building the mesh | `colmap poisson_mesher` (or `delaunay_mesher` for dense runs). **Without CUDA** the sparse points are filtered (track length, reprojection error, statistical outliers), given normals by local PCA oriented toward the cameras that saw them (`backend/geometry.py`), and Poisson-meshed. |
| Cleaning up & exporting | The subject is found where the cameras' viewing rays meet, and everything further than 0.8x the median camera distance is cropped ("Isolate the object"); the largest connected piece of mesh is kept ("Keep largest piece"). The model is turned upright (average camera "up" becomes +Y), rotated so the first photo looks at its front, centred, and scaled so its largest dimension is 1 unit. Then GLB, OBJ (with vertex colours) and PLY are written. |

### Dense vs. sparse

COLMAP's dense stereo needs CUDA. With a CUDA build and an NVIDIA GPU, "Dense reconstruction: Auto" gives
detailed meshes with hundreds of thousands of triangles. Without one, Model Forge still produces a mesh, built
from the sparse points: a good overall shape but coarse detail. The chips under the title show which you have.

### Quality presets

| | Feature image size | Max features | Dense image size | Poisson depth (dense / sparse) | Video frame size |
|---|---|---|---|---|---|
| Draft | 1000 px | 4096 | 1000 px | 9 / 8 | 1280 px |
| Standard | 1600 px | 8192 | 1600 px | 10 / 9 | 1920 px |
| High | 3200 px | 16384 | 2400 px | 11 / 10 | 3840 px |

### Where files go

Each build gets a folder in `~/ModelForge/` (override with `MODEL_FORGE_WORKSPACE`):

```
20260928-1821_ceramic-mug_eaefd7/
├── output/          model.glb, model.obj, model.ply, points.ply, points.glb
├── sparse/          COLMAP sparse model (cameras, images, points)
├── logs/            full COLMAP/ffmpeg output (pipeline.log)
└── images/, database.db, dense/   working files, deleted afterwards unless "Keep working files" is on
```

### Other details

- **Picking files** (`backend/native_dialog.py`): as in Frame Guard, **Add files…** and **Add folder…** ask the
  local server to show the operating system's own dialog (via Tk), since browsers hide real paths. If that
  can't be shown, an in-page browser (`backend/file_browser.py`) is used instead. You can also paste a path.
- **Viewer** (`frontend/viewer.js`): three.js (vendored in `frontend/vendor/three`, MIT) with mesh/point
  toggles, colour/clay/wireframe shading, and markers showing where each photo was taken.
- **Reports** (`backend/report.py`): stats, front and side previews of the point cloud (inline SVG), time per
  stage, settings, and which images were placed. "Download PDF" uses headless Edge/Chrome like Frame Guard
  (`MODEL_FORGE_BROWSER` to override).
- Refreshing the page keeps your place: the build's id is in the URL (`#job=…`). Builds run one at a time,
  since COLMAP uses every core. Jobs live in memory, so restarting the server forgets them, but files stay on disk.

## Project structure

```
model-forge/
├── backend/
│   ├── main.py           FastAPI app & routes
│   ├── pipeline.py       reconstruction stages, progress, presets
│   ├── tools.py          finding/running COLMAP, ffmpeg, ffprobe
│   ├── media.py          photo/video handling, frame extraction, thumbnails
│   ├── sparse_model.py   reads COLMAP's text model; finds the subject, up direction, framing
│   ├── geometry.py       PLY I/O, normals, clean-up, GLB/OBJ export
│   ├── jobs.py           background job manager (progress, cancel)
│   ├── report.py         HTML/JSON report generation
│   ├── pdf_export.py     HTML report -> PDF via headless Edge/Chrome
│   ├── native_dialog.py  native OS file/folder dialogs
│   └── file_browser.py   fallback in-page directory listing
├── frontend/             vanilla HTML/CSS/JS UI + three.js viewer
├── run.py                entry point (uvicorn, port 8757)
└── requirements.txt
```
