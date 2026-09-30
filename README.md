# Model Forge

A desktop application that turns photos and/or videos of an object into a coloured 3D model. It uses
[COLMAP](https://colmap.github.io/) for photogrammetry (working out where every shot was taken, then building a
point cloud and a mesh) and `ffmpeg` to pull frames out of videos. Results can be viewed in the app,
exported as **GLB, OBJ or PLY**, and summarised in an HTML/PDF report.

It opens in its own native window, using the operating system's web engine through
[pywebview](https://pywebview.flowrl.com/) (Edge WebView2 on Windows, WebKit on macOS, WebKitGTK or Qt on
Linux). **No web server runs and no network port is opened**: the window's JavaScript calls the Python
backend directly. It's the sibling of [Quick Capture](../Quick-Capture/README.md) and
[Frame Guard](../README.md): the same vanilla HTML/CSS/JS, the same look, no build step.

## Requirements

- Python 3.10+
- A system web engine:
  - Windows 10/11: Edge WebView2, which is already installed.
  - macOS: nothing extra.
  - Linux: GTK and WebKit2GTK (e.g. `sudo apt install python3-gi gir1.2-webkit2-4.1`), or
    `pip install "pywebview[qt]"`.
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

## Run from source

```bash
.venv\Scripts\python.exe app.py
```

Pass `--debug` to enable the web inspector.

## Build a standalone app

```bash
python -m pip install pyinstaller
python -m PyInstaller --clean ModelForge.spec
```

This builds a single file, `dist/ModelForge.exe` (`dist/ModelForge` on Linux/macOS), with the Model Forge
icon. It runs on another machine without Python installed. COLMAP and ffmpeg are **not** bundled; the app finds
them on `PATH` (or through `COLMAP_BIN` / `FFMPEG_BIN` / `FFPROBE_BIN`) just as it does from source. Each launch
unpacks the app to a temp folder first, so it takes a second or two to open.

- **Windows:** run `ModelForge.exe`. Pin it to the Start menu or taskbar like any other program.
- **macOS:** run `dist/ModelForge`.
- **Linux:** copy `dist/ModelForge` and `modelforge.png` to `/opt/ModelForge/`, then install
  `model-forge.desktop` into `~/.local/share/applications/`.

The icon lives in `modelforge.ico` (every Windows size, 16–256 px) and `modelforge.png` (1024 px). To use a
different one, replace those two files and rebuild.

PyInstaller builds for the OS it runs on, so build the Windows `.exe` on Windows.

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

- **Picking files**: **Add files…** and **Add folder…** open the operating system's own dialog, attached to
  the app window. You can also paste a path.
- **Viewer** (`frontend/viewer.js`): three.js (vendored in `frontend/vendor/three`, MIT) with mesh/point
  toggles, colour/clay/wireframe shading, and markers showing where each photo was taken. With no server to
  fetch from, the GLBs reach it through the bridge in 4 MiB pieces.
- **Saving**: each model file, and the report as PDF or JSON, is saved through the operating system's Save
  dialog. **Open folder** shows the build's `output/` folder.
- **Reports** (`backend/report.py`): stats, front and side previews of the point cloud (inline SVG), time per
  stage, settings, and which images were placed. **View Report** opens it in its own window. **Save PDF** uses
  headless Edge/Chrome like Frame Guard (`MODEL_FORGE_BROWSER` to override). Without either, the report opens
  with the print dialog, where "Save as PDF" works.
- Reloading the window keeps your place: the build's id is in the URL (`#job=…`). Builds run one at a time,
  since COLMAP uses every core. Jobs live in memory, so closing the app forgets them, but files stay on disk.

## Project structure

```
model-forge/
├── backend/
│   ├── api.py            the methods the window calls (window.pywebview.api.*), with input validation
│   ├── pipeline.py       reconstruction stages, progress, presets
│   ├── tools.py          finding/running COLMAP, ffmpeg, ffprobe
│   ├── media.py          photo/video handling, frame extraction, thumbnails
│   ├── sparse_model.py   reads COLMAP's text model; finds the subject, up direction, framing
│   ├── geometry.py       PLY I/O, normals, clean-up, GLB/OBJ export
│   ├── jobs.py           background job manager (progress, cancel)
│   ├── report.py         HTML/JSON report generation
│   └── pdf_export.py     HTML report -> PDF via headless Edge/Chrome
├── frontend/             vanilla HTML/CSS/JS UI + three.js viewer; styles.css + fonts/ are the shared tool style kit
├── app.py                entry point: opens the native window (no server, no port)
├── ModelForge.spec       PyInstaller one-file build
├── modelforge.ico/.png   the app icon
├── model-forge.desktop   Linux menu launcher
└── requirements.txt
```
