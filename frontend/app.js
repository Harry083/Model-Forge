import { createViewer } from "./viewer.js";

const state = {
  items: [], // {path, name, kind, ...inspect info}
  jobId: null,
  pollTimer: null,
  cuda: false,
  viewer: null,
};

const $ = (sel) => document.querySelector(sel);

const mediaGrid = $("#media-grid");
const mediaEmpty = $("#media-empty");
const captureSummary = $("#capture-summary");
const captureStatus = $("#capture-status");
const pathInput = $("#path-input");
const runBtn = $("#run-btn");
const cancelBtn = $("#cancel-btn");
const progressSection = $("#progress-section");
const progressFill = $("#progress-bar-fill");
const progressStage = $("#progress-stage");
const progressLabel = $("#progress-label");
const progressPct = $("#progress-pct");
const stageList = $("#stage-list");
const logDetails = document.querySelector(".log-details");
const logOutput = $("#log-output");
const errorSection = $("#error-section");
const resultsSection = $("#results-section");
const framesSelect = $("#frames-select");
const denseSelect = $("#dense-select");
const mesherSelect = $("#mesher-select");

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function formatBytes(bytes) {
  if (bytes === undefined || bytes === null) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let val = bytes;
  let i = 0;
  while (val >= 1024 && i < units.length - 1) {
    val /= 1024;
    i++;
  }
  return `${val.toFixed(val < 10 && i > 0 ? 2 : 0)} ${units[i]}`;
}

function formatDuration(secs) {
  if (!secs && secs !== 0) return "";
  secs = Math.round(secs);
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  return m ? `${m}:${String(s).padStart(2, "0")}` : `0:${String(s).padStart(2, "0")}`;
}

function formatElapsed(secs) {
  secs = Math.round(secs || 0);
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  if (m < 60) return `${m}m ${String(secs % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const fmtInt = (n) => (n === null || n === undefined ? "-" : Number(n).toLocaleString());

// ---------- backend calls ----------
// The page talks to Python directly through pywebview's bridge; there is no HTTP server or port.
const bridgeReady = new Promise((resolve) => {
  if (window.pywebview && window.pywebview.api) resolve();
  else window.addEventListener("pywebviewready", resolve, { once: true });
});

async function api(method, ...args) {
  await bridgeReady;
  const res = await window.pywebview.api[method](...args);
  if (!res || !res.ok) throw new Error((res && res.error) || `${method} failed`);
  return res.data;
}

// An output file, fetched in base64 pieces and reassembled (a file:// page can't fetch it directly).
async function readOutput(jobId, name) {
  const parts = [];
  let offset = 0;
  let size = 0;
  do {
    const chunk = await api("model_chunk", jobId, name, offset);
    const bin = atob(chunk.data);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    parts.push(bytes);
    size = chunk.size;
    if (chunk.end === offset) break; // file shrank underneath us
    offset = chunk.end;
  } while (offset < size);
  return new Blob(parts).arrayBuffer();
}

// Thumbnails come back as data: URLs; a few at a time so a big folder doesn't start hundreds of ffmpegs.
const thumbQueue = [];
const thumbCache = new Map(); // path -> data: URL (the grid is rebuilt on every add/remove)
let thumbsActive = 0;
function queueThumb(img, path) {
  if (thumbCache.has(path)) {
    img.src = thumbCache.get(path);
    return;
  }
  thumbQueue.push({ img, path });
  pumpThumbs();
}
function pumpThumbs() {
  while (thumbsActive < 4 && thumbQueue.length) {
    const { img, path } = thumbQueue.shift();
    if (!img.isConnected) continue;
    thumbsActive++;
    api("thumb", path)
      .then((data) => {
        thumbCache.set(path, data.src);
        img.src = data.src;
      })
      .catch(() => img.remove())
      .finally(() => {
        thumbsActive--;
        pumpThumbs();
      });
  }
}

// ---------- environment ----------
async function loadHealth() {
  const chips = $("#env-chips");
  try {
    const data = await api("health");
    const c = data.colmap || {};
    state.cuda = !!c.cuda;
    const out = [];
    out.push(c.found
      ? `<span class="chip ok">COLMAP ${escapeHtml(c.version || "")}</span>`
      : `<span class="chip bad">COLMAP not found</span>`);
    if (c.found) {
      out.push(c.cuda
        ? `<span class="chip ok">CUDA · dense meshes</span>`
        : `<span class="chip warn">No CUDA · meshes from sparse points</span>`);
    }
    out.push(data.binaries.ffmpeg
      ? `<span class="chip ok">ffmpeg</span>`
      : `<span class="chip ${hasVideos() ? "bad" : "warn"}">ffmpeg not found · needed for video</span>`);
    out.push(`<span class="chip" title="Where models are saved">${escapeHtml(data.workspace)}</span>`);
    chips.innerHTML = out.join("");
    if (!c.cuda) {
      denseSelect.querySelector('option[value="on"]').disabled = true;
    }
    syncMesherOptions();
  } catch (e) {
    chips.innerHTML = `<span class="chip bad">${escapeHtml(e.message)}</span>`;
  }
}

function syncMesherOptions() {
  const dense = denseSelect.value === "on" || (denseSelect.value === "auto" && state.cuda);
  const delaunay = mesherSelect.querySelector('option[value="delaunay"]');
  delaunay.disabled = !dense;
  if (!dense) mesherSelect.value = "poisson";
}
denseSelect.addEventListener("change", syncMesherOptions);

// ---------- capture list ----------
function hasVideos() {
  return state.items.some((i) => i.kind === "video" || (i.kind === "folder" && i.videos));
}

function normPath(p) {
  return p.replace(/[\\/]+$/, "").toLowerCase();
}

async function addPaths(paths) {
  const known = new Set(state.items.map((i) => normPath(i.path)));
  const fresh = [...new Set(paths.map((p) => p.trim()).filter(Boolean))].filter((p) => !known.has(normPath(p)));
  if (!fresh.length) return;
  captureStatus.textContent = `Reading ${fresh.length} item${fresh.length === 1 ? "" : "s"}…`;
  captureStatus.className = "file-status";
  try {
    const data = await api("inspect", fresh);
    const errors = data.items.filter((i) => i.kind === "error");
    state.items.push(...data.items.filter((i) => i.kind !== "error"));
    if (errors.length) {
      captureStatus.textContent = errors.map((e) => e.error).join(" · ");
      captureStatus.className = "file-status err";
    } else {
      captureStatus.textContent = "";
    }
  } catch (e) {
    captureStatus.textContent = e.message;
    captureStatus.className = "file-status err";
  }
  renderMedia();
}

function removeItem(path) {
  state.items = state.items.filter((i) => i.path !== path);
  renderMedia();
}

function renderMedia() {
  mediaGrid.querySelectorAll(".media-item").forEach((el) => el.remove());
  mediaEmpty.classList.toggle("hidden", state.items.length > 0);

  for (const item of state.items) {
    const el = document.createElement("div");
    el.className = `media-item ${item.kind}`;
    el.title = item.path;
    let inner = "";
    if (item.kind === "folder") {
      const parts = [];
      if (item.images) parts.push(`${item.images} photo${item.images === 1 ? "" : "s"}`);
      if (item.videos) parts.push(`${item.videos} video${item.videos === 1 ? "" : "s"}`);
      inner = `<div><div class="folder-glyph" aria-hidden="true">▣</div>
        <div class="folder-count">${escapeHtml(parts.join(" · ") || "empty")}</div></div>`;
    } else {
      inner = `<img alt="" />`;
      if (item.kind === "video") {
        inner += `<span class="media-badge">▶ ${escapeHtml(formatDuration(item.duration) || "video")}</span>`;
      }
    }
    inner += `<div class="media-caption">${escapeHtml(item.name)}</div>
      <button class="media-remove" type="button" aria-label="Remove ${escapeHtml(item.name)}">&times;</button>`;
    el.innerHTML = inner;
    el.querySelector(".media-remove").addEventListener("click", () => removeItem(item.path));
    mediaGrid.appendChild(el);
    const img = el.querySelector("img");
    if (img) {
      img.addEventListener("error", () => img.remove());
      queueThumb(img, item.path);
    }
  }
  renderSummary();
}

function estimateImages() {
  const perVideo = Number(framesSelect.value);
  let photos = 0;
  let videos = 0;
  for (const i of state.items) {
    if (i.kind === "image") photos++;
    else if (i.kind === "video") videos++;
    else if (i.kind === "folder") {
      photos += i.images || 0;
      videos += i.videos || 0;
    }
  }
  return { photos, videos, total: photos + videos * perVideo };
}

function renderSummary() {
  const { photos, videos, total } = estimateImages();
  runBtn.disabled = !(state.items.length && total >= 3) || !!state.pollTimer;
  if (!state.items.length) {
    captureSummary.innerHTML = "";
    return;
  }
  const parts = [];
  if (photos) parts.push(`${photos} photo${photos === 1 ? "" : "s"}`);
  if (videos) parts.push(`${videos} video${videos === 1 ? "" : "s"} (~${videos * Number(framesSelect.value)} frames)`);
  let html = `<span>${parts.join(" · ") || "no photos"}</span><span>→ ${total} images to process</span>`;
  if (total > 0 && total < 12) html += `<span class="note-warn">Few images — 20 or more gives far better results</span>`;
  if (total > 400) html += `<span class="note-warn">Many images — expect a long run; Draft quality helps</span>`;
  html += `<button class="clear-link" type="button" id="clear-btn">Clear all</button>`;
  captureSummary.innerHTML = html;
  $("#clear-btn").addEventListener("click", () => {
    state.items = [];
    renderMedia();
  });
}
framesSelect.addEventListener("change", renderSummary);

$("#path-add-btn").addEventListener("click", () => {
  addPaths([pathInput.value]);
  pathInput.value = "";
});
pathInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    addPaths([pathInput.value]);
    pathInput.value = "";
  }
});

function lastDir() {
  const last = state.items[state.items.length - 1];
  return last ? last.path : "";
}

// Add files / folder open the operating system's own dialog, attached to the app window.
async function withButton(btn, busyLabel, fn) {
  const label = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = busyLabel;
  try {
    await fn();
  } catch (e) {
    captureStatus.textContent = e.message;
    captureStatus.className = "file-status err";
  } finally {
    btn.disabled = false;
    btn.innerHTML = label;
  }
}

$("#add-files-btn").addEventListener("click", (e) =>
  withButton(e.currentTarget, "Opening…", async () => {
    const data = await api("pick_files", lastDir());
    if (data.paths?.length) await addPaths(data.paths);
  })
);
$("#add-folder-btn").addEventListener("click", (e) =>
  withButton(e.currentTarget, "Opening…", async () => {
    const data = await api("pick_folder", lastDir());
    if (data.path) await addPaths([data.path]);
  })
);

// ---------- run ----------
runBtn.addEventListener("click", startBuild);
cancelBtn.addEventListener("click", async () => {
  if (!state.jobId) return;
  cancelBtn.disabled = true;
  try {
    await api("cancel", state.jobId);
  } catch (e) {
    console.error(e);
  }
});

async function startBuild() {
  errorSection.classList.add("hidden");
  resultsSection.classList.add("hidden");
  progressSection.classList.remove("hidden");
  progressFill.style.width = "0%";
  progressPct.textContent = "0%";
  progressStage.textContent = "Starting…";
  progressLabel.textContent = "";
  stageList.innerHTML = "";
  logOutput.textContent = "";
  runBtn.disabled = true;
  cancelBtn.disabled = false;
  cancelBtn.classList.remove("hidden");

  const body = {
    inputs: state.items.map((i) => i.path),
    name: $("#name-input").value.trim(),
    quality: $("#quality-select").value,
    frames_per_video: Number(framesSelect.value),
    matcher: $("#matcher-select").value,
    dense: denseSelect.value,
    mesher: mesherSelect.value,
    crop: $("#crop-check").checked,
    keep_largest: $("#largest-check").checked,
    keep_workspace: $("#keep-check").checked,
  };

  try {
    const data = await api("reconstruct", body);
    state.jobId = data.job_id;
    history.replaceState(null, "", `#job=${data.job_id}`);
    pollJob();
    progressSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    showError(e.message);
    resetControls();
  }
}

function pollJob() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  const tick = async () => {
    try {
      const logLines = logDetails.open ? 120 : 0;
      const job = await api("job", state.jobId, logLines);
      updateProgress(job);
      if (job.status === "done") {
        stopPolling();
        showResults(job);
      } else if (job.status === "error") {
        stopPolling();
        showError(job.error || "Reconstruction failed");
      } else if (job.status === "cancelled") {
        stopPolling();
        progressStage.textContent = "Cancelled";
        progressLabel.textContent = "";
      }
    } catch (e) {
      stopPolling();
      showError(e.message);
    }
  };
  state.pollTimer = setInterval(tick, 800);
  tick();
}

function stopPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = null;
  resetControls();
}

function updateProgress(job) {
  const pct = Math.max(0, Math.min(100, job.percent || 0));
  progressFill.style.width = `${pct}%`;
  progressPct.textContent = `${Math.floor(pct)}%`;
  progressStage.textContent = job.status === "queued" ? "Waiting for the previous build…" : job.stage_label;
  progressLabel.textContent = [job.detail, formatElapsed(job.elapsed)].filter(Boolean).join(" · ");

  const activeIdx = job.stages.findIndex((s) => s.key === job.stage);
  stageList.innerHTML = job.stages
    .map((s, i) => {
      let cls = "";
      if (job.status === "done" || (activeIdx >= 0 && i < activeIdx)) cls = "done";
      else if (i === activeIdx) cls = job.status === "error" ? "failed" : "active";
      return `<li class="${cls}">${escapeHtml(s.label)}</li>`;
    })
    .join("");

  if (job.log) {
    const atBottom = logOutput.scrollTop + logOutput.clientHeight >= logOutput.scrollHeight - 8;
    logOutput.textContent = job.log.join("\n");
    if (atBottom) logOutput.scrollTop = logOutput.scrollHeight;
  }
}

function resetControls() {
  renderSummary();
  cancelBtn.classList.add("hidden");
}

function showError(message) {
  errorSection.textContent = message;
  errorSection.classList.remove("hidden");
}

// ---------- results ----------
const OUTPUT_INFO = {
  mesh_glb: { ext: "GLB", name: "Mesh · glTF", desc: "Web, Blender, Windows 3D Viewer" },
  mesh_obj: { ext: "OBJ", name: "Mesh · OBJ", desc: "Vertex colours, most 3D tools" },
  mesh_ply: { ext: "PLY", name: "Mesh · PLY", desc: "MeshLab, CloudCompare" },
  points_ply: { ext: "PLY", name: "Point cloud", desc: "Coloured points" },
};

function card(label, value, sub) {
  return `<div class="score-card">
    <div class="metric-name">${escapeHtml(label)}</div>
    <div class="metric-value">${escapeHtml(value)}</div>
    ${sub ? `<div class="metric-sub"><span>${escapeHtml(sub)}</span></div>` : ""}
  </div>`;
}

async function showResults(job) {
  const result = job.result;
  const st = result.stats;
  progressSection.classList.add("hidden");
  resultsSection.classList.remove("hidden");
  $("#results-title").textContent = job.name || "Your model";

  viewReportBtn.dataset.jobId = job.id;
  pdfBtn.dataset.jobId = job.id;
  openFolderBtn.dataset.jobId = job.id;

  const placedCls = st.registered_images / st.images;
  $("#score-cards").innerHTML = [
    card("Images placed", `${fmtInt(st.registered_images)} / ${fmtInt(st.images)}`,
      placedCls < 0.6 ? "many images could not be placed" : `${plural(st.photos, "photo")} · ${plural(st.videos, "video")}`),
    card("Points", fmtInt(st.points), result.dense ? "dense cloud" : "sparse cloud"),
    card("Mesh", st.mesh_faces ? fmtInt(st.mesh_faces) : "-", st.mesh_faces ? `triangles · ${fmtInt(st.mesh_vertices)} vertices` : "no mesh"),
    card("Reprojection error", st.mean_reprojection_error != null ? `${st.mean_reprojection_error.toFixed(2)} px` : "-", "lower is better · under 1.5 is good"),
    card("Time", formatElapsed(st.elapsed_seconds), result.settings.quality + " quality"),
  ].join("");

  $("#warnings").innerHTML = (result.warnings || []).map((w) => `<div class="warning">${escapeHtml(w)}</div>`).join("");

  const outputs = result.outputs || {};
  $("#download-list").innerHTML = Object.entries(OUTPUT_INFO)
    .filter(([key]) => outputs[key])
    .map(([key, info]) => `<button class="download-item" type="button" data-file="${escapeHtml(outputs[key])}">
        <span class="download-ext">${info.ext}</span>
        <span class="download-text"><span class="download-name">${info.name}</span><span class="download-desc">${info.desc}</span></span>
      </button>`)
    .join("") + `<button class="download-item" type="button" data-report="json">
        <span class="download-ext">JSON</span>
        <span class="download-text"><span class="download-name">Report data</span><span class="download-desc">Stats, settings, cameras</span></span>
      </button>`;
  // each one asks where to save through the operating system's Save dialog, then copies the file there
  $("#download-list").querySelectorAll(".download-item").forEach((btn) =>
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        const data = btn.dataset.report
          ? await api("save_report", job.id, btn.dataset.report)
          : await api("save_output", job.id, btn.dataset.file);
        if (data.path) $("#workspace-path").textContent = `Saved ${data.path}`;
      } catch (e) {
        showError(e.message);
      } finally {
        btn.disabled = false;
      }
    })
  );
  $("#workspace-path").textContent = `Saved in ${result.workspace}`;

  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  await loadViewer(job.id, outputs, result.cameras);
}

async function loadViewer(jobId, outputs, cameras) {
  const loading = $("#viewer-loading");
  loading.textContent = "Loading model…";
  loading.classList.remove("hidden");
  try {
    if (!state.viewer) {
      state.viewer = createViewer($("#viewer-canvas"));
      wireViewerToolbar();
    }
    const [meshGlb, pointsGlb] = await Promise.all([
      outputs.mesh_glb ? readOutput(jobId, outputs.mesh_glb) : null,
      outputs.points_glb ? readOutput(jobId, outputs.points_glb) : null,
    ]);
    const info = await state.viewer.load({ meshGlb, pointsGlb, cameras });
    document.querySelector('[data-show="mesh"]').disabled = !info.hasMesh;
    document.querySelectorAll("[data-shade]").forEach((b) => (b.disabled = !info.hasMesh));
    setActive("[data-show]", info.hasMesh ? "mesh" : "points", "show");
    state.viewer.setShow(info.hasMesh ? "mesh" : "points");
    loading.classList.add("hidden");
  } catch (e) {
    loading.textContent = `Could not show the model: ${e.message}`;
  }
}

function setActive(selector, value, key) {
  document.querySelectorAll(selector).forEach((b) => b.classList.toggle("active", b.dataset[key] === value));
}

function wireViewerToolbar() {
  document.querySelectorAll("[data-show]").forEach((btn) =>
    btn.addEventListener("click", () => {
      setActive("[data-show]", btn.dataset.show, "show");
      state.viewer.setShow(btn.dataset.show);
      document.querySelectorAll("[data-shade]").forEach((b) => (b.disabled = btn.dataset.show !== "mesh"));
    })
  );
  document.querySelectorAll("[data-shade]").forEach((btn) =>
    btn.addEventListener("click", () => {
      setActive("[data-shade]", btn.dataset.shade, "shade");
      state.viewer.setShade(btn.dataset.shade);
    })
  );
  const camerasBtn = $("#cameras-btn");
  camerasBtn.addEventListener("click", () => {
    camerasBtn.classList.toggle("active");
    state.viewer.setCameras(camerasBtn.classList.contains("active"));
  });
  const spinBtn = $("#spin-btn");
  spinBtn.addEventListener("click", () => {
    spinBtn.classList.toggle("active");
    state.viewer.setSpin(spinBtn.classList.contains("active"));
  });
  $("#reset-btn").addEventListener("click", () => state.viewer.reset());
}

// ---------- report / folder ----------
const viewReportBtn = $("#report-view-btn");
viewReportBtn.addEventListener("click", async () => {
  const jobId = viewReportBtn.dataset.jobId;
  if (!jobId) return;
  try {
    await api("view_report", jobId); // opens in its own window
  } catch (e) {
    showError(e.message);
  }
});

const pdfBtn = $("#report-pdf-btn");
pdfBtn.addEventListener("click", async () => {
  const jobId = pdfBtn.dataset.jobId;
  if (!jobId) return;
  const label = pdfBtn.textContent;
  pdfBtn.disabled = true;
  pdfBtn.textContent = "Preparing PDF…";
  try {
    // renders with headless Edge/Chrome, then asks where to save. With neither installed, the report
    // opens with the print dialog instead, where "Save as PDF" works.
    const data = await api("save_report", jobId, "pdf");
    if (data.path) $("#workspace-path").textContent = `Saved ${data.path}`;
  } catch (e) {
    showError(e.message);
  } finally {
    pdfBtn.disabled = false;
    pdfBtn.textContent = label;
  }
});

const openFolderBtn = $("#open-folder-btn");
openFolderBtn.addEventListener("click", async () => {
  const jobId = openFolderBtn.dataset.jobId;
  if (!jobId) return;
  try {
    await api("open_folder", jobId);
  } catch (e) {
    showError(e.message);
  }
});

// ---------- init ----------
loadHealth();
renderMedia();

// #job=<id> reopens a build (so reloading the window mid-run, or after it, keeps your place)
const resumeId = /job=([0-9a-f]+)/.exec(location.hash)?.[1];
if (resumeId) {
  api("job", resumeId)
    .then((job) => {
      state.jobId = resumeId;
      if (job.status === "done") {
        showResults(job);
      } else if (job.status === "queued" || job.status === "running") {
        progressSection.classList.remove("hidden");
        cancelBtn.classList.remove("hidden");
        pollJob();
      } else if (job.status === "error") {
        progressSection.classList.remove("hidden");
        updateProgress(job);
        showError(job.error || "Reconstruction failed");
      }
    })
    .catch(() => history.replaceState(null, "", location.pathname));
}
