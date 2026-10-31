const $ = (id) => document.getElementById(id);
const statusEl = $("status");

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("err", isError);
}

// Random per-browser ID: the server only shows videos uploaded from this browser
function getSessionId() {
  const make = () => (crypto.randomUUID ? crypto.randomUUID() : Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join(""));
  try {
    let id = localStorage.getItem("visiontrack-session");
    if (!id) {
      id = make();
      localStorage.setItem("visiontrack-session", id);
    }
    return id;
  } catch {
    return make(); // storage blocked: session lasts until the page is closed
  }
}
const SESSION_ID = getSessionId();

function withSession(options) {
  return { ...options, headers: { ...options.headers, "X-Session-Id": SESSION_ID } };
}

async function api(path, options = {}) {
  const res = await fetch(path, withSession(options));
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Shared with live.js
window.VT = { $, setStatus, escapeHtml, renderCounts, conf: () => Number($("conf").value), limits: null };

function renderCounts(el, counts, empty = "Nothing detected") {
  const entries = Object.entries(counts);
  el.innerHTML = entries.length
    ? entries.map(([name, n]) => `<li>${escapeHtml(name)} <b>${n}</b></li>`).join("")
    : `<li class="none">${empty}</li>`;
}

// ---------- Tabs ----------

let currentTab = "image";

function showTab(tab) {
  currentTab = tab;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== tab));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("hidden", v.dataset.view !== tab));
  setStatus("");
  window.dispatchEvent(new CustomEvent("vt-tab", { detail: tab }));
  try { localStorage.setItem("visiontrack-tab", tab); } catch {}
}

$("tabs").addEventListener("click", (e) => {
  const tab = e.target.closest("button")?.dataset.tab;
  if (tab) showTab(tab);
});

// ---------- Confidence ----------

const confInput = $("conf");
confInput.addEventListener("input", () => { $("confValue").textContent = Number(confInput.value).toFixed(2); });
// Re-run the last image at the new threshold (cheap: one request, no new upload from disk)
confInput.addEventListener("change", () => { if (currentTab === "image" && lastImage) detectImage(lastImage); });

async function showLimits() {
  const l = await api("/api/limits");
  window.VT.limits = l;
  confInput.value = l.default_conf;
  $("confValue").textContent = Number(l.default_conf).toFixed(2);
  $("videoHint").textContent = `or click to browse · up to ${l.max_video_seconds} s, ${l.max_video_mb} MB`;
  $("limits").textContent =
    `Images up to ${l.max_image_mb} MB · videos up to ${l.max_video_seconds} s and ${l.max_video_mb} MB, ` +
    `sampled at ${l.video_fps} fps · uploads are deleted after processing, results after ${l.job_ttl_minutes} min`;
}

// ---------- Image ----------

let lastImage = null;

async function detectImage(file) {
  lastImage = file;
  setStatus(`Detecting objects in ${file.name}…`);
  const form = new FormData();
  form.append("file", file);
  try {
    const r = await api(`/api/detect?conf=${confInput.value}`, { method: "POST", body: form });
    $("imageOut").src = r.image;
    renderCounts($("imageCounts"), r.counts);
    $("imageDets").innerHTML = r.detections.length
      ? r.detections.map((d) => `<li><span>${escapeHtml(d.class)}</span><span class="conf">${d.confidence.toFixed(2)}</span></li>`).join("")
      : `<li class="none">No objects above ${Number(confInput.value).toFixed(2)} confidence</li>`;
    $("imageMeta").textContent = `${r.detections.length} detections · ${r.width}×${r.height} · ${r.model} · ${r.inference_ms} ms inference`;
    $("imageWelcome").classList.add("hidden");
    $("imageResult").classList.remove("hidden");
    setStatus("");
  } catch (e) {
    setStatus(e.message, true);
  }
}

$("imageFile").addEventListener("change", (e) => {
  if (e.target.files[0]) detectImage(e.target.files[0]);
  e.target.value = "";
});

async function sampleFile(url, name, type) {
  const res = await fetch(url);
  return new File([await res.blob()], name, { type });
}

$("sampleImage").addEventListener("click", async () => detectImage(await sampleFile("/samples/street.jpg", "street.jpg", "image/jpeg")));

// ---------- Video jobs ----------

const ACTIVE = new Set(["queued", "reading", "detecting"]);
const STEPS = ["queued", "reading", "detecting", "done"];
let currentJob = null;
let pollTimer = null;
let videoUrl = null;

function expiresIn(epochSeconds) {
  const mins = Math.max(0, Math.round((epochSeconds * 1000 - Date.now()) / 60000));
  return `expires in ${mins}m`;
}

function jobMeta(j) {
  if (j.status === "error") return "failed";
  if (ACTIVE.has(j.status)) {
    if (j.status === "queued") return j.position > 1 ? `queued · #${j.position}` : "queued…";
    return `${Math.round(j.progress * 100)}%`;
  }
  const total = Object.values(j.counts).reduce((a, b) => a + b, 0);
  return `${total} objects · ${j.expires_at ? expiresIn(j.expires_at) : ""}`;
}

async function refreshJobs() {
  const list = await api("/api/jobs");
  const ul = $("jobs");
  if (!list.length) {
    ul.innerHTML = '<li class="empty">No videos yet</li>';
    return list;
  }
  ul.innerHTML = list
    .map((j) => `
      <li data-id="${j.id}" data-state="${ACTIVE.has(j.status) ? "active" : j.status}" class="${j.id === currentJob ? "active" : ""}">
        <div class="info">
          <span class="name" title="${escapeHtml(j.filename)}">${escapeHtml(j.filename)}</span>
          <span class="meta ${j.status === "error" ? "err" : ""}">${jobMeta(j)}</span>
        </div>
        ${ACTIVE.has(j.status) ? "" : '<button title="Delete" data-del="1">✕</button>'}
      </li>`)
    .join("");
  return list;
}

$("jobs").addEventListener("click", async (e) => {
  const li = e.target.closest("li[data-id]");
  if (!li) return;
  if (e.target.dataset.del) {
    try {
      await api(`/api/jobs/${li.dataset.id}`, { method: "DELETE" });
      if (currentJob === li.dataset.id) resetVideoView();
      refreshJobs();
    } catch (err) {
      setStatus(err.message, true);
    }
    return;
  }
  openJob(li.dataset.id);
});

function resetVideoView() {
  currentJob = null;
  clearTimeout(pollTimer);
  $("progress").classList.add("hidden");
  $("videoResult").classList.add("hidden");
  $("videoWelcome").classList.remove("hidden");
}

function showProgress(j) {
  $("videoWelcome").classList.add("hidden");
  $("videoResult").classList.add("hidden");
  $("progress").classList.remove("hidden");
  $("progressTitle").textContent = j.filename;
  const idx = STEPS.indexOf(j.status);
  const failed = j.status === "error";
  $("steps").classList.toggle("failed", failed);
  document.querySelectorAll("#steps li").forEach((li, i) => {
    li.classList.toggle("done", !failed && i < idx);
    li.classList.toggle("current", !failed && i === idx);
  });
  if (failed) {
    // Mark the step that was running when it failed (best guess: the furthest one reached)
    const reached = j.frames || j.progress > 0 ? 2 : j.duration ? 1 : 0;
    document.querySelectorAll("#steps li")[reached].classList.add("current");
  }
  $("bar").style.width = `${Math.round((j.progress || 0) * 100)}%`;
  const msg = $("progressMsg");
  msg.textContent = failed ? j.error : j.status === "queued" && j.position > 1 ? `Waiting for ${j.position - 1} video(s) ahead` : j.message;
  msg.classList.toggle("err", failed);
}

async function showVideo(j) {
  $("progress").classList.add("hidden");
  $("videoWelcome").classList.add("hidden");
  // <video> can't send the session header, so fetch the file and play it from memory
  const res = await fetch(`/api/jobs/${j.id}/video`, withSession({}));
  if (!res.ok) throw new Error("The video has expired");
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = URL.createObjectURL(await res.blob());
  const player = $("videoOut");
  player.src = videoUrl;
  player.play().catch(() => {});
  $("videoDownload").href = videoUrl;
  $("videoDownload").download = j.filename.replace(/\.[^.]+$/, "") + "-tracked.mp4";
  renderCounts($("videoCounts"), j.counts, "No objects tracked");
  const speed = j.seconds ? ` in ${j.seconds} s (${(j.frames / j.seconds).toFixed(1)} frames/s on the server)` : "";
  $("videoMeta").textContent =
    `${j.frames} frames at ${window.VT.limits?.video_fps ?? 5} fps from ${j.duration?.toFixed(1)} s of video, processed${speed}. ` +
    "Counts are unique track IDs: an object counts once while it stays tracked.";
  $("videoResult").classList.remove("hidden");
}

async function openJob(id) {
  clearTimeout(pollTimer);
  currentJob = id;
  document.querySelectorAll("#jobs li").forEach((li) => li.classList.toggle("active", li.dataset.id === id));
  try {
    const j = await api(`/api/jobs/${id}`);
    if (currentJob !== id) return;
    if (j.status === "done") {
      await showVideo(j);
      refreshJobs();
    } else {
      showProgress(j);
      if (ACTIVE.has(j.status)) pollTimer = setTimeout(() => openJob(id), 1000);
      else refreshJobs();
    }
  } catch (e) {
    setStatus(e.message, true);
    refreshJobs();
  }
}

async function uploadVideo(file) {
  const l = window.VT.limits;
  if (l && file.size > l.max_video_mb * 1024 * 1024) {
    setStatus(`That file is over ${l.max_video_mb} MB`, true);
    return;
  }
  setStatus(`Uploading ${file.name}…`);
  const form = new FormData();
  form.append("file", file);
  try {
    const job = await api("/api/jobs", { method: "POST", body: form });
    setStatus("");
    await refreshJobs();
    openJob(job.id);
  } catch (e) {
    setStatus(e.message, true);
  }
}

$("videoFile").addEventListener("change", (e) => {
  if (e.target.files[0]) uploadVideo(e.target.files[0]);
  e.target.value = "";
});

$("sampleVideo").addEventListener("click", async () => uploadVideo(await sampleFile("/samples/crosswalk.mp4", "crosswalk.mp4", "video/mp4")));

// ---------- Drag and drop (routes to the open tab) ----------

let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; document.body.classList.add("dragging"); });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove("dragging"); } });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove("dragging");
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (file.type.startsWith("video/") || /\.(mp4|mov|webm|mkv|avi|m4v)$/i.test(file.name)) {
    showTab("video");
    uploadVideo(file);
  } else {
    showTab("image");
    detectImage(file);
  }
});

// ---------- Start ----------

(async () => {
  try {
    const saved = localStorage.getItem("visiontrack-tab");
    if (saved && saved !== "image") showTab(saved);
  } catch {}
  try {
    await showLimits();
    const list = await refreshJobs();
    const running = list.find((j) => ACTIVE.has(j.status));
    if (running) openJob(running.id);
  } catch (e) {
    setStatus(e.message, true);
  }
})();
