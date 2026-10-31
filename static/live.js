// Live webcam detection, entirely in the browser: the same YOLO26n ONNX model the server uses,
// run by ONNX Runtime Web (WebGPU when available, otherwise WebAssembly). No frames are uploaded.
// Pre/post-processing mirrors app/detector.py; the tracker mirrors app/tracker.py.

const ORT_VERSION = "1.30.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const PAD = 114; // Ultralytics' letterbox grey
const { $, setStatus, renderCounts } = window.VT;

let ort = null;
let session = null;
let classes = [];
let backend = "";
let size = 320;
let stream = null;
let running = false;
let tracker = null;
let mirror = true; // webcam is shown as a selfie view; the sample clip is not

// ---------- Model ----------

async function loadModel() {
  if (session) return;
  setStatus("Loading ONNX Runtime Web…");
  ort = await import(`${ORT_BASE}ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.logLevel = "error"; // hide the expected "some nodes run on CPU" notice under WebGPU

  const limits = window.VT.limits || (await (await fetch("/api/limits")).json());
  setStatus("Downloading the model (~10 MB, cached afterwards)…");
  const [model, names] = await Promise.all([
    fetch(limits.browser_model).then((r) => r.arrayBuffer()),
    fetch(limits.classes).then((r) => r.json()),
  ]);
  classes = names;
  size = Number(limits.browser_model.match(/-(\d+)\.onnx$/)?.[1] || 320);

  // WebGPU is much faster where it exists; fall back to WASM if it is missing or fails to start
  const providers = "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"];
  for (const ep of providers) {
    try {
      setStatus(`Starting the model on ${ep === "webgpu" ? "WebGPU" : "WebAssembly"}…`);
      session = await ort.InferenceSession.create(model, { executionProviders: [ep], graphOptimizationLevel: "all", logSeverityLevel: 3 });
      backend = ep;
      break;
    } catch (e) {
      console.warn(`${ep} failed, trying the next backend`, e);
    }
  }
  if (!session) throw new Error("Could not start ONNX Runtime in this browser");
}

// ---------- Pre/post-processing (matches app/detector.py) ----------

const prep = document.createElement("canvas");
const prepCtx = prep.getContext("2d", { willReadFrequently: true });

function preprocess(video) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const r = Math.min(size / vh, size / vw);
  const nw = Math.round(vw * r), nh = Math.round(vh * r);
  const left = Math.round((size - nw) / 2 - 0.1), top = Math.round((size - nh) / 2 - 0.1);
  prep.width = prep.height = size;
  prepCtx.fillStyle = `rgb(${PAD},${PAD},${PAD})`;
  prepCtx.fillRect(0, 0, size, size);
  prepCtx.drawImage(video, left, top, nw, nh);
  const { data } = prepCtx.getImageData(0, 0, size, size); // RGBA, row-major
  const plane = size * size;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    input[i] = data[i * 4] / 255;
    input[i + plane] = data[i * 4 + 1] / 255;
    input[i + 2 * plane] = data[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", input, [1, 3, size, size]), r, left, top, vw, vh };
}

function postprocess(out, { r, left, top, vw, vh }, conf) {
  // End-to-end head: (1, N, 6) rows of [x1, y1, x2, y2, score, class] in input pixels
  const [, n, k] = out.dims;
  if (k !== 6) throw new Error(`Expected an end-to-end (1, N, 6) output, got ${out.dims}`);
  const d = out.data;
  const dets = [];
  for (let i = 0; i < n; i++) {
    const score = d[i * 6 + 4];
    if (score < conf) continue;
    const clip = (v, max) => Math.min(Math.max(v, 0), max);
    dets.push({
      box: [
        clip((d[i * 6] - left) / r, vw), clip((d[i * 6 + 1] - top) / r, vh),
        clip((d[i * 6 + 2] - left) / r, vw), clip((d[i * 6 + 3] - top) / r, vh),
      ],
      score,
      cls: Math.round(d[i * 6 + 5]),
    });
  }
  return dets;
}

// ---------- Tracker (port of app/tracker.py) ----------

function iou(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
}

class Tracker {
  constructor(iouThreshold = 0.2, maxAge = 15, minHits = 3) {
    Object.assign(this, { iouThreshold, maxAge, minHits, tracks: [], nextId: 1, counts: {} });
  }

  update(dets) {
    const predict = (t) => t.box.map((v, i) => v + t.vel[i] * (t.misses + 1));
    const pairs = [];
    this.tracks.forEach((t, ti) => {
      const p = predict(t);
      dets.forEach((d, di) => {
        if (d.cls !== t.cls) return;
        const v = iou(p, d.box);
        if (v >= this.iouThreshold) pairs.push([v, ti, di]);
      });
    });
    pairs.sort((a, b) => b[0] - a[0]);
    const usedT = new Set(), usedD = new Set();
    for (const [, ti, di] of pairs) {
      if (usedT.has(ti) || usedD.has(di)) continue;
      usedT.add(ti);
      usedD.add(di);
      const t = this.tracks[ti], d = dets[di];
      t.vel = t.vel.map((v, i) => 0.5 * ((d.box[i] - t.box[i]) / (t.misses + 1)) + 0.5 * v);
      Object.assign(t, { box: d.box, score: d.score, misses: 0 });
      t.hits++;
    }
    this.tracks.forEach((t, ti) => { if (!usedT.has(ti)) t.misses++; });
    dets.forEach((d, di) => {
      if (!usedD.has(di)) this.tracks.push({ box: d.box, cls: d.cls, score: d.score, vel: [0, 0, 0, 0], hits: 1, misses: 0, id: null });
    });
    const visible = [];
    for (const t of this.tracks) {
      if (t.id === null && t.hits >= this.minHits) {
        t.id = this.nextId++;
        const name = classes[t.cls];
        this.counts[name] = (this.counts[name] || 0) + 1;
      }
      if (t.id !== null && t.misses === 0) visible.push(t);
    }
    this.tracks = this.tracks.filter((t) => t.misses <= this.maxAge && (t.id !== null || t.misses === 0));
    return visible;
  }
}

// ---------- Drawing ----------

const colorFor = (key) => `hsl(${Math.round(((key * 0.618034) % 1) * 360)}, 95%, 60%)`;

function drawBoxes(canvas, items) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  ctx.clearRect(0, 0, W, canvas.height);
  const lw = Math.max(2, Math.round(W / 400));
  ctx.lineWidth = lw;
  ctx.font = `600 ${Math.max(12, Math.round(W / 60))}px Inter, system-ui, sans-serif`;
  ctx.textBaseline = "top";
  for (const { box, label, key } of items) {
    // A webcam is mirrored with CSS (selfie view); mirror coordinates here so text stays readable
    const [x1, x2] = mirror ? [W - box[2], W - box[0]] : [box[0], box[2]];
    const [, y1, , y2] = box;
    const color = colorFor(key);
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const th = parseInt(ctx.font.match(/(\d+)px/)[1], 10) + 6;
    const tw = ctx.measureText(label).width + 8;
    const ty = y1 - th >= 0 ? y1 - th : y1;
    ctx.fillStyle = color;
    ctx.fillRect(x1 - lw / 2, ty, tw, th);
    ctx.fillStyle = "#000";
    ctx.fillText(label, x1 + 4 - lw / 2, ty + 3);
  }
}

// ---------- Loop ----------

async function loop() {
  const video = $("cam"), canvas = $("overlay");
  let fps = 0, last = performance.now(), lastPanel = 0;
  while (running) {
    if (video.readyState < 2) {
      await new Promise((r) => requestAnimationFrame(r));
      continue;
    }
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const t0 = performance.now();
    const input = preprocess(video);
    const outputs = await session.run({ [session.inputNames[0]]: input.tensor });
    const dets = postprocess(outputs[session.outputNames[0]], input, window.VT.conf());
    const latency = performance.now() - t0;
    if (!running) break;

    const tracking = $("liveTrack").checked;
    const items = tracking
      ? tracker.update(dets).map((t) => ({ box: t.box, label: `#${t.id} ${classes[t.cls]}`, key: t.id }))
      : dets.map((d) => ({ box: d.box, label: `${classes[d.cls]} ${d.score.toFixed(2)}`, key: d.cls }));
    drawBoxes(canvas, items);

    const now = performance.now();
    const inst = 1000 / Math.max(now - last, 1);
    fps = fps ? 0.9 * fps + 0.1 * inst : inst;
    last = now;
    if (now - lastPanel > 250) {
      lastPanel = now;
      $("fps").textContent = fps.toFixed(1);
      $("latency").textContent = Math.round(latency);
      const inView = {};
      dets.forEach((d) => { inView[classes[d.cls]] = (inView[classes[d.cls]] || 0) + 1; });
      renderCounts($("liveCounts"), Object.fromEntries(Object.entries(inView).sort((a, b) => b[1] - a[1])));
      $("liveUniqueHead").classList.toggle("hidden", !tracking);
      $("liveUnique").classList.toggle("hidden", !tracking);
      if (tracking) renderCounts($("liveUnique"), Object.fromEntries(Object.entries(tracker.counts).sort((a, b) => b[1] - a[1])), "None yet");
    }
    // Yield so the page stays responsive even when inference is very fast
    await new Promise((r) => requestAnimationFrame(r));
  }
}

// useSample: run on the bundled clip instead of a camera, for visitors without one
async function start(useSample = false) {
  const button = $("camToggle");
  button.disabled = $("camSample").disabled = true;
  try {
    const video = $("cam");
    if (useSample) {
      await loadModel();
      video.srcObject = null;
      video.src = "/samples/crosswalk.mp4";
      video.loop = true;
    } else {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access needs HTTPS (or localhost) and a modern browser");
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
      await loadModel();
      video.removeAttribute("src");
      video.srcObject = stream;
    }
    mirror = !useSample;
    $("liveMedia").classList.toggle("mirror", mirror);
    await video.play();
    tracker = new Tracker();
    $("backend").textContent = `${backend === "webgpu" ? "WebGPU" : "WebAssembly (CPU)"} · YOLO26n ${size} px · ONNX Runtime Web ${ORT_VERSION}`;
    $("liveWelcome").classList.add("hidden");
    $("liveResult").classList.remove("hidden");
    setStatus("");
    running = true;
    button.textContent = useSample ? "Stop" : "Stop camera";
    button.classList.add("stop");
    loop().catch((e) => { setStatus(`Inference failed: ${e.message}`, true); stop(); });
  } catch (e) {
    const msg = {
      NotAllowedError: "Camera permission was denied. Allow it in the address bar, or try the sample clip.",
      NotFoundError: "No camera found. Check it is connected and switched on, or try the sample clip.",
      NotReadableError: "The camera is in use by another app. Close it, or try the sample clip.",
    }[e.name] || e.message;
    setStatus(msg, true);
    stop();
  } finally {
    button.disabled = $("camSample").disabled = false;
  }
}

function stop() {
  running = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  $("cam").pause();
  const button = $("camToggle");
  button.textContent = "Start camera";
  button.classList.remove("stop");
}

$("camToggle").addEventListener("click", () => (running ? stop() : start()));
$("camSample").addEventListener("click", () => {
  if (running) stop();
  start(true);
});
// Release the camera when leaving the tab
window.addEventListener("vt-tab", (e) => { if (e.detail !== "live" && running) stop(); });
