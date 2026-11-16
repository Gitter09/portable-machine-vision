const $ = (id) => document.getElementById(id);
let referenceImage = null;
let area = null;
let dragStart = null;
let currentJob = null;
let currentRelease = null;
let activeRelease = "";
let lightVerified = false;
let labelProject = null;
let labelIndex = 0;
let labelImage = null;
let labelBoxes = [];
let labelDrag = null;
let labelDirty = false;
let hostMode = "trainer";
let repeatInspection = false;
let capturePending = false;

function showInspection(releaseId) {
  if (hostMode !== "runtime" || !releaseId) return;
  $("inspection").hidden = false;
  $("inspection-link").hidden = false;
  $("inspection-model").textContent = `Model: ${releaseId}`;
  $("model").hidden = true;
  location.hash = "inspection";
}

function stopInspection() {
  repeatInspection = false;
  $("capture-stop").hidden = true;
  $("capture-repeat").disabled = capturePending;
}

async function captureInspection() {
  if (capturePending) return;
  capturePending = true;
  $("capture-once").disabled = true;
  $("capture-repeat").disabled = true;
  $("inspection-status").textContent = "Capturing and inspecting…";
  try {
    const result = await api("/api/capture", { method: "POST" });
    const image = new Image();
    image.src = result.preview;
    await image.decode();
    const canvas = $("inspection-canvas");
    canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);
    const scaleX = canvas.width / result.source_width;
    const scaleY = canvas.height / result.source_height;
    const detections = result.detections || [];
    $("inspection-detections").replaceChildren();
    context.lineWidth = 3;
    context.font = "22px Segoe UI";
    for (const detection of detections) {
      const box = detection.box_source;
      const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(1)}%`;
      if (box) {
        context.strokeStyle = "#cf9b4a";
        context.strokeRect(box.x * scaleX, box.y * scaleY, box.width * scaleX, box.height * scaleY);
        const labelY = Math.max(26, box.y * scaleY);
        context.fillStyle = "#000";
        context.fillRect(box.x * scaleX, labelY - 26, context.measureText(label).width + 12, 30);
        context.fillStyle = "#efeae2";
        context.fillText(label, box.x * scaleX + 6, labelY - 3);
      }
      const item = document.createElement("li"); item.textContent = label;
      $("inspection-detections").append(item);
    }
    canvas.hidden = false; $("inspection-empty").hidden = true;
    $("inspection-summary").textContent = `${detections.length} detection${detections.length === 1 ? "" : "s"}`;
    $("inspection-status").textContent = `Captured ${new Date().toLocaleTimeString()}${repeatInspection ? " · Testing continues" : " · Ready for the next sample"}`;
  } catch (error) {
    stopInspection();
    $("inspection-status").textContent = `Capture failed: ${error.message}`;
    $("inspection-summary").textContent = "The displayed image, if any, is from the previous successful capture.";
  } finally {
    capturePending = false;
    $("capture-once").disabled = repeatInspection;
    $("capture-repeat").disabled = repeatInspection;
    if (repeatInspection) setTimeout(() => { if (repeatInspection) captureInspection(); }, 1000);
  }
}

$("capture-once").addEventListener("click", captureInspection);
$("capture-repeat").addEventListener("click", () => {
  repeatInspection = true; $("capture-stop").hidden = false; captureInspection();
});
$("capture-stop").addEventListener("click", () => {
  stopInspection(); $("capture-once").disabled = capturePending;
  $("inspection-status").textContent = capturePending ? "Finishing current capture…" : "Testing stopped. Ready for a sample.";
});

function message(text, kind = "success") {
  const node = $("message"); node.textContent = text; node.className = `message ${kind}`; node.hidden = false;
  node.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "The action could not be completed");
  return body;
}
function drawReference() {
  if (!referenceImage) return;
  const canvas = $("reference-canvas"), context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(referenceImage, 0, 0, canvas.width, canvas.height);
  if (area) {
    context.fillStyle = "rgba(207,155,74,.19)"; context.strokeStyle = "#cf9b4a"; context.lineWidth = 8;
    context.fillRect(...area); context.strokeRect(...area);
    $("area-details").textContent = `${Math.round(area[2])} × ${Math.round(area[3])} pixel area`;
    $("save-area").disabled = false;
  }
}
function setStep(id, done) {
  const link = document.querySelector(`.step-link[href="#${id}"]`);
  if (link) link.classList.toggle("complete", Boolean(done));
}
function renderConnections(health) {
  const statuses = [
    ["SiMa SDK worker", health.board, "Not running yet. Import a model, then activate it or choose Start SDK inspection."],
    ["Camera", health.camera, "Checked when the SDK worker starts. Check the camera address if activation fails."],
    ["Light", lightVerified, health.light_port ? "Port found. Choose ‘Test light briefly’ to confirm it responds." : "Connect the light controller to the selected COM port."],
  ];
  $("connection-status").replaceChildren();
  for (const [name, good, help] of statuses) {
    const row = document.createElement("div"); row.className = good ? "connection good" : "connection pending";
    const label = document.createElement("strong"); label.textContent = name;
    const value = document.createElement("span"); value.textContent = good ? "Ready" : help;
    row.append(label, value); $("connection-status").append(row);
  }
  setStep("connect", statuses.every((item) => item[1]));
}
function loadRelease(release) {
  currentRelease = release.version;
  if (hostMode === "runtime" && activeRelease === release.version) showInspection(activeRelease);
  $("model-empty").hidden = true; $("model-details").hidden = false;
  $("model-version").textContent = release.version;
  $("model-score").textContent = Number.isFinite(release.map50) ? release.map50.toFixed(3) : "—";
  const states = { trained_not_compiled: "SiMa build required", compiled_not_board_verified: hostMode === "trainer" ? "Ready to export" : "Ready to send to board", staged_not_verified: "Ready for board check", board_verified: "Verified on board" };
  $("model-board-status").textContent = states[release.state] || release.state;
  $("export-model").hidden = hostMode !== "trainer" || release.state !== "compiled_not_board_verified";
  if (!$("export-model").hidden) $("export-model").href = `/api/releases/${release.version}/export`;
  $("send-model").hidden = hostMode !== "runtime" || !["compiled_not_board_verified", "staged_not_verified"].includes(release.state);
  $("send-model").disabled = release.state === "staged_not_verified";
  $("activate-model").disabled = !["staged_not_verified", "board_verified"].includes(release.state) || activeRelease === release.version;
  $("rollback-model").hidden = hostMode !== "runtime" || !activeRelease || !document.body.dataset.previousRelease;
  const gallery = $("model-previews"); gallery.replaceChildren();
  for (const name of release.review_images || []) {
    const image = document.createElement("img"); image.src = `/api/releases/${release.version}/review/${encodeURIComponent(name)}`;
    image.alt = "Prediction on a held-out image"; gallery.append(image);
  }
  setStep("model", activeRelease === release.version);
}
function pointOnCanvas(event, canvas = $("reference-canvas")) {
  const rect = canvas.getBoundingClientRect();
  return [Math.max(0, Math.min(3200, Math.round((event.clientX - rect.left) * 3200 / rect.width))), Math.max(0, Math.min(2256, Math.round((event.clientY - rect.top) * 2256 / rect.height)))];
}
function drawLabeledImage() {
  if (!labelImage) return;
  const canvas = $("label-canvas"), context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(labelImage, 0, 0, canvas.width, canvas.height);
  const boxes = labelDrag ? [...labelBoxes, labelDrag] : labelBoxes;
  for (const box of boxes) {
    context.strokeStyle = "#cf9b4a"; context.fillStyle = "rgba(207,155,74,.16)"; context.lineWidth = 8;
    context.fillRect(box.x, box.y, box.width, box.height);
    context.strokeRect(box.x, box.y, box.width, box.height);
    context.font = "bold 36px Segoe UI"; context.fillStyle = "#cf9b4a";
    context.fillText(labelProject.classes[box.class_id], box.x + 4, Math.max(40, box.y - 12));
  }
  $("undo-box").disabled = labelBoxes.length === 0;
}
function showLabelImage(index) {
  labelIndex = index;
  const record = labelProject.images[index];
  $("labeler-count").textContent = `Image ${index + 1} of ${labelProject.images.length} · ${labelProject.images.filter((image) => image.reviewed).length} reviewed`;
  labelDirty = false;
  $("previous-labeled").disabled = index === 0;
  $("next-labeled").disabled = index === labelProject.images.length - 1;
  labelBoxes = record.boxes.map((box) => ({ ...box })); labelDrag = null;
  $("empty-labeled-image").checked = record.reviewed && labelBoxes.length === 0;
  $("empty-labeled-image").disabled = labelBoxes.length > 0;
  $("labeler-feedback").textContent = "Draw boxes for every object. Save changes before switching images.";
  labelImage = null;
  $("save-labeled-image").disabled = true;
  const source = new Image();
  source.onload = () => { if (labelIndex !== index) return; labelImage = source; $("save-labeled-image").disabled = false; drawLabeledImage(); };
  source.onerror = () => message("This saved camera image could not be opened.", "error");
  source.src = `/api/annotations/${labelProject.project_id}/images/${record.id}`;
  const complete = labelProject.images.length >= 2 && labelProject.images.every((image) => image.reviewed);
  $("download-labeled").hidden = !complete;
  $("train-labeled").hidden = !complete;
  if (complete) $("download-labeled").href = `/api/annotations/${labelProject.project_id}/export`;
}
function markLabelDirty() {
  labelDirty = true;
  $("previous-labeled").disabled = true;
  $("next-labeled").disabled = true;
  $("labeler-feedback").textContent = "Unsaved changes: save this image before switching.";
}
function openLabelProject(project) {
  labelProject = project;
  $("labeler-panel").open = true;
  $("labeler-workspace").hidden = false;
  $("labeler-start").hidden = true;
  $("label-class").replaceChildren();
  project.classes.forEach((name, index) => {
    const option = document.createElement("option"); option.value = String(index); option.textContent = name;
    $("label-class").append(option);
  });
  if (project.images.length) {
    const next = project.images.findIndex((image) => !image.reviewed);
    showLabelImage(next < 0 ? project.images.length - 1 : next);
  }
}
async function loadReference() {
  const source = new Image();
  source.onload = () => { referenceImage = source; $("reference-workspace").hidden = false; drawReference(); };
  source.src = `/api/reference?t=${Date.now()}`;
}
function renderJob(job) {
  $("job-panel").hidden = false;
  const labels = { queued: "Waiting to start", checking_images: "Checking image package", preparing_images: "Preparing images", training: "Training on this PC", validating: "Checking model quality", exporting: "Preparing board model", compiling: "Building for SiMa", needs_board_compile: "Training complete", needs_board_check: "Board build complete", failed: "Training stopped", interrupted: "Training was interrupted" };
  $("job-title").textContent = labels[job.phase] || job.phase;
  $("job-phase").textContent = job.phase === "failed" ? "Needs attention" : "In progress";
  $("job-error").hidden = !(job.error || job.note);
  $("job-error").textContent = job.error || job.note || "";
  $("retry-training").hidden = !["failed", "interrupted"].includes(job.phase);
  $("compile-training").hidden = job.phase !== "needs_board_compile";
  if (["needs_board_compile", "needs_board_check"].includes(job.phase)) {
    $("job-phase").textContent = "Candidate ready";
    $("job-description").textContent = "The model trained successfully. It still needs a verified SiMa build before activation.";
    api(`/api/releases/${job.release}`).then(loadRelease).catch(() => {});
    setStep("images", true);
  }
}
async function pollJob() {
  if (!currentJob) return;
  try {
    const job = await api(`/api/jobs/${currentJob}`); renderJob(job);
    if (!new Set(["failed", "interrupted", "needs_board_compile", "needs_board_check"]).has(job.phase)) setTimeout(pollJob, 2500);
  } catch (error) { message(error.message, "error"); }
}
window.addEventListener("DOMContentLoaded", async () => {
  try {
    const state = await api("/api/state"), config = state.config;
    hostMode = state.mode;
    document.body.dataset.mode = hostMode;
    $("local-note").textContent = hostMode === "trainer" ? "NVIDIA workstation · trainer" : "Admin PC · runtime";
    for (const id of hostMode === "trainer" ? ["connect"] : ["area", "images"]) {
      document.querySelector(`.step-link[href="#${id}"]`)?.parentElement.setAttribute("hidden", "");
    }
    document.querySelectorAll(".steps li:not([hidden]) .step-link span").forEach((number, index) => { number.textContent = String(index + 1); });
    if (hostMode === "trainer") {
      $("model-heading").textContent = "Review and export your model";
      $("model-intro").textContent = "When the SiMa build succeeds, download one model package and import it on the Admin PC.";
      $("model-empty").textContent = "No model candidate yet. Add training images to create one.";
      $("model-note").textContent = "Once the SiMa build is ready, download this package and import it on the Admin PC. Training does not change the running machine.";
    } else {
      $("model-heading").textContent = "Import and activate a model";
      $("model-intro").textContent = "Bring the compiled model package from the NVIDIA workstation. The current machine model stays unchanged until you activate the new one.";
      $("model-empty").textContent = "No model package imported yet. Choose the ZIP exported on the NVIDIA workstation.";
      $("connect").hidden = Boolean(config.runtime_setup_complete);
      document.querySelector('.step-link[href="#connect"]')?.parentElement.toggleAttribute("hidden", Boolean(config.runtime_setup_complete));
      $("model-import").hidden = Boolean(config.active_release || state.releases?.length);
    }
    $("board-host").value = config.board_host; $("camera-ip").value = config.camera_ip;
    $("light-port").value = config.light_port; $("base-weights").value = config.base_weights;
    $("compiler-image").value = config.compiler_image;
    activeRelease = config.active_release; document.body.dataset.previousRelease = config.previous_release;
    $("start-sdk").hidden = !activeRelease || state.health.board;
    if (hostMode === "runtime") renderConnections(state.health);
    if (hostMode === "runtime" && activeRelease) showInspection(activeRelease);
    if (hostMode === "trainer" && config.reference_image) { area = config.roi; loadReference(); }
    if (hostMode === "trainer") setStep("area", Boolean(config.reference_image && config.roi));
    if (state.jobs?.length) { currentJob = state.jobs[0].job_id; pollJob(); }
    else if (state.releases?.length) { loadRelease(await api(`/api/releases/${state.releases.at(-1)}`)); }
    const savedProject = hostMode === "trainer" && localStorage.getItem("demeter-label-project");
    if (savedProject) {
      try { openLabelProject(await api(`/api/annotations/${savedProject}`)); }
      catch (_) { localStorage.removeItem("demeter-label-project"); }
    }
  } catch (error) { message(error.message, "error"); }
});
$("save-connections").addEventListener("click", async () => {
  try {
    await api("/api/config", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ board_host: $("board-host").value.trim(), camera_ip: $("camera-ip").value.trim(), light_port: $("light-port").value.trim(), runtime_setup_complete: true }) });
    const state = await api("/api/state");
    renderConnections(state.health);
    $("connect").hidden = true;
    document.querySelector('.step-link[href="#connect"]')?.parentElement.setAttribute("hidden", "");
    message("Connection settings saved.");
  } catch (error) { message(error.message, "error"); }
});
$("save-tools").addEventListener("click", async () => {
  try {
    await api("/api/config", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ base_weights: $("base-weights").value.trim(), compiler_image: $("compiler-image").value.trim() }) });
    message("Training locations saved.");
  } catch (error) { message(error.message, "error"); }
});
$("reference-file").addEventListener("change", async (event) => {
  const file = event.target.files?.[0]; if (!file) return;
  try { await api("/api/reference", { method: "POST", body: file }); area = null; $("save-area").disabled = true; setStep("area", false); await loadReference(); message("Reference image loaded. Draw the area to inspect."); }
  catch (error) { message(error.message, "error"); }
});
$("reference-canvas").addEventListener("pointerdown", (event) => { dragStart = pointOnCanvas(event); $("reference-canvas").setPointerCapture(event.pointerId); });
$("reference-canvas").addEventListener("pointermove", (event) => {
  if (!dragStart) return;
  const end = pointOnCanvas(event);
  area = [Math.max(0, Math.min(dragStart[0], end[0])), Math.max(0, Math.min(dragStart[1], end[1])), Math.min(3200, Math.abs(end[0] - dragStart[0])), Math.min(2256, Math.abs(end[1] - dragStart[1]))];
  drawReference();
});
$("reference-canvas").addEventListener("pointerup", () => { dragStart = null; if (area && (area[2] < 20 || area[3] < 20)) { area = null; drawReference(); } });
$("save-area").addEventListener("click", async () => {
  try { await api("/api/config", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ roi: area }) }); setStep("area", true); message("Inspection area saved."); }
  catch (error) { message(error.message, "error"); }
});
$("dataset-file").addEventListener("change", (event) => { const file = event.target.files?.[0]; $("start-training").disabled = !file; $("upload-name").textContent = file?.name || ""; });
$("start-training").addEventListener("click", async () => {
  const file = $("dataset-file").files?.[0]; if (!file) return;
  $("start-training").disabled = true;
  try {
    const result = await api("/api/train", { method: "POST", body: file }); currentJob = result.job_id; pollJob();
    message(`Image package accepted: ${result.images} images, ${result.classes.length} classes. Training has started.`);
  } catch (error) { $("start-training").disabled = false; message(error.message, "error"); }
});
$("send-model").addEventListener("click", async () => {
  if (!currentRelease) return;
  $("send-model").disabled = true;
  try { await api(`/api/releases/${currentRelease}/stage`, { method: "POST" }); loadRelease(await api(`/api/releases/${currentRelease}`)); message("Candidate copied to the SiMa board and checked. Place a sample item under the camera before activation."); }
  catch (error) { $("send-model").disabled = false; message(error.message, "error"); }
});
$("activate-model").addEventListener("click", async () => {
  if (!currentRelease) return;
  $("activate-model").disabled = true;
  try { message("Starting the model and camera…"); const result = await api(`/api/releases/${currentRelease}/activate`, { method: "POST" }); activeRelease = result.active_release; document.body.dataset.previousRelease = result.previous_release; $("start-sdk").hidden = true; $("model-import").hidden = true; loadRelease(await api(`/api/releases/${currentRelease}`)); message("Inspection is ready. Place a sample and choose Capture and inspect."); }
  catch (error) { $("activate-model").disabled = false; message(error.message, "error"); }
});
$("rollback-model").addEventListener("click", async () => {
  try { const result = await api("/api/rollback", { method: "POST" }); activeRelease = result.active_release; document.body.dataset.previousRelease = result.previous_release; loadRelease(await api(`/api/releases/${activeRelease}`)); message("Previous model restored and checked on the board."); }
  catch (error) { message(error.message, "error"); }
});
const links = [...document.querySelectorAll(".step-link")];
const sectionObserver = new IntersectionObserver((entries) => {
  const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
  if (visible.length) for (const link of links) link.classList.toggle("active", link.hash === `#${visible[0].target.id}`);
}, { rootMargin: "-15% 0px -60% 0px" });
for (const section of document.querySelectorAll("main section")) sectionObserver.observe(section);
$("retry-training").addEventListener("click", async () => {
  if (!currentJob) return;
  try { const result = await api(`/api/jobs/${currentJob}/retry`, { method: "POST" }); currentJob = result.job_id; pollJob(); message("Training restarted with the same image package."); }
  catch (error) { message(error.message, "error"); }
});
$("compile-training").addEventListener("click", async () => {
  if (!currentJob) return;
  try {
    $("compile-training").disabled = true;
    await api(`/api/jobs/${currentJob}/compile`, { method: "POST" });
    pollJob();
    message("SiMa build restarted from the trained model; no retraining.");
  } catch (error) { message(error.message, "error"); }
  finally { $("compile-training").disabled = false; }
});
$("check-light").addEventListener("click", async () => {
  $("check-light").disabled = true;
  try { await api("/api/light/check", { method: "POST" }); lightVerified = true; renderConnections((await api("/api/state")).health); message("Light controller responded and was switched off again."); }
  catch (error) { lightVerified = false; message(error.message, "error"); }
  finally { $("check-light").disabled = false; }
});
$("create-label-set").addEventListener("click", async () => {
  const files = [...$("raw-images").files];
  const classes = $("class-names").value.split(",").map((name) => name.trim()).filter(Boolean);
  if (!classes.length || files.length < 2) return message("Enter object names and choose at least two camera images.", "error");
  $("create-label-set").disabled = true;
  try {
    const project = await api("/api/annotations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ classes }) });
    localStorage.setItem("demeter-label-project", project.project_id);
    for (let index = 0; index < files.length; index++) {
      $("create-label-set").textContent = `Adding image ${index + 1} of ${files.length}…`;
      await api(`/api/annotations/${project.project_id}/images`, { method: "POST", body: files[index] });
    }
    openLabelProject(await api(`/api/annotations/${project.project_id}`));
    message("Images added. Draw a box around every object in the first image.");
  } catch (error) {
    message(`${error.message} Any images already added are saved locally; reload this page to continue.`, "error");
  } finally { $("create-label-set").disabled = false; $("create-label-set").textContent = "Start drawing boxes"; }
});
$("label-canvas").addEventListener("pointerdown", (event) => {
  if (!labelImage) return;
  const [x, y] = pointOnCanvas(event, $("label-canvas"));
  labelDrag = { class_id: Number($("label-class").value), x, y, width: 0, height: 0, startX: x, startY: y };
  $("label-canvas").setPointerCapture(event.pointerId);
});
$("label-canvas").addEventListener("pointermove", (event) => {
  if (!labelDrag) return;
  const [x, y] = pointOnCanvas(event, $("label-canvas"));
  labelDrag.x = Math.min(labelDrag.startX, x); labelDrag.y = Math.min(labelDrag.startY, y);
  labelDrag.width = Math.abs(x - labelDrag.startX); labelDrag.height = Math.abs(y - labelDrag.startY);
  drawLabeledImage();
});
$("label-canvas").addEventListener("pointerup", () => {
  if (!labelDrag) return;
  if (labelDrag.width >= 2 && labelDrag.height >= 2) {
    const { class_id, x, y, width, height } = labelDrag;
    labelBoxes.push({ class_id, x, y, width, height });
    markLabelDirty();
    $("empty-labeled-image").checked = false;
    $("empty-labeled-image").disabled = true;
  }
  labelDrag = null; drawLabeledImage();
});
$("undo-box").addEventListener("click", () => { labelBoxes.pop(); markLabelDirty(); $("empty-labeled-image").disabled = labelBoxes.length > 0; drawLabeledImage(); });
$("previous-labeled").addEventListener("click", () => showLabelImage(labelIndex - 1));
$("next-labeled").addEventListener("click", () => showLabelImage(labelIndex + 1));
$("new-label-set").addEventListener("click", () => {
  if (labelDirty) return $("labeler-feedback").textContent = "Save this image before starting a new set.";
  localStorage.removeItem("demeter-label-project"); labelProject = null; labelImage = null;
  $("labeler-workspace").hidden = true; $("labeler-start").hidden = false;
  $("raw-images").value = ""; $("class-names").value = "";
  message("Ready for a new image set. The previous set remains saved on this PC.");
});
$("more-images").addEventListener("change", async (event) => {
  const files = [...event.target.files]; if (!files.length || !labelProject) return;
  if (labelDirty) { event.target.value = ""; $("labeler-feedback").textContent = "Save this image before adding more."; return; }
  try {
    for (let index = 0; index < files.length; index++) {
      await api(`/api/annotations/${labelProject.project_id}/images`, { method: "POST", body: files[index] });
    }
    labelProject = await api(`/api/annotations/${labelProject.project_id}`);
    showLabelImage(labelProject.images.length - files.length);
    message(`${files.length} new image${files.length === 1 ? "" : "s"} added. Review each new image before training.`);
  } catch (error) {
    labelProject = await api(`/api/annotations/${labelProject.project_id}`).catch(() => labelProject);
    showLabelImage(Math.max(0, labelProject.images.findIndex((image) => !image.reviewed)));
    message(`${error.message} Earlier images are still saved.`, "error");
  } finally { event.target.value = ""; }
});
$("empty-labeled-image").addEventListener("change", markLabelDirty);
$("model-package-file").addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  $("model-package-name").textContent = file?.name || "";
  $("import-model").disabled = !file;
});
$("import-model").addEventListener("click", async () => {
  const file = $("model-package-file").files?.[0]; if (!file) return;
  $("import-model").disabled = true;
  try {
    const result = await api("/api/releases/import", { method: "POST", body: file });
    $("model-import").hidden = true;
    loadRelease(await api(`/api/releases/${result.release}`));
    message("Model package checked and imported. Send it to the SiMa board when the existing machine app is stopped.");
  } catch (error) { $("import-model").disabled = false; message(error.message, "error"); }
});
$("start-sdk").addEventListener("click", async () => {
  $("start-sdk").disabled = true;
  try {
    await api("/api/session/start", { method: "POST" });
    $("start-sdk").hidden = true;
    renderConnections((await api("/api/state")).health);
    showInspection(activeRelease);
    message("SDK inspection is running. Use Quit and release camera before opening the existing application.");
  } catch (error) { $("start-sdk").disabled = false; message(error.message, "error"); }
});
$("quit-sdk").addEventListener("click", async () => {
  stopInspection();
  $("quit-sdk").disabled = true;
  try {
    const result = await api("/api/session/quit", { method: "POST" });
    document.querySelector("main").hidden = true;
    const note = document.createElement("p");
    note.className = "exit-note";
    note.textContent = result.message;
    document.body.append(note);
  } catch (error) {
    $("quit-sdk").disabled = false;
    message(`Could not confirm that the camera was released: ${error.message}`, "error");
  }
});
$("save-labeled-image").addEventListener("click", async () => {
  if (!labelProject) return;
  if (!labelBoxes.length && !$("empty-labeled-image").checked) {
    $("labeler-feedback").textContent = "Draw a box, or explicitly check that this image has no objects.";
    $("empty-labeled-image").focus();
    return;
  }
  $("save-labeled-image").disabled = true;
  try {
    const record = labelProject.images[labelIndex];
    await api(`/api/annotations/${labelProject.project_id}/images/${record.id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ boxes: labelBoxes }) });
    labelProject = await api(`/api/annotations/${labelProject.project_id}`);
    const next = labelProject.images.findIndex((image) => !image.reviewed);
    showLabelImage(next < 0 ? labelIndex : next);
    $("labeler-feedback").textContent = next < 0 ? "All images reviewed. Download the ZIP or start training." : `Saved. Continue with image ${next + 1}.`;
  } catch (error) { $("labeler-feedback").textContent = error.message; }
  finally { $("save-labeled-image").disabled = false; }
});
$("train-labeled").addEventListener("click", async () => {
  if (!labelProject) return;
  $("train-labeled").disabled = true;
  try {
    const result = await api(`/api/annotations/${labelProject.project_id}/train`, { method: "POST" });
    currentJob = result.job_id; pollJob(); message("Training started from your reviewed images.");
  } catch (error) { $("train-labeled").disabled = false; message(error.message, "error"); }
});
