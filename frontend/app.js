const $ = (id) => document.getElementById(id);
const canvas = $("scene"),
  ctx = canvas.getContext("2d"),
  preview = $("preview").getContext("2d");
const classes = [
  "IC missing",
  "Terminal present",
  "IC present",
  "Terminal missing",
];
let source = "mock",
  scenario = "cycle",
  paused = false,
  sequence = 0,
  started = Date.now(),
  previous = "",
  generation = 0,
  lastSequence = null,
  lastReceived = null;
const recovery = globalThis.VisionRecovery;
let liveAbort = new AbortController(),
  failures = 0,
  freshUntil = 0,
  currentReceiver = null,
  lastResult = null,
  statsPending = false,
  statsAt = 0,
  previewFrames = null,
  previewProgressAt = 0,
  previewRetryAt = 0,
  previewFailures = 0,
  supervisorState = "not-supervised";
const samples = [];
const history = [];
const components = [
  {
    name: "Integrated circuit",
    present: 2,
    missing: 0,
    x: 470,
    y: 300,
    w: 180,
    h: 125,
  },
  {
    name: "Screw terminal",
    present: 1,
    missing: 3,
    x: 740,
    y: 355,
    w: 140,
    h: 100,
  },
];
function rounded(context, x, y, w, h, r, color) {
  context.fillStyle = color;
  context.beginPath();
  context.roundRect(x, y, w, h, r);
  context.fill();
}
function schematic(context, missingIC, missingTerminal, t = 0) {
  context.save();
  context.scale(context.canvas.width / 1280, context.canvas.height / 720);
  context.fillStyle = "#e8ebed";
  context.fillRect(0, 0, 1280, 720);
  context.strokeStyle = "#dce1e4";
  context.lineWidth = 1;
  for (let x = 0; x < 1280; x += 40) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, 720);
    context.stroke();
  }
  for (let y = 0; y < 720; y += 40) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(1280, y);
    context.stroke();
  }
  context.fillStyle = "#d0d5d9";
  context.beginPath();
  context.ellipse(640, 395, 390, 260, 0, 0, Math.PI * 2);
  context.fill();
  context.fillStyle = "#dce0e3";
  context.beginPath();
  context.ellipse(640, 375, 390, 260, 0, 0, Math.PI * 2);
  context.fill();
  context.translate(640, 370);
  context.rotate(Math.sin(t / 8) * 0.015);
  context.translate(-640, -370);
  rounded(context, 335, 210, 610, 320, 14, "#265a51");
  rounded(context, 347, 222, 586, 296, 8, "#326e61");
  context.strokeStyle = "#76a58b";
  context.lineWidth = 3;
  for (let i = 0; i < 9; i++) {
    context.beginPath();
    context.moveTo(380, 255 + i * 27);
    context.lineTo(420 + i * 13, 255 + i * 27);
    context.lineTo(460 + i * 13, 285 + i * 20);
    context.lineTo(885, 285 + i * 20);
    context.stroke();
  }
  for (const [x, y] of [
    [365, 240],
    [915, 240],
    [365, 500],
    [915, 500],
  ]) {
    context.fillStyle = "#d6d5bd";
    context.beginPath();
    context.arc(x, y, 10, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "#737e7a";
    context.beginPath();
    context.arc(x, y, 5, 0, Math.PI * 2);
    context.fill();
  }
  for (let i = 0; i < 10; i++) {
    rounded(context, 475 + i * 17, 292, 8, 15, 1, "#c1c4b8");
    rounded(context, 475 + i * 17, 419, 8, 15, 1, "#c1c4b8");
  }
  if (!missingIC) {
    rounded(context, 470, 305, 180, 112, 5, "#202c32");
    context.fillStyle = "#b4c2c2";
    context.font = "15px sans-serif";
    context.fillText("EDGE • IC", 503, 352);
    context.font = "11px sans-serif";
    context.fillText("ILLUSTRATIVE", 503, 373);
    context.beginPath();
    context.arc(484, 319, 3, 0, Math.PI * 2);
    context.fill();
  }
  if (!missingTerminal) {
    rounded(context, 745, 360, 135, 88, 5, "#79a945");
    for (let i = 0; i < 3; i++) {
      rounded(context, 757 + i * 39, 372, 30, 42, 3, "#405b35");
      context.fillStyle = "#b7beb0";
      context.beginPath();
      context.arc(772 + i * 39, 386, 9, 0, Math.PI * 2);
      context.fill();
      context.strokeStyle = "#5f685b";
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(766 + i * 39, 386);
      context.lineTo(778 + i * 39, 386);
      context.stroke();
    }
  }
  for (let i = 0; i < 5; i++) {
    rounded(context, 400 + i * 24, 455, 13, 29, 2, "#c5b995");
    rounded(context, 696, 265 + i * 22, 25, 10, 2, "#c8b991");
  }
  context.fillStyle = "#c6dfce";
  context.font = "12px sans-serif";
  context.fillText("PORTABLE VISION / DEMONSTRATION PCB", 438, 493);
  context.restore();
}
function boxes(detections) {
  for (const d of detections) {
    const q = d.source_quad;
    if (
      !Array.isArray(q) ||
      q.length !== 4 ||
      !q.every(
        (p) => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite),
      )
    )
      continue;
    const color = [0, 3].includes(d.class_id) ? "#a52f38" : "#176447";
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    q.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
    ctx.closePath();
    ctx.stroke();
    const text = `${classes[d.class_id]}  ${Math.round(d.score * 100)}%`;
    ctx.font = "600 17px sans-serif";
    const width = ctx.measureText(text).width + 20;
    const x = Math.max(0, Math.min(1280 - width, q[0][0])),
      y = Math.max(27, q[0][1]);
    rounded(ctx, x, y - 29, width, 26, 3, color);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, x + 10, y - 10);
  }
}
function mockResult() {
  const states = ["present", "terminal", "present", "ic", "both", "unknown"];
  const state =
    scenario === "cycle"
      ? states[Math.floor((Date.now() - started) / 6500) % states.length]
      : scenario;
  const detections =
    state === "unknown"
      ? []
      : components.map((c) => ({
          class_id: (c.name === "Integrated circuit"
            ? ["ic", "both"]
            : ["terminal", "both"]
          ).includes(state)
            ? c.missing
            : c.present,
          score: c.name === "Integrated circuit" ? 0.974 : 0.948,
          source_quad: [
            [c.x - 12, c.y - 14],
            [c.x + c.w + 12, c.y - 14],
            [c.x + c.w + 12, c.y + c.h + 12],
            [c.x - 12, c.y + c.h + 12],
          ],
        }));
  return {
    state:
      state === "stale"
        ? "stale"
        : state === "disconnected"
          ? "disconnected"
          : "matched",
    crop_status: state === "unknown" ? "not_located" : "located",
    sequence: ++sequence,
    received_unix: Date.now() / 1000,
    processing_ms: 63,
    detections,
    mockState: state,
  };
}
function renderFindings(r) {
  $("detections").replaceChildren();
  for (const c of components) {
    const found =
      r.state === "matched"
        ? r.detections.filter((d) =>
            [c.present, c.missing].includes(d.class_id),
          )
        : [];
    const conflict =
      found.some((d) => d.class_id === c.present) &&
      found.some((d) => d.class_id === c.missing);
    const d = found.slice().sort((a, b) => b.score - a.score)[0];
    const label = conflict
      ? "Conflicting"
      : d
        ? d.class_id === c.missing
          ? "Missing"
          : "Present"
        : "Unknown";
    const row = document.createElement("article");
    row.className = "detection";
    const heading = document.createElement("h3");
    heading.textContent = c.name;
    const result = document.createElement("div");
    result.className = "result";
    const status = document.createElement("strong");
    status.textContent = label;
    status.className =
      label === "Missing"
        ? "absent"
        : ["Unknown", "Conflicting"].includes(label)
          ? "neutral"
          : "";
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = d && !conflict ? `${(d.score * 100).toFixed(1)}%` : "—";
    const small = document.createElement("small");
    small.textContent = "confidence";
    score.append(small);
    result.append(status, score);
    row.append(heading, result);
    $("detections").append(row);
  }
  $("count").textContent =
    r.state === "matched"
      ? `${r.detections.length} findings`
      : "No current findings";
  const signature =
    r.state === "matched"
      ? r.detections
          .map((d) => classes[d.class_id])
          .sort()
          .join(" · ") || "PCB not located"
      : r.state;
  if (signature !== previous) {
    previous = signature;
    history.unshift({
      text: signature,
      time: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
    });
    history.splice(3);
    $("history").replaceChildren(
      ...history.map((h) => {
        const li = document.createElement("li"),
          text = document.createElement("span"),
          time = document.createElement("time");
        text.textContent = h.text;
        time.textContent = h.time;
        li.append(text, time);
        return li;
      }),
    );
  }
}
function renderStatus(r) {
  const valid = r.state === "matched";
  $("empty").hidden = valid;
  $("frame").textContent = valid ? `Frame ${r.sequence}` : "Frame —";
  $("emptyTitle").textContent =
    r.state === "stale"
      ? "Inspection result is stale"
      : r.state === "unmatched"
        ? "Waiting for a matching frame"
        : "No current inspection";
  $("emptyText").textContent =
    r.state === "disconnected"
      ? "Receiver unavailable. Reconnecting automatically; no action needed while the receiver restarts."
      : "Overlays are cleared until a fresh, exact-matched result is available.";
  if (source === "live" && !valid) showRecoveryAdvice();
  renderFindings(r);
}
function showRecoveryAdvice() {
  const advice = {
    "cleanup-unconfirmed-manual-check-required":
      "Automatic session renewal stopped because worker cleanup could not be confirmed. Check the last session and board before restarting.",
    "retry-limit-reached":
      "Automatic session renewal reached its retry limit. Check the receiver logs and board connection, then restart the supervisor.",
    "low-disk-space-manual-check-required":
      "Automatic renewal stopped: less than 2 GiB of host disk space remains. Free space, then restart the supervisor.",
    "stopped-by-presenter":
      "Session supervisor was stopped. Start it again when ready; the dashboard will reconnect automatically.",
  };
  if (advice[supervisorState])
    $("emptyText").textContent = advice[supervisorState];
}
function resetLive() {
  liveAbort.abort();
  liveAbort = new AbortController();
  freshUntil = 0;
  lastResult = null;
  lastSequence = null;
  currentReceiver = null;
  failures = 0;
  samples.length = 0;
  statsAt = 0;
  previewFrames = null;
  previewProgressAt = 0;
  previewRetryAt = 0;
  previewFailures = 0;
  supervisorState = "not-supervised";
  $("live").removeAttribute("src");
  $("live").hidden = true;
}
function previewUnavailable() {
  $("live").hidden = true;
  $("live").removeAttribute("src");
  $("preview-status").hidden = source !== "live";
  $("preview-status").textContent =
    "Camera preview unavailable · reconnecting automatically";
}
function refreshPreview(stats) {
  if (!Number.isSafeInteger(stats.decoded_frames) || stats.decoded_frames < 0)
    return;
  if (stats.decoded_frames !== previewFrames) {
    previewFrames = stats.decoded_frames;
    previewProgressAt = Date.now();
  }
  if (Date.now() - previewProgressAt > 2000 || !stats.decoded_frames) return;
  if (!$("live").getAttribute("src") && Date.now() >= previewRetryAt) {
    $("live").src =
      `/bridge/live.mjpg?receiver_id=${encodeURIComponent(currentReceiver || "")}&t=${Date.now()}`;
    $("live").hidden = false;
    $("preview-status").hidden = true;
  }
}
function refreshStats(token) {
  if (statsPending || Date.now() - statsAt < 1000) return;
  statsPending = true;
  statsAt = Date.now();
  recovery
    .json("/bridge/stats.json", liveAbort.signal)
    .then((stats) => {
      if (token !== generation || stats.receiver_id !== currentReceiver) return;
      $("videoRate").textContent = Number.isFinite(stats.decoded_arrival_fps)
        ? stats.decoded_arrival_fps.toFixed(1)
        : "—";
      refreshPreview(stats);
      $("diagnostics").textContent = JSON.stringify(
        {
          ...stats,
          inspection_rate_note:
            "Distinct results received by dashboard, not board model throughput.",
        },
        null,
        2,
      );
    })
    .catch(async (error) => {
      if (token !== generation) return;
      $("videoRate").textContent = "—";
      $("diagnostics").textContent = error.message;
      try {
        const status = await recovery.json(
          "/bridge/status.json",
          liveAbort.signal,
        );
        if (token !== generation) return;
        supervisorState = status.supervisor_state;
        $("diagnostics").textContent = JSON.stringify(status, null, 2);
        if (!$("empty").hidden) showRecoveryAdvice();
      } catch {
        /* Receiver/server outage remains visible; the next bounded poll retries. */
      }
    })
    .finally(() => {
      statsPending = false;
    });
}
async function liveTick() {
  const token = generation;
  try {
    const r = recovery.normalize(
      await recovery.json("/bridge/inspection.json", liveAbort.signal),
    );
    if (token !== generation) return;
    if (r.receiver_id !== currentReceiver) {
      currentReceiver = r.receiver_id;
      lastSequence = null;
      samples.length = 0;
      previewFrames = null;
      previewProgressAt = 0;
      freshUntil = 0;
      lastResult = null;
      renderStatus({ state: "waiting", detections: [] });
      previewUnavailable();
    }
    if (r.state === "matched" && recovery.key(r) !== lastSequence) {
      const image = await recovery.image(
        `/bridge${r.image_url}?receiver_id=${encodeURIComponent(r.receiver_id || "")}`,
        liveAbort.signal,
      );
      if (token !== generation) return;
      const checked = recovery.normalize(r);
      if (checked.state === "matched") {
        ctx.clearRect(0, 0, 1280, 720);
        ctx.drawImage(image, 0, 0, 1280, 720);
        boxes(r.detections);
        lastSequence = recovery.key(r);
        samples.push(Date.now());
      } else r.state = checked.state;
    }
    if (r.state === "matched") {
      freshUntil =
        Date.now() +
        Math.max(0, recovery.FRESH_MS - (Date.now() - r.received_unix * 1000));
      lastResult = r;
    } else {
      freshUntil = 0;
      lastResult = null;
    }
    failures = 0;
    renderStatus(r);
    const age = Date.now() - Number(r.received_unix) * 1000;
    $("age").textContent = Number.isFinite(age)
      ? Math.max(0, Math.round(age))
      : "—";
    while (samples.length && samples[0] < Date.now() - 5000) samples.shift();
    $("aiRate").textContent =
      samples.length > 1
        ? (
            (samples.length - 1) /
            Math.max(0.001, (Date.now() - samples[0]) / 1000)
          ).toFixed(1)
        : "—";
    refreshStats(token);
  } catch (error) {
    if (token !== generation) return;
    failures++;
    freshUntil = 0;
    lastResult = null;
    renderStatus({ state: "disconnected", detections: [] });
    $("age").textContent = $("aiRate").textContent = "—";
    $("diagnostics").textContent = error.message;
    previewUnavailable();
  }
}
function healthTick() {
  if (source === "live") {
    refreshStats(generation);
    if (lastResult) {
      $("age").textContent = Math.max(
        0,
        Math.round(Date.now() - lastResult.received_unix * 1000),
      );
      if (Date.now() >= freshUntil) {
        renderStatus({ state: "stale", detections: [] });
        freshUntil = 0;
        lastResult = null;
        $("aiRate").textContent = "—";
      }
    }
    if (previewProgressAt && Date.now() - previewProgressAt > 2000) {
      previewUnavailable();
      $("videoRate").textContent = "—";
    }
  }
  setTimeout(healthTick, 100);
}
async function tick() {
  if (source === "mock") {
    if (!paused) {
      const r = mockResult();
      lastReceived = r.received_unix;
      schematic(
        ctx,
        ["ic", "both"].includes(r.mockState),
        ["terminal", "both"].includes(r.mockState),
      );
      boxes(r.detections);
      schematic(
        preview,
        ["ic", "both"].includes(r.mockState),
        ["terminal", "both"].includes(r.mockState),
        sequence / 15,
      );
      renderStatus(r);
      $("age").textContent = "0";
      $("aiRate").textContent = r.state === "matched" ? "15.8" : "—";
      $("videoRate").textContent = r.state === "disconnected" ? "—" : "30.0";
    } else {
      $("age").textContent = lastReceived
        ? Math.round(Date.now() - lastReceived * 1000)
        : "—";
      if (lastReceived && Date.now() - lastReceived * 1000 > 500)
        renderStatus({ state: "stale", detections: [] });
      $("aiRate").textContent = "0.0";
    }
  } else await liveTick();
  const seconds = Math.floor((Date.now() - started) / 1000);
  $("elapsed").textContent =
    `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  setTimeout(tick, source === "mock" ? 100 : recovery.retryDelay(failures));
}
$("fullscreen").onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $("inspection-video").requestFullscreen();
  } catch {
    $("diagnostics").textContent =
      "Fullscreen unavailable. Use your browser’s fullscreen command.";
  }
};
$("scenario").onchange = (e) => {
  scenario = e.target.value;
  previous = "";
};
$("pause").onclick = () => {
  paused = !paused;
  $("pause").replaceChildren();
  const icon = document.createElement("img");
  icon.src = `assets/${paused ? "play" : "pause"}.svg`;
  icon.alt = "";
  $("pause").append(icon, paused ? "Resume simulation" : "Pause simulation");
  $("pause").setAttribute("aria-pressed", String(paused));
};
$("source").onchange = (e) => {
  source = e.target.value;
  try {
    globalThis.localStorage?.setItem("portable-vision-source", source);
  } catch {
    /* Private browsing may block storage. */
  }
  generation++;
  resetLive();
  started = Date.now();
  lastSequence = null;
  previous = "";
  history.length = 0;
  $("pause").hidden = source === "live";
  $("viewTag").hidden = source !== "mock";
  $("viewTag").textContent = "SCHEMATIC SIMULATION · NOT CAMERA FOOTAGE";
  $("previewLabel").textContent =
    source === "mock" ? "Illustrative preview" : "Unannotated feed";
  $("preview").hidden = source !== "mock";
  $("live").hidden = true;
  $("preview-status").hidden = source !== "live";
  $("scenario").disabled = $("pause").disabled = source !== "mock";
  $("rateLabel").textContent =
    source === "mock" ? "AI inspection rate" : "Results received / second";
  $("sessionNote").textContent =
    source === "mock" ? "" : "Dashboard connected time";
  $("diagnostics").textContent = "Simulation mode. No hardware telemetry.";
  canvas.setAttribute(
    "aria-label",
    source === "mock"
      ? "Illustrative PCB inspection with component detection boxes"
      : "Exact-frame camera inspection with detection boxes",
  );
  renderStatus({ state: "waiting", detections: [] });
};
$("live").onerror = $("live").onload = () => {
  if (source !== "live") return;
  previewFailures++;
  previewRetryAt = Date.now() + recovery.retryDelay(previewFailures);
  previewUnavailable();
};
document.addEventListener?.("visibilitychange", () => {
  if (document.visibilityState === "visible" && source === "live") {
    generation++;
    resetLive();
    renderStatus({ state: "waiting", detections: [] });
  }
});
try {
  if (globalThis.localStorage?.getItem("portable-vision-source") === "live") {
    $("source").value = "live";
    $("source").onchange({ target: { value: "live" } });
  }
} catch {
  /* Storage is optional; source remains explicitly selectable. */
}
healthTick();
tick();
