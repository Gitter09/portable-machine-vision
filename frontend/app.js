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
  lastReceived = null,
  liveCount = 0,
  liveStart = null;
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
      ? "Receiver unavailable. Check the local receiver and its session, then try live mode again."
      : "Overlays are cleared until a fresh, exact-matched result is available.";
  renderFindings(r);
}
async function liveTick() {
  const token = generation;
  try {
    const response = await fetch("/bridge/inspection.json", {
      signal: AbortSignal.timeout(2000),
      cache: "no-store",
    });
    if (!response.ok) throw Error("Receiver unavailable");
    const r = await response.json();
    if (token !== generation) return;
    const age = Date.now() - r.received_unix * 1000;
    if (!Number.isFinite(age) || age > 500) r.state = "stale";
    r.detections = (r.detections || []).filter(
      (d) =>
        Number.isInteger(d.class_id) &&
        d.class_id >= 0 &&
        d.class_id < 4 &&
        Number.isFinite(d.score) &&
        d.score >= 0 &&
        d.score <= 1,
    );
    if (
      r.state === "matched" &&
      r.image_url &&
      /^\/frames\/[a-f0-9]+\.jpg$/.test(r.image_url)
    ) {
      if (r.sequence !== lastSequence) {
        const image = new Image();
        image.src = "/bridge" + r.image_url;
        await image.decode();
        if (token !== generation) return;
        if (Date.now() - r.received_unix * 1000 > 500) {
          r.state = "stale";
        } else {
          ctx.clearRect(0, 0, 1280, 720);
          ctx.drawImage(image, 0, 0, 1280, 720);
          boxes(r.detections);
          lastSequence = r.sequence;
          liveCount++;
          liveStart ??= Date.now();
        }
      }
    } else if (r.state === "matched") r.state = "unmatched";
    renderStatus(r);
    $("age").textContent = Number.isFinite(age)
      ? Math.max(0, Math.round(Date.now() - r.received_unix * 1000))
      : "—";
    $("aiRate").textContent =
      liveStart && Date.now() - liveStart > 1000
        ? (liveCount / ((Date.now() - liveStart) / 1000)).toFixed(1)
        : "—";
    $("sessionNote").textContent = "Dashboard connected time";
    const statsResponse = await fetch("/bridge/stats.json", {
      signal: AbortSignal.timeout(2000),
    });
    if (!statsResponse.ok) throw Error("Stats unavailable");
    const stats = await statsResponse.json();
    if (token !== generation) return;
    $("videoRate").textContent = Number.isFinite(stats.decoded_arrival_fps)
      ? stats.decoded_arrival_fps.toFixed(1)
      : "—";
    $("diagnostics").textContent = JSON.stringify(
      {
        ...stats,
        inspection_rate_note:
          "Observed distinct results received by dashboard; not board model-call throughput.",
      },
      null,
      2,
    );
  } catch (error) {
    if (token !== generation) return;
    renderStatus({ state: "disconnected", detections: [] });
    $("age").textContent =
      $("aiRate").textContent =
      $("videoRate").textContent =
        "—";
    $("diagnostics").textContent = error.message;
  }
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
  setTimeout(tick, source === "mock" ? 100 : 120);
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
  generation++;
  started = Date.now();
  lastSequence = null;
  liveCount = 0;
  liveStart = null;
  previous = "";
  history.length = 0;
  $("pause").hidden = source === "live";
  $("viewTag").hidden = source !== "mock";
  $("viewTag").textContent = "SCHEMATIC SIMULATION · NOT CAMERA FOOTAGE";
  $("previewLabel").textContent =
    source === "mock" ? "Illustrative preview" : "Unannotated feed";
  $("preview").hidden = source !== "mock";
  $("live").hidden = source !== "live";
  if (source === "live") $("live").src = "/bridge/live.mjpg";
  else $("live").removeAttribute("src");
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
tick();
