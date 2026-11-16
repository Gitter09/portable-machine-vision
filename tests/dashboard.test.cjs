// Offline state/contract tests. No browser or third-party packages required.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
function element() {
  return {
    hidden: false,
    textContent: "",
    children: [],
    width: 1280,
    height: 720,
    append(...children) {
      this.children.push(...children);
    },
    replaceChildren(...children) {
      this.children = children;
    },
    setAttribute(name, value) {
      this[name] = value;
    },
    getAttribute(name) {
      return this[name] || null;
    },
    removeAttribute(name) {
      delete this[name];
    },
    getContext() {
      return new Proxy(
        { canvas: this },
        {
          get: (target, key) =>
            key in target
              ? target[key]
              : key === "measureText"
                ? () => ({ width: 100 })
                : () => {},
        },
      );
    },
  };
}
function app(extras = {}) {
  const elements = new Map();
  const get = (id) => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  const sandbox = {
    document: { getElementById: get, createElement: element },
    setTimeout: () => {},
    clearTimeout: () => {},
    Date,
    console,
    AbortSignal,
    AbortController,
    ...extras,
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync("frontend/recovery.js", "utf8"), sandbox);
  vm.runInContext(fs.readFileSync("frontend/app.js", "utf8"), sandbox);
  return { run: (code) => vm.runInContext(code, sandbox), get };
}
test("pause button switches its icon and label on each click", () => {
  const { run, get } = app();
  run("document.getElementById('pause').onclick()");
  assert.equal(get("pause").children[0].src, "assets/play.svg");
  assert.equal(get("pause").children[1], "Resume simulation");
  run("document.getElementById('pause').onclick()");
  assert.equal(get("pause").children[0].src, "assets/pause.svg");
  assert.equal(get("pause").children[1], "Pause simulation");
});
test("pause button is hidden for Live and restored for simulation", () => {
  const { run, get } = app();
  run("document.getElementById('source').onchange({target:{value:'live'}})");
  assert.equal(get("pause").hidden, true);
  run("document.getElementById('source').onchange({target:{value:'mock'}})");
  assert.equal(get("pause").hidden, false);
});
test("fullscreen requests only the inspection viewport", async () => {
  const { run } = app();
  run(
    "document.getElementById('inspection-video').requestFullscreen = async () => { document.requestedTarget = 'inspection-video'; }",
  );
  await run("document.getElementById('fullscreen').onclick()");
  assert.equal(run("document.requestedTarget"), "inspection-video");
});
test("simulation preserves four-class IDs for every component state", () => {
  const { run } = app();
  for (const [scenario, expected] of [
    ["present", [2, 1]],
    ["ic", [0, 1]],
    ["terminal", [2, 3]],
    ["both", [0, 3]],
    ["unknown", []],
  ]) {
    const result = run(`scenario='${scenario}';mockResult()`);
    assert.deepEqual(
      Array.from(result.detections, (d) => d.class_id),
      expected,
    );
  }
});
test("stale, unmatched, and disconnected hide overlays and current findings", () => {
  const { run, get } = app();
  for (const state of ["stale", "unmatched", "disconnected"]) {
    run(`renderStatus({state:'${state}',detections:[{class_id:2,score:.99}]})`);
    assert.equal(get("empty").hidden, false);
    assert.equal(get("count").textContent, "No current findings");
    assert.equal(
      get("detections").children[0].children[1].children[0].textContent,
      "Unknown",
    );
  }
});
test("conflicting predictions do not silently choose a present state", () => {
  const { run, get } = app();
  run(
    "renderStatus({state:'matched',detections:[{class_id:2,score:.99},{class_id:0,score:.9}]})",
  );
  assert.equal(
    get("detections").children[0].children[1].children[0].textContent,
    "Conflicting",
  );
});
function livePacket() {
  return {
    state: "matched",
    sequence: 1,
    receiver_id: "session-a",
    received_unix: Date.now() / 1000,
    image_url: "/frames/" + "a".repeat(64) + ".jpg",
    detections: [],
  };
}
class ImmediateImage {
  set src(value) {
    if (value) queueMicrotask(() => this.onload?.());
  }
}
test("freshness watchdog clears a displayed result without waiting for HTTP", () => {
  const { run, get } = app();
  run(
    "source='live';lastResult={received_unix:(Date.now()-1000)/1000};freshUntil=Date.now()-1;healthTick()",
  );
  assert.equal(get("empty").hidden, false);
  assert.equal(get("emptyTitle").textContent, "Inspection result is stale");
});
test("inspection recovers after receiver failure and ignores unavailable stats", async () => {
  let failed = true;
  const fetch = async (url) => {
    if (url.includes("stats")) return new Promise(() => {});
    return failed
      ? { ok: false, status: 503 }
      : { ok: true, json: async () => livePacket() };
  };
  const { run, get } = app({ fetch, Image: ImmediateImage });
  run("document.getElementById('source').onchange({target:{value:'live'}})");
  await run("liveTick()");
  assert.equal(get("empty").hidden, false);
  failed = false;
  await run("liveTick()");
  assert.equal(get("empty").hidden, true);
  assert.equal(run("failures"), 0);
});
test("camera preview reconnects after a failed stream and hides a stalled preview", () => {
  const { run, get } = app();
  run(
    "source='live';currentReceiver='session-a';refreshPreview({decoded_frames:10})",
  );
  assert.equal(get("live").hidden, false);
  assert.match(get("live").src, /receiver_id=session-a/);
  run("document.getElementById('live').onerror()");
  assert.equal(get("live").hidden, true);
  run("previewRetryAt=0;refreshPreview({decoded_frames:11})");
  assert.equal(get("live").hidden, false);
  run("previewProgressAt=Date.now()-3000;healthTick()");
  assert.equal(get("live").hidden, true);
});
test("manual-intervention safety stops are shown instead of promising a restart", () => {
  const { run, get } = app();
  run(
    "source='live';supervisorState='cleanup-unconfirmed-manual-check-required';renderStatus({state:'disconnected',detections:[]})",
  );
  assert.match(get("emptyText").textContent, /cleanup could not be confirmed/);
});
test("source switch cancels old inspection results before they can draw", async () => {
  let release;
  const fetch = () =>
    new Promise((resolve) => {
      release = resolve;
    });
  const { run, get } = app({ fetch, Image: ImmediateImage });
  run("source='live'");
  const pending = run("liveTick()");
  run("document.getElementById('source').onchange({target:{value:'mock'}})");
  release({ ok: true, json: async () => livePacket() });
  await pending;
  assert.equal(run("lastSequence"), null);
  assert.equal(get("count").textContent, "No current findings");
});
