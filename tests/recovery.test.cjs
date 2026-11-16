const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
function helpers(extras = {}) {
  const sandbox = {
    Date,
    setTimeout,
    clearTimeout,
    AbortController,
    ...extras,
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync("frontend/recovery.js", "utf8"), sandbox);
  return sandbox.VisionRecovery;
}
function packet(overrides = {}) {
  return {
    state: "matched",
    sequence: 1,
    receiver_id: "a",
    received_unix: Date.now() / 1000,
    image_url: "/frames/" + "a".repeat(64) + ".jpg",
    detections: [
      {
        class_id: 2,
        score: 0.98,
        source_quad: [
          [0, 0],
          [20, 0],
          [20, 20],
          [0, 20],
        ],
      },
    ],
    ...overrides,
  };
}
test("stale and future-dated results cannot be displayed", () => {
  const h = helpers();
  assert.equal(
    h.normalize(packet({ received_unix: (Date.now() - 501) / 1000 })).state,
    "stale",
  );
  assert.equal(
    h.normalize(packet({ received_unix: (Date.now() + 5000) / 1000 })).state,
    "stale",
  );
  assert.equal(h.normalize(packet()).state, "matched");
});
test("malformed packets, geometry, confidence and frame paths fail closed", () => {
  const h = helpers();
  for (const override of [
    { detections: {} },
    { detections: [null] },
    { detections: [{ class_id: 2, score: NaN }] },
    { sequence: -1 },
    { image_url: "https://example.com/frame.jpg" },
    { detections: Array(101).fill({}) },
  ]) {
    assert.notEqual(h.normalize(packet(override)).state, "matched");
  }
  assert.throws(() => h.normalize(null));
});
test("restarted sequences and identical images get a new pairing key", () => {
  const h = helpers(),
    r = packet();
  assert.notEqual(h.key(r), h.key({ ...r, receiver_id: "new-session" }));
  assert.notEqual(
    h.key(r),
    h.key({ ...r, received_unix: r.received_unix + 1 }),
  );
});
test("hung image load rejects within its bound and cancels the image", async () => {
  let instance;
  class Image {
    constructor() {
      instance = this;
    }
  }
  const h = helpers({ Image });
  await assert.rejects(h.image("/frame", undefined, 20), /timed out/);
  assert.equal(instance.src, "");
});
test("image loading is cancelled when data source changes", async () => {
  class Image {}
  const h = helpers({ Image }),
    controller = new AbortController();
  const pending = h.image("/frame", controller.signal, 500);
  controller.abort();
  await assert.rejects(pending, /cancelled/);
});
test("successful image loads resolve without waiting for a decode promise", async () => {
  class Image {
    set src(value) {
      this.url = value;
      if (value) queueMicrotask(() => this.onload?.());
    }
  }
  const h = helpers({ Image });
  assert.equal((await h.image("/frame")).url, "/frame");
});
test("hung HTTP requests abort and subsequent requests recover", async () => {
  let fail = true;
  const fetch = (_url, { signal }) =>
    fail
      ? new Promise((resolve, reject) =>
          signal.addEventListener("abort", () => reject(Error("aborted"))),
        )
      : Promise.resolve({ ok: true, json: async () => ({ state: "matched" }) });
  const h = helpers({ fetch });
  await assert.rejects(h.json("/inspection", undefined, 20), /aborted/);
  fail = false;
  assert.equal((await h.json("/inspection")).state, "matched");
});
test("invalid HTTP and JSON responses reject without disabling future polling", async () => {
  const h = helpers({ fetch: async () => ({ ok: false, status: 503 }) });
  await assert.rejects(h.json("/inspection"), /503/);
  const malformed = helpers({
    fetch: async () => ({
      ok: true,
      json: async () => {
        throw Error("bad JSON");
      },
    }),
  });
  await assert.rejects(malformed.json("/inspection"), /bad JSON/);
});
test("reconnect backoff stays bounded even after long outages", () => {
  const h = helpers();
  assert.equal(h.retryDelay(0), 150);
  assert.equal(h.retryDelay(1000), 3000);
});
