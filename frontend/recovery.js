/* Small, independently tested recovery primitives shared by the live dashboard. */
(() => {
  const FRESH_MS = 500;
  function normalize(raw, now = Date.now()) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw))
      throw Error("Invalid inspection object");
    const age = now - Number(raw.received_unix) * 1000;
    const result = { ...raw, detections: [] };
    const states = ["waiting", "matched", "unmatched", "stale", "disconnected"];
    result.state = states.includes(raw.state) ? raw.state : "unmatched";
    if (!Number.isFinite(age) || age < -100 || age > FRESH_MS)
      result.state = "stale";
    if (!Number.isSafeInteger(raw.sequence) || raw.sequence < 0)
      result.state = "unmatched";
    if (result.state !== "matched") return result;
    if (
      !/^\/frames\/[a-f0-9]{64}\.jpg$/.test(raw.image_url || "") ||
      !Array.isArray(raw.detections) ||
      raw.detections.length > 100
    ) {
      result.state = "unmatched";
      return result;
    }
    result.detections = raw.detections.filter(
      (d) =>
        d &&
        Number.isInteger(d.class_id) &&
        d.class_id >= 0 &&
        d.class_id < 4 &&
        Number.isFinite(d.score) &&
        d.score >= 0 &&
        d.score <= 1 &&
        Array.isArray(d.source_quad) &&
        d.source_quad.length === 4 &&
        d.source_quad.every(
          (p) =>
            Array.isArray(p) &&
            p.length === 2 &&
            p.every(Number.isFinite) &&
            Math.abs(p[0]) <= 12800 &&
            Math.abs(p[1]) <= 7200,
        ),
    );
    // A malformed detection packet must not become an apparently valid empty finding.
    if (result.detections.length !== raw.detections.length) {
      result.state = "unmatched";
      result.detections = [];
    }
    return result;
  }
  async function json(url, signal, timeout = 1800) {
    const controller = new AbortController();
    const cancel = () => controller.abort();
    if (signal?.aborted) controller.abort();
    signal?.addEventListener("abort", cancel, { once: true });
    const timer = setTimeout(cancel, timeout);
    try {
      const response = await fetch(url, {
        signal: controller.signal,
        cache: "no-store",
      });
      if (!response.ok)
        throw Error(
          `Receiver HTTP ${response.status}; reconnecting automatically`,
        );
      return await response.json();
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", cancel);
    }
  }
  function image(url, signal, timeout = 1200) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      let timer;
      const clean = () => {
        clearTimeout(timer);
        img.onload = img.onerror = null;
        signal?.removeEventListener("abort", cancel);
      };
      const fail = (message) => {
        clean();
        img.src = "";
        reject(Error(message));
      };
      const cancel = () => fail("Inspection image cancelled");
      img.onload = () => {
        clean();
        resolve(img);
      };
      img.onerror = () => fail("Inspection image unavailable; retrying");
      if (signal?.aborted) {
        cancel();
        return;
      }
      signal?.addEventListener("abort", cancel, { once: true });
      timer = setTimeout(
        () => fail("Inspection image timed out; retrying"),
        timeout,
      );
      img.src = url;
    });
  }
  function key(result) {
    return `${result.receiver_id || "receiver"}:${result.image_url}:${result.sequence}:${result.received_unix}`;
  }
  function retryDelay(failures) {
    return Math.min(3000, 150 * 2 ** Math.min(failures, 5));
  }
  globalThis.VisionRecovery = Object.freeze({
    normalize,
    json,
    image,
    key,
    retryDelay,
    FRESH_MS,
  });
})();
