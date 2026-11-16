# Summit dashboard

`frontend/` presents IC and screw-terminal presence/absence detections, not an overall judgement. It is an offline-capable summit candidate, not a production-qualified inspection application. Large1080p/4K and narrow layouts are implemented; actual rendered Mac/board acceptance remains required.

## Run offline

```sh
python3 frontend/serve.py
```

Open http://127.0.0.1:8088. No npm install/network/board or Python third-party packages required. Default is schematic simulation unless Live was previously selected in browser storage; Live errors never silently switch to simulation. The image is labelled schematic, not footage; sample rates/confidence are illustrative. Presenter settings is a keyboard-accessible anchored drill-down source/scenario dropdown. Pause/Resume is beside the headline only in simulation, with a dynamic icon. The topbar has no source badge.

Desktop presentation budgets viewport height for inspection plus performance strip. Title/fullscreen controls and legend float over the image on backdrop-blurred translucent surfaces. Image scaling is contained, not cropped/stretched. Very short windows retain a590px minimum; narrow screens stack. Fullscreen expands only the inspection viewport; Escape exits. No PCB-location overlay, confidence disclaimer, preview description or inner findings scrollbar. Frame ID remains at bottom right.

## Live and automatic recovery

By default the server discovers running loopback receiver URLs under `local/runs/*/session.json`, newest records first, max128 parsed. It tolerates partial writes and ignores completed, invalid, future, external-origin and >600s-old sessions. The default dashboard URL remains8088 while receiver ports change. A custom `--session-root` must match the launcher/supervisor output root.

For a specific existing receiver, use `python3 frontend/serve.py --receiver http://127.0.0.1:PORT`; this pins the port and disables discovery. Settings → Live receiver connects read-only. The bridge allowlists inspection/stats/JPEG/MJPEG endpoints and a sanitized recovery-status endpoint, rejects cross-session frame requests, limits body/connection/stream resources, restricts serving to UI assets and loopback Host headers, and never starts SSH or a camera itself. Server responses disable caching and add content-type/CSP/frame-embedding protections.

Client HTTP/image deadlines, capped retry backoff, independent freshness watchdog, preview reconnects, tab-resume cancellation, frame/session identity checks and independent stats polling prevent an unavailable endpoint from permanently stopping updates. Malformed/conflicting/stale/unmatched results never become a present/pass verdict. Full recovery behavior and safety stops: [summit runbook](../operations/summit-runbook.md).

## Session renewal

The optional root `run-summit.py` supervises sequential calls to the original frozen launcher. It requires a private profile and prompt-once or existing key authentication. Natural completed runs renew automatically; confirmed-ended failures have bounded retries; uncertain cleanup/auth/preflight failure stops safely. It does not remove the300s worker bound, modify the frozen package, bypass hashes, displace a decoder owner, install services or alter the default `run-portable.py` path. Morning two-cycle/hardware acceptance is mandatory before trusting continuous renewal. See the runbook for exact commands, temporary credential handling and disk/evidence caveats.

## Measurement truth

Canvas boxes use `source_quad` only on their exact hash-matched inspected image. A500ms host-receipt freshness threshold is checked every100ms independently of fetch progress; browser scheduling can delay execution. Result age is not optical latency. Conflicting presence/absence candidates display Conflicting. Confidence is not accuracy. AI rate in Live is a rolling estimate of distinct results observed by the dashboard, not board model calls/s; video FPS is receiver decoded-arrival telemetry, not browser render FPS. Session elapsed is dashboard time. No unit counter or yield is computed.

## Assets and tests

Owner-supplied `frontend/assets/bits-wilp.png` is unchanged from the clipboard image; obtain higher-resolution artwork before4K presentation. Lucide SVGs and license are vendored locally. No real footage was available for this build.

```sh
python3 -m pytest tests/test_dashboard_server.py tests/test_dashboard_recovery.py tests/test_summit_supervisor.py -q
node --test tests/dashboard.test.cjs tests/recovery.test.cjs
python3 -B deployment/portable-v1/verify.py
```

Fault-injection tests cover HTTP outages, malformed/oversized payloads, discovery/port changes, stale requests, image hangs/abort, freshness, preview retries, source switches, supervisor limits/locks/cleanup evidence, and private-profile deletion. These are not real browser, Mac wheel, camera, decoder or hardware-soak tests.
