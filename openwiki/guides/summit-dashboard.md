# Summit dashboard

The standalone dashboard in `frontend/` presents presence/absence detections, not an overall pass/fail judgement. It is designed for large 1080p/4K displays and includes a responsive narrow layout. On desktops wider than 1000px, the intro, inspection view and performance strip share a viewport-height presentation area; the image is contained without cropping or stretching. Supporting preview/history panels follow below. Very short windows retain a 590px minimum presentation area and can scroll. The introductory subtitle and redundant side connection text are removed; the header mode badge and inspection failure states remain. Actual browser visual validation and board integration remain required.

## Offline demo

```sh
python3 frontend/serve.py
```

Open http://127.0.0.1:8088. No npm install, network, board, or Python third-party packages required. Default mode is marked as schematic simulation inside the inspection image; there is no topbar source badge. The PCB is an authored schematic, not camera footage. Sample rates and confidence are illustrative, not board measurements. Open Presenter controls (settings icon) to choose both-present, either/both missing, not located, stale, or disconnected scenarios. Pause stops simulation updates; result age continues and the inspection becomes stale. Fullscreen is available from the inspection panel header and expands only the inspection viewport (image and detection overlays), not the dashboard. View tags are hidden in fullscreen; essential stale/disconnected messages remain. Escape exits. Presenter controls are a top-right anchored dropdown: Data source and Simulation scenario drill into stacked subpanels. Pause/Resume is permanently visible beside the headline, with a dynamic play/pause icon and pressed state (simulation only; hidden in live mode). Escape/Left goes back, Escape at root closes, Up/Down/Home/End navigate, Enter/Space selects, Tab dismisses, and outside clicks close. Choices show selection state and a brief confirmation; button press/hover/focus states are visible and reduced-motion preferences are respected. The redundant confidence disclaimer and frame-pairing caption are omitted from the display; these constraints still apply. The detections panel has no internal scrollbar. The smaller headline reserves more height for inspection; its title/fullscreen controls and legend float on translucent, backdrop-blurred overlays rather than consuming image height. The live preview description and PCB-location overlay text are removed; the frame identifier remains at the bottom right.

## Existing receiver integration

Start the normal portable launcher separately. Sessions remain bounded to 300 seconds. Copy its printed loopback panel URL:

```sh
python3 frontend/serve.py --receiver http://127.0.0.1:58078
```

Choose Live receiver in Presenter controls. The read-only bridge permits only `/inspection.json`, `/stats.json`, `/frames/<hash>.jpg`, `/live.mjpg` on the configured loopback receiver. It does not collect credentials, SSH, change models, displace workers, or extend session bounds. Update the receiver argument if a later session uses another panel port.

The inspected canvas draws `source_quad` coordinates only on its matching image. Stale (>500ms since host receipt), unmatched, missing-image, or disconnected results hide overlays and clear current findings. Opposing presence/absence predictions display Conflicting rather than selecting a verdict. Confidence is not accuracy. AI inspection rate in live mode is distinct results observed by the dashboard per elapsed receiving time, NOT actual model calls/s; video FPS comes from receiver decoded-arrival telemetry. Session elapsed is dashboard time, not worker uptime. No unit counter or yield is computed.

## Assets and verification

`frontend/assets/bits-wilp.png`: owner-supplied clipboard logo, original resolution; replace with high-resolution official artwork before 4K presentation. Icons are vendored from Lucide (ISC license in assets). All assets load locally. No recordings were available for this build.

Run `python3 -m pytest tests/test_dashboard_server.py -q`, `node --test tests/dashboard.test.cjs` and `node --check frontend/app.js`. Remaining acceptance: rendered 1080p/4K/narrow layout, keyboard/contrast checks, real frame pairing, receiver disconnect/restart and session expiry on the summit machine. No physical-board or accuracy qualification is implied.
