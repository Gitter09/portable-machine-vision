# Summit runbook: recovery-enabled dashboard

## Readiness

This is a recovery-enabled **candidate**, not a production or Mac/board qualification. Offline HTTP fault injection, browser-state logic and supervisor-policy tests run without the board. Real video pairing, Mac permissions, fullscreen rendering, multiple renewal cycles and extended stability must pass onsite before relying on unattended renewal. Frozen `deployment/portable-v1` remains hash-identical; no board services or models were changed.

## Two terminals, one dashboard URL

Complete `MAC_TESTING.md` setup first (native arm64 Python 3.11, Ethernet route, offline dependencies). Do not start a second camera feed or inference launcher.

Terminal 1:

```sh
./local/host-env/bin/python frontend/serve.py
```

Open **http://127.0.0.1:8088**. If an old dashboard server is already running, stop that terminal with Ctrl+C and restart it so the recovery-enabled server is loaded. Settings → Data source → Live receiver. Source selection is remembered where browser local storage is available; errors never silently substitute simulation.

Terminal 2, initial two-cycle acceptance:

```sh
mkdir -p local
test -e local/summit-link.json || cp config/portable-link.example.json local/summit-link.json
# Edit local/summit-link.json for the existing board/camera link if necessary.
./local/host-env/bin/python -B run-summit.py \
  --config local/summit-link.json --board-host 192.168.1.20 \
  --ask-password --seconds 60 --max-sessions 2
```

Enter the board password **in your terminal**, not in chat. The supervisor prompts once and creates a private temporary profile (directory private, file mode0600 on Mac), reused for child sessions and deleted on ordinary exit/Ctrl+C. It is never committed or logged. A forced kill/power loss may leave this file under the OS temporary directory with prefix `portable-vision-auth-`; remove that directory after stopping the supervisor if necessary. Existing SSH-key authentication is an alternative: replace `--ask-password` with `--ssh-key /path/to/existing/key`; no keys are installed automatically. Encrypted keys require an already functioning agent/noninteractive setup.

After the two-cycle and recovery tests pass, run the same command with `--seconds 300` and without `--max-sessions 2`. This opts into sequential renewal until Ctrl+C. Renewals have a **visible startup gap**, not seamless video. Each child still verifies the frozen package and uses its original 20..300s board bound. The dashboard discovers changing loopback ports from `local/runs/*/session.json` automatically. If using a custom `--session-root`, pass the same directory to both commands. `--receiver http://127.0.0.1:PORT` on the dashboard intentionally disables discovery; use it only for a fixed receiver.

## What recovers automatically

- Receiver unavailable/HTTP errors/partial JSON: bounded retries, backoff capped at3s, no page refresh required.
- Delayed/failed inspection JPEG: load deadline1.2s, then retry. Requests abort when switching source. Frame identity includes receiver session, image hash, sequence and host receipt time.
- Stale/future/invalid metadata: inspection findings disappear; watchdog checks every100ms against a500ms host-receipt freshness threshold, independently of HTTP/image progress. Browser scheduling can delay checks; this is not optical latency.
- Stats failure: cannot block or invalidate otherwise fresh inspections. Stats poll independently about once/second.
- Broken/ended MJPEG: retry with cache-busting URL and bounded backoff. Unchanging decoded-frame telemetry for2s hides the preview instead of displaying a frozen frame as live. Telemetry progress is not proof of browser render FPS.
- Receiver restart/new session/port: discover fresh running records (maximum600s old), reject cross-session frame requests, reacquire source image and preview. Existing requests carry a session identity; stale results cannot overwrite a source switch.
- Browser tab resume: cancel old requests, discard previous result and request fresh data.
- Natural bounded-session completion: opt-in supervisor starts the next session **only after** child exit and terminal worker/host evidence. Confirmed-ended failures retry with a limit (default3 consecutive failures); successful sessions reset this count.

## What deliberately requires intervention

- Cleanup cannot be confirmed, worker reports missing, host watchdog expires, host-key/authentication failure, or decoder occupied: supervisor stops rather than steal hardware or blindly create overlapping jobs. The existing board job remains self-time-bounded; inspect the last session before restarting.
- Retry limit reached or host free disk below2GiB: stop with an actionable dashboard message. Evidence is never deleted automatically. Board free space still requires operator monitoring; repeated sessions create board job directories.
- Stop requested by presenter: no automatic renewal. Ctrl+C the supervisor stops only its owned host process group and delegates exact-owned-board-worker cleanup to the frozen launcher. Failed SSH cleanup cannot guarantee immediate board stop.
- Board/camera power loss, Ethernet routing/firewall changes, laptop sleep/power loss, dashboard-server termination: fix the physical/host issue or restart the relevant terminal. No OS autostart service, power management, broad process kill or network reconfiguration is installed. Keep the Mac awake on power for the demo.

A per-checkout OS lock prevents two summit supervisors even with different session roots. On Mac/POSIX the launcher inherits this lock, retaining ownership if the supervising parent crashes while that launcher is still alive. Do not run another clone, ordinary launcher or capture feed concurrently. The frozen busy check is still enforced. Windows process-group cleanup has not been exercised in this recovery pass.

## Morning go/no-go checklist

1. Confirm `python3.11 -B deployment/portable-v1/verify.py` passes55 files and offline environment checks pass.
2. Two60s sessions: second starts only after first ends; same dashboard URL reconnects without manual port edits. Logs remain free of worker/receiver errors and samples use exact matched frames.
3. Stop receiver/supervisor: overlay clears within roughly freshness threshold + scheduling delay; no old frames are presented as current. Restart with new port: image and preview recover automatically.
4. Briefly disconnect Ethernet, restore it, inspect final cleanup reports. If cleanup is unconfirmed the supervisor **must stop**; verify the board is idle before manually restarting. Dashboard should reconnect by itself once a safe new receiver is running.
5. Pause/resume simulation and switch back to Live: simulation never appears automatically during a Live failure. Preview and inspection fullscreen, Escape, menu,1080p/4K layout and keyboard focus work.
6. At least20–30min under the300s renewal mode with known specimens. Check disk space, thermal/load stability, scene findings, RTP gaps and recovery gaps. This is a summit acceptance check, not production accuracy/soak qualification.

## Logs and fallback

`local/runs/summit-status.json` records supervisor state. Each `summit-*` directory contains the frozen launcher's session/worker/receiver evidence. Check those before a manual restart; never share passwords/private profiles. Receiver diagnostics includes supervisor safety-stop state when available.

If automatic renewal fails qualification, use the original bounded `run-portable.py` flow, with dashboard auto-discovery still enabled. If the board is unavailable, choose Offline simulation explicitly; it is visibly schematic, not recorded footage. Keep an actual recording as a stronger fallback when the original footage becomes available.
