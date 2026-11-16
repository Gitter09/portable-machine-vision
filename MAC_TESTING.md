# Mac M3: clone-only setup and testing

## Before turning off / packing hardware

Carry SiMa power, camera power/PoE, existing switch, cables and Mac USB-C Ethernet
adapter. Existing board SDK/plugins and camera configuration must remain installed.
This repo includes model, native Linux library, runtime and OFFLINE host wheels;
no Windows PC, training directory, compiler host or external artifact is needed.
Native CPython3.11 arm64 must be installed on Mac BEFORE offline travel. Python
itself/macOS/board OS are platform prerequisites. Actual macOS27 testing is yours;
we did not execute the Mac wheels here.

## 1. Clone and check dependencies are present

```sh
git clone https://github.com/gitter09/portable-machine-vision.git
cd portable-machine-vision
python3.11 -c 'import platform; print(platform.machine())'
python3.11 -B deployment/portable-v1/verify.py
```

Architecture MUST print `arm64`, not `x86_64`/Rosetta. Verification must pass55
files. All payload files are ordinary Git files, no LFS/artifact download needed.
Keep checkout path reasonably short; do not edit frozen deployment payload.

## 2. Wire private network

Connect Mac + board + camera to SAME isolated switch. Summit Wi-Fi/router is not
required. Board currently has192.168.1.20/24; camera192.168.1.10. No board network
settings were changed by this work. Configure Mac ETHERNET adapter manually:

- IPv4 e.g.192.168.1.30 (ensure not used elsewhere)
- Subnet255.255.255.0
- Router/DNS blank for this link

If Wi-Fi uses same subnet, disable it while testing or resolve route conflict.
Never overwrite existing camera/board IP blindly. Check:

```sh
ping -c 3 192.168.1.20
route -n get 192.168.1.20
```

Route should be Ethernet adapter, NOT Wi-Fi. Application detects/rejects a known
Wi-Fi route. Address examples are operator configuration, not code constants.

## 3. Install host environment OFFLINE inside clone

```sh
python3.11 setup-host.py
./local/host-env/bin/python -m pip check
```

Installs bundled Mac arm64 wheels using --no-index, creates `local/host-env`, tests
imports/H264 codec. Requires no PyPI/network. If local/host-env already exists,
setup refuses to overwrite it: reuse it or remove/rename only that local env after
stopping all runs. Never modify system Python. Runtime logs go to local/runs.

## 4. Start viewer (password prompted)

Put known labelled PCB specimen in view, camera powered, plate clear/safe:

```sh
./local/host-env/bin/python -B run-portable.py \
  --board-host 192.168.1.20 --seconds 60
```

Password is entered invisibly at prompt. No secret stored in Git. The checked-in
public SSH fingerprint pins the actual SiMa board identity; identity mismatch
must be investigated, NOT bypassed. If your existing SSH-key authentication works,
use local profile/--ssh-key options instead; never commit private keys.

Allow Python Local Network/incoming network access if macOS prompts. Do not disable
firewall globally. UDP5010/5012 must be free. App selects local return IPv4 via
route to board and binds SSH to that source; no hardcoded workstation address.
Browser URL prints in Terminal and is recorded in local/runs/<job>/session.json.
Open it manually if browser does not open automatically.

- Left pane: live camera (expected around30 decodedFPS, render FPS not measured).
- Right pane: coordinates drawn on EXACT hash-matched inspected frame, not old
  boxes on current uninspected video. UNKNOWN/stale/unmatched/disconnected is NOT pass.
- Expected prior Windows/board benchmark about15–16 actual model calls/sec with
  PCB located. Mac display/matching performance still needs measurement.
- No-board UNKNOWN loop rate is NOT model inference FPS.

## 5. Stop/restart and checks

Press Ctrl+C in launch Terminal. Host requests stop only for its exact owned board
script/output; worker releases decoder. If link is lost, immediate stop cannot be
guaranteed: session20..300s self-bound plus timeout remains. Wait/check session
before restarting; never start separate camera graphs simultaneously.

Inspect latest local/runs job: session.json, worker.json, receiver.json,
transform-preflight.json. Confirm:

- Worker completed/stopped without errors.
- feature_roi/track_pose/coordinates_only true; cpp_input_kernel parallel8.
- Native preflight says packed bytes exact.
- Model calls = worker.located (NOT all inference_frames if unknown exists).
- Receiver host_errors empty; decoded video near30FPS; metadata/source matches.
- Zero packet gaps desirable, not assumed guaranteed. Matching can occur after
  metadata arrival; unmatched results must not display guessed overlays.

Restart with same command and NEW output (default automatic). Test60s, then300s.
Check known both-present, IC-missing/terminal-present, both-missing specimens,
several rotations/positions and temporary occlusion. Do not infer production
accuracy from smooth streaming alone. Existing diagnostic confidence.05 is not
an independently qualified production decision threshold.

## Troubleshooting / reporting

- No ping: cable/switch/power/manual Ethernet subnet/duplicate IP first.
- Wrong route: fix conflicting Wi-Fi/Ethernet subnets; no automatic network rewrite.
- Host-key error: verify board public key; do not turn off identity checks.
- Wheel/import failure: confirm arm64 CPython3.11 and inspect setup output. Wheels
  are macOS14+arm64 compatible builds, but reported macOS27 execution untested here.
- Board occupied: let owned prior run finish/stop it; never kill arbitrary processes.
- No image: allow local network/firewall, verify camera power and RTSP config.
- All UNKNOWN: verify PCB actually visible/within supported colour/scale/framing.
- Large delay: record both processing_ms and receiver stats; neither proves optical
  latency. Film real motion and display together to assess physical delay.

For help share release ID, Mac `sw_vers`, Python architecture/version, failure text,
worker/receiver/session/preflight JSON after checking for local addresses/secrets.
Do NOT share password, private key or local credentials profile.

## Current boundaries

This portable candidate is Windows-smoke-tested with owned stop/restart, not
Mac-qualified. Latest recent network tests had no PCB, not renewed model quality
validation. Earlier accepted board pipeline/quality evidence remains historical.
Sessions remain bounded300s, no unattended continuous supervisor/autostart/soak
qualification yet. Frontend lifecycle/auth/retention work is separate.

Repository runtime is self-contained. Some historical development/training scripts
reference old sibling evidence folders; they are NOT required by root setup-host.py
or run-portable.py. Historical Windows run-production.ps1 requires old local
selection and is NOT the Mac entry point.
