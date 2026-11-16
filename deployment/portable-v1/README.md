# Network-portable package: Mac M3 and isolated switch

This is a PORTABLE CANDIDATE beside the accepted Windows production r2. Mac execution
is NOT yet tested. Owner reports M3 Air/macOS27.0. Use native arm64 CPython3.11;
macOS host never loads the Linux AArch64 .so: it is uploaded to SiMa only.

## What to carry

SiMa board, existing camera and camera power/PoE, existing unmanaged switch,
Ethernet cables, USB-C Ethernet adapter for Mac, board power, this FULL package
ZIP and a native arm64 CPython3.11 installer/interpreter installed before travel.
No Windows PC/GPU, compiler host, summit router or internet required at runtime.
Host decodes/matches camera video and runs browser; all model work stays on SiMa.

## Private Ethernet setup (operator action, not changed by software)

Connect board + camera + Mac to the existing switch. Board's sole end0 currently
has192.168.1.20/24 AND172.21.30.120/24. Camera192.168.1.10 is reachable; RTSP OPTIONS
returned200. Existing settings were read only, not modified. Multiple persistent
network configs/managers exist; do not rewrite board networking during travel.

On Mac System Settings → Network → the ETHERNET adapter → Details → TCP/IP:
choose manual IPv4, e.g.192.168.1.30, subnet255.255.255.0; leave router/DNS empty
for this private link. Ensure no other device uses that address. These are local
configuration examples, NOT code constants. If Wi-Fi uses the same subnet, disable
Wi-Fi during the private test or resolve the route conflict first. App rejects a
Mac route identified as Wi-Fi. No need to alter board/camera gateway for same-subnet
local communication. Verify ping192.168.1.20 before attempting launch.

## Offline install (Mac Terminal)

After extracting package to a fresh directory, cd into it:

```sh
python3.11 -c 'import platform; print(platform.machine())'  # must print arm64
python3.11 -B verify.py
python3.11 setup-host.py --package . --environment "$HOME/.venvs/portable-vision"
```

Setup creates a NEW environment and installs nine pinned native Mac-compatible
wheels OFFLINE, tests imports/H264 decoder and pip consistency. Never modifies
Ethernet, system Python or board SDK. Downloaded macOS14+arm64 wheels are selected
for M3; actual installation on reported macOS27 remains unverified until run.
Windows wheels are also bundled, chosen on Windows. Python itself is not bundled.

## Private config and launch

Copy config/portable-link.example.json OUTSIDE this package, e.g.
`$HOME/.config/portable-vision/link.json` (create its parent first). Set board.host
to192.168.1.20 (or your chosen reachable board hostname/IP). camera.rtsp is local
configuration. The public SSH-key pin matches this actual board's prior evidence;
wrong identity aborts before password authentication. Never commit passwords.
Use --ask-password (prompt) or configure an existing SSH key via --ssh-key.
No keys/passwords installed or changed automatically.

```sh
"$HOME/.venvs/portable-vision/bin/python" -B run-portable.py \
  --package . --config "$HOME/.config/portable-vision/link.json" \
  --ask-password --seconds 300
```

Route lookup auto-selects the Mac Ethernet return address, then SSH binds to that
source address. Board sends video5010 and metadata5012 to that address. No fixed
workstation IP. Optional --board-host/--receiver-host overrides must be valid/local
private IPv4. Permit Python local-network/incoming access if macOS prompts;
do not disable firewall globally. Browser HTTP is loopback only; URL printed and
saved in session.json. Local browser on Mac is required, no external frontend yet.

Live pane is camera; separate canvas is EXACT pixel-hash matched inspected frame.
Unmatched/UNKNOWN/stale/disconnected never means pass. Native packing/model/tracking
parameters stay those of accepted r2. No claimed detector accuracy improvement.

## Stop and recovery

Ctrl+C in Terminal requests cleanup and signals ONLY your authenticated user's
exact owned worker script AND output directory on board. Worker handles SIGINT/
SIGTERM and releases decoder. If SSH/link fails, immediate remote stop cannot be
guaranteed: original20..300s job bound plus timeout still applies. Wait/check board
processes before restart; never kill by broad name. No unattended continuous
supervisor/autostart/reconnect or power-loss recovery qualification added.

## Tests completed here / what is still pending

Windows route-selected smoke:747/747 matched; ready-state owned stop reported
worker/session stopped and host exit0. Immediate restart completed,570 decoded
frames/~30.07FPS,479 metadata/468 matched AT ARRIVAL,0 RTP gaps/host errors.
All these recent frames were UNKNOWN/no PCB: NOT model speed/quality evidence.
Unmatched-at-arrival can match later; overlays require actual match, never guess.

Actual M3 install, Ethernet route, macOS firewall and browser performance still
need operator testing. PCB-present quality/inference parity and a longer soak
must follow. Do not call the portable candidate production-qualified on Mac yet.
Current Windows production selection and earlier immutable fallbacks remain intact.

## GitHub / secrets / artifacts

Repository preparation is separate. Local credentials/private keys/envs/artifacts
are ignored and wiki plaintext passwords removed before import. ZIP includes model,
native binary and offline wheels which must be carried/downloaded as an artifact;
Git clone alone is not a complete runtime. Git initialization/meaningful commits/
push remain pending until this portability package is prepared and reviewed.
