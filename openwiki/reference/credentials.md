---
type: reference
title: Credentials and Endpoints
description: Local credential handling, board identity and private-link endpoints.
status: active
tags: [credentials, endpoints, networking]
---

# Credentials and endpoints

**Never commit passwords/private SSH keys.** `config/credentials.json`, local
portable-link profiles and environment secrets are ignored. Use external local
profiles or `--ask-password`; only `portable-link.example.json` is publishable.
Historical plaintext password rows were removed before the planned GitHub import.

## Portable board

| Field | Current lab value |
|---|---|
| SSH addresses | management172.21.30.120, camera subnet192.168.1.20 |
| SSH user | sima |
| Password | external local credentials file or interactive prompt, NEVER this wiki |
| Hostname | modalix; .local/mDNS availability NOT verified |
| Active physical Ethernet | end0, holds BOTH IPv4 addresses (not two separate NICs) |
| Release root | /home/sima/demeter-vision |
| Historical launcher releases | /media/nvme/portable-release-* |

Current public SSH key SHA256 (hex):
`3e0408ab1e375e57f56cb898abb17d829ed8131d5a7c6af84f11f7bc6cc5eba0`.
Matches earlier replay evidence. Portable launcher uses the explicit pin, or normal
known_hosts with rejection of unknown keys; no silent acceptance of new identities.
Changing network addresses does not change device identity. No SSH keys installed
or board networking changed by the portability work.

## Camera and isolated switch

- Configured RTSP: `rtsp://192.168.1.10:5005/routecam`, H2641280x720/~30FPS, TCP.
- OPTIONS returned RTSP200 from board during read-only check. No media graph opened.
- Camera endpoint belongs in local `camera.rtsp` data, not application constants.
- Existing switch connects camera, board and Mac. No summit router/DHCP/internet
  needed for runtime. Carry existing camera power/PoE and Mac Ethernet adapter.
- Board address192.168.1.20 is persistent in both observed ifupdown and systemd-
  networkd settings.172.21.30.120 also appears in persistent20-end0.network.
  Multiple network managers/configs exist: do not rewrite them casually.
- Mac must have a compatible private Ethernet address (e.g.192.168.1.30/24 with
  no router/DNS for that private link). This is operator-configured link data,
  not hardcoded application addressing. Verify no duplicate/conflicting Wi-Fi route.

## Runtime endpoints

- Board SSH22; camera RTSP5005 on CAMERA, not the viewing Mac.
- Camera video to route-selected host UDP5010; coordinate metadata UDP5012.
- Browser HTTP127.0.0.1 ephemeral port on viewing computer, in session.json.
- Old SDK worker5005/lease5006 and focus HTTP8080 are not the selected browser API.
  Hardware decoder remains exclusive; do not run separate capture/inference graphs.

## Compiler / fixed station boundaries

Admin compiler (rebuild only, NOT runtime dependency):172.21.30.222, useradmin,
ModelSDK2.0.0 container, builds under C:/Users/Admin/Desktop/ICTerminalBuilds.
Credentials remain in external local config or ADMIN_SSH_PASSWORD, never this wiki.
The fixed AOI station and its board are separate; do not stage/restart them.

## Portability status

Owner reports Apple Silicon M3 Air/macOS27.0. Native arm64 host environment and
actual Mac/switch/board test are pending. Windows-selected r2 package is unchanged
and its Windows wheels/PowerShell launcher are NOT claimed Mac-ready. Runtime
camera processing/inference stays on SiMa; host video decoding/frame matching/
browser rendering and SSH lifecycle move to the Mac.
