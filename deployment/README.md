# Clone-contained runtime

`portable-v1/` is the complete frozen portable runtime, model, SiMa native library,
reference ONNX and Windows/Mac offline host wheels. All individual files are below
GitHub's100MiB limit: commit these ordinary Git files, NOT an external ZIP or LFS
pointer dependency. `verify.py` checks55 files in the original frozen payload.
No sibling IC_TERMINAL_TRAIN/MODEL/fixed-station directories are runtime dependencies.

From repo root on Mac M3, using native arm64 CPython3.11:

```sh
python3.11 setup-host.py
./local/host-env/bin/python -B run-portable.py --board-host 192.168.1.20
```

Setup installs bundled wheels OFFLINE into `local/host-env` INSIDE THIS CLONE.
Run prompts for SSH password; no secrets committed. Outputs go to `local/runs`
inside clone. Board address is operator input, receiver auto-selected from route.
Use isolated switch/camera/board/Mac Ethernet as documented in package README.
A manual compatible Mac Ethernet address is still necessary; app doesn't change it.

Python3.11 interpreter/macOS/Ethernet hardware and existing SiMa OS/SDK are platform
prerequisites, not repository dependencies. No GPU/Windows machine/Internet/PyPI,
external model/wheel folder or artifact download is needed at runtime.
Actual Mac installation remains to be validated. Package Windows smoke/owned stop
passed without PCB; not fresh detection quality evidence.20..300s bound retained.

Historical evidence files inside payload may mention old absolute paths; they are
provenance, not read at launch. Use ROOT launchers above, not archived Windows
production selectors or development/training analysis scripts.
