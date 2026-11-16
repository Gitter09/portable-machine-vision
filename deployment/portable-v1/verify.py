"""Offline integrity verification. Does not connect to the board."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
m=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
if actual != set(m['files'])|{'MANIFEST.json'}:raise SystemExit('FAIL: missing/extra package files')
for name,sha in m['files'].items():
 p=root/name
 if p.is_symlink() or not p.resolve().is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise SystemExit('FAIL: '+name)
print('PASS: '+str(len(m['files']))+' files verified; package '+m['package_id'])
