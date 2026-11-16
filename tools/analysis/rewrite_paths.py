"""Rewrite the moved analysis scripts to use the shared data root.

Second pass: handles paths that carry a sub-directory, e.g.
``Path(r"...\\IC_TERMINAL_TRAIN\\newset")`` becomes ``DATA / "newset"``.
Already-rewritten files are skipped because they no longer contain the literal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP = {"analysis_paths.py", "rewrite_paths.py"}

ROOT = r"C:\\Users\\DELL\\Desktop\\d-meter\\IC_TERMINAL_TRAIN"
PATTERN = re.compile(
    r'Path\(\s*r?["\']' + ROOT + r'((?:\\[^"\']+)?)["\']\s*\)')

IMPORT = "from analysis_paths import DATA\n"


def replacement(match: re.Match) -> str:
    suffix = match.group(1)
    if not suffix:
        return "DATA"
    parts = [part for part in suffix.split("\\") if part]
    return "DATA / " + " / ".join(json.dumps(part) for part in parts)


def main() -> None:
    changed, skipped = [], []
    for path in sorted(HERE.glob("*.py")):
        if path.name in SKIP:
            continue
        text = path.read_text(encoding="utf-8")
        if "IC_TERMINAL_TRAIN" not in text:
            continue
        original = text
        text = PATTERN.sub(replacement, text)
        if text == original:
            skipped.append(path.name)
            continue
        if "IC_TERMINAL_TRAIN" in text or IMPORT not in text:
            lines = text.splitlines(keepends=True)
            insert = 0
            for index, line in enumerate(lines[:40]):
                if line.startswith(("import ", "from ")):
                    insert = index + 1
            lines.insert(insert, IMPORT)
            text = "".join(lines)
        path.write_text(text, encoding="utf-8")
        changed.append(path.name)

    print(f"rewritten: {len(changed)}")
    for name in changed:
        print(f"  {name}")
    print(f"still containing the literal: {len(skipped)}")
    for name in skipped:
        print(f"  {name}")


if __name__ == "__main__":
    main()
