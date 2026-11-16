"""Count semantics derived from detections, shared by host and board paths.

A model that emits "there are 9 pills" cannot be checked against a picture; one
that emits nine boxes can. Keeping the derivation in one module means the board
worker, the host detector and the release review all compute the same numbers.
"""

from __future__ import annotations

from typing import Any, Iterable

#: The two families and what a count means for each. Data, not branching code:
#: adding a family is a label change and a retrain, not a new code path.
STATE_CLASSES: dict[str, tuple[str, ...]] = {
    "pills": ("cavity_full", "cavity_empty"),
    "caps": ("cap_ok", "cap_hole"),
}


def summarise(detections: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Derive counts from detections rather than predicting a number.

    `expected` is the number of cavities the part has, when the caller knows it.
    A card with 9 cavity_full and 1 cavity_empty has one missing pill; without
    `expected` the missing count is not knowable, and saying so is better than
    reporting a number that looks like an answer.
    """
    by_class: dict[str, int] = {}
    for detection in detections:
        name = str(detection.get("class_name", ""))
        by_class[name] = by_class.get(name, 0) + 1

    summary: dict[str, Any] = {
        "count": sum(by_class.values()),
        "by_class": by_class,
    }
    for family, (present, absent) in STATE_CLASSES.items():
        if present not in by_class and absent not in by_class:
            continue
        ok = by_class.get(present, 0)
        defective = by_class.get(absent, 0)
        if family == "pills":
            summary[family] = {
                "pills_present": ok,
                "missing_pills": defective,
                "cavities": ok + defective,
            }
        else:
            summary[family] = {
                "caps_total": ok + defective,
                "caps_with_hole": defective,
            }
    return summary
