"""Score recorded model output and constant-strategy controls with the same criteria."""

import argparse
import json
from pathlib import Path

from context_blind_cases import cases as blind_cases
from context_cases import cases, evaluate


def score(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    actual = []
    for row in rows:
        case_source = blind_cases if row.get("suite") == "blind" else cases
        case = next(x for x in case_source(row["run"] - 1) if x["id"] == row["case"])
        result = row.get("analysis")
        accepted = bool(result and evaluate(case, result))
        if bool(row["accepted"]) != accepted:
            raise ValueError("semantic score differs from recorded score")
        actual.append((case, accepted, row))
    controls = {}
    for name, signals in [
        ("always_literal", []),
        (
            "always_all_signals",
            ["humor", "irony", "sarcasm", "wordplay", "provocation", "ambiguity", "topic_change"],
        ),
    ]:
        controls[name] = sum(
            evaluate(case, {"intent": "request", "signals": signals, "boundary_requests": []})
            for case, _, _ in actual
        )
    return {
        "total": len(rows),
        "accepted": sum(ok for _, ok, _ in actual),
        "groups": {
            group: {
                "total": sum(row.get("suite", "development") == group for _, _, row in actual),
                "accepted": sum(
                    ok and row.get("suite", "development") == group for _, ok, row in actual
                ),
            }
            for group in sorted({row.get("suite", "development") for _, _, row in actual})
        },
        "constant_controls": controls,
        "failures": [{"run": row["run"], "case": row["case"]} for _, ok, row in actual if not ok],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.results), ensure_ascii=False, indent=2))
