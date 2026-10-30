"""Run unit checks, legacy verifiers, and all preregistered BET gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


COMMANDS = [
    [sys.executable, "-m", "pytest"],
    [sys.executable, "verify_theory.py"],
    [sys.executable, "verify_width.py"],
    [sys.executable, "verify_stitch.py"],
    [sys.executable, "verify_p1_close.py"],
    *[[sys.executable, f"experiments/bet{i}_{name}.py"] for i, name in [
        (1, "validity"), (2, "width"), (3, "kappa"), (4, "growth"), (5, "ab")
    ]],
]


def main() -> int:
    rows = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for command in COMMANDS:
        print("RUN", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=env)
        rows.append({"command": command, "returncode": completed.returncode})
    passed = all(row["returncode"] == 0 for row in rows)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "run_all.json").write_text(
        json.dumps({"pass": passed, "commands": rows}, indent=2) + "\n"
    )
    print("ALL", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
