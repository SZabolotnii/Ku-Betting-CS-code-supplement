"""Run the supplementary checks without altering the original BET protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
COMMANDS = [
    [sys.executable, "experiments/verify_quadratic_family.py"],
    [sys.executable, "experiments/rev_misspec.py"],
    [sys.executable, "experiments/rev_strategy.py"],
]


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    rows = []
    for command in COMMANDS:
        print("RUN", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=env)
        rows.append({"command": command, "returncode": completed.returncode})
    passed = all(row["returncode"] == 0 for row in rows)
    (ROOT / "results" / "run_revision.json").write_text(
        json.dumps({"pass": passed, "commands": rows}, indent=2) + "\n"
    )
    print("REVISION", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
