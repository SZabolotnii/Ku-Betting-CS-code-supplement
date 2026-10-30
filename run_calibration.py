"""Run the independent-calibration (CAL-1..CAL-4) study after the BET and REV checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
COMMANDS = [
    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
     "tests/test_calibration.py"],
    [sys.executable, "experiments/verify_calibration_width.py"],
    [sys.executable, "experiments/cal_transfer.py", "cal13"],
    [sys.executable, "experiments/cal_penalty_audit.py"],
    [sys.executable, "experiments/cal_transfer.py", "cal2"],
    [sys.executable, "experiments/make_calibration_figures.py"],
    [sys.executable, "experiments/cal_transfer.py", "cal4"],
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
    (ROOT / "results" / "run_calibration.json").write_text(
        json.dumps({"pass": passed, "commands": rows}, indent=2) + "\n"
    )
    print("CALIBRATION", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
