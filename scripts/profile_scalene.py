"""Framework profiling for toxic comment model training using Scalene.

Scalene is preferred over cProfile for this sklearn pipeline because it
attributes time spent inside C extensions (TF-IDF vectorization, BLAS calls
inside Logistic Regression) to the originating Python line, where cProfile
reports the time only against the C wrapper. It also reports CPU, system,
and memory usage on the same line-by-line view.

Run from the project root:

    python scripts/profile_scalene.py

The script shells out to `scalene` rather than importing it, because
Scalene must load before the target module to instrument C extensions
correctly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROFILE_DIR = Path("reports/profiling")
HTML_REPORT = PROFILE_DIR / "training_scalene_profile.html"


def run_scalene_profile() -> None:
    """Profile `toxic_comment_classifier.train_model` with Scalene."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "scalene",
        "--html",
        "--outfile",
        str(HTML_REPORT),
        "--cpu",
        "--memory",
        "--reduced-profile",
        "-m",
        "toxic_comment_classifier.train_model",
    ]

    print(f"Running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)

    if completed.returncode != 0:
        print(
            f"Scalene exited with code {completed.returncode}. Confirm `pip install scalene` succeeded.",
            file=sys.stderr,
        )
        sys.exit(completed.returncode)

    print(f"Scalene HTML profile saved to {HTML_REPORT}")
    print("Open it in a browser to inspect line-by-line CPU + memory.")


if __name__ == "__main__":
    run_scalene_profile()
