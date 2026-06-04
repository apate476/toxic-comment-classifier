"""CPU profiling script for toxic comment model training."""

from __future__ import annotations

import cProfile
import pstats
from pathlib import Path

from toxic_comment_classifier.train_model import main

PROFILE_DIR = Path("reports/profiling")
PROFILE_OUTPUT = PROFILE_DIR / "training_cpu_profile.prof"
PROFILE_TEXT = PROFILE_DIR / "training_cpu_profile.txt"


def run_cpu_profile() -> None:
    """Profile the training pipeline using cProfile."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    profiler.dump_stats(PROFILE_OUTPUT)

    with PROFILE_TEXT.open("w", encoding="utf-8") as file:
        stats = pstats.Stats(profiler, stream=file)
        stats.sort_stats("cumulative")
        stats.print_stats(30)

    print(f"CPU profile saved to {PROFILE_OUTPUT}")
    print(f"Readable profile saved to {PROFILE_TEXT}")


if __name__ == "__main__":
    run_cpu_profile()
