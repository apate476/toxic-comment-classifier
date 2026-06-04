"""Memory profiling script for toxic comment model training."""

from __future__ import annotations

from pathlib import Path

from memory_profiler import memory_usage

from toxic_comment_classifier.train_model import main

PROFILE_DIR = Path("reports/profiling")
MEMORY_REPORT = PROFILE_DIR / "training_memory_profile.txt"


def run_memory_profile() -> None:
    """Profile peak memory usage during training."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    memory_values = memory_usage((main, (), {}), interval=1.0)

    peak_memory = max(memory_values)
    starting_memory = memory_values[0]
    memory_increase = peak_memory - starting_memory

    report = f"""Training Memory Profile

Starting memory usage: {starting_memory:.2f} MiB
Peak memory usage: {peak_memory:.2f} MiB
Memory increase: {memory_increase:.2f} MiB
Number of memory samples: {len(memory_values)}
"""

    MEMORY_REPORT.write_text(report, encoding="utf-8")

    print(report)
    print(f"Memory profile saved to {MEMORY_REPORT}")


if __name__ == "__main__":
    run_memory_profile()
