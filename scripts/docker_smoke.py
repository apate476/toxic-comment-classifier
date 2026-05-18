"""One-shot smoke test executed inside the Docker container.

Verifies package import, config path resolution, bind-mount visibility,
core dependency availability, and that DVC-pulled raw data is reachable
from inside the container.

Run via:
    docker compose run --rm --entrypoint python toxic_comment_classifier \\
        scripts/docker_smoke.py
"""

import os
import sys


def main() -> int:
    print("=== Python ===")
    print(sys.version)

    print("\n=== Package import ===")
    import toxic_comment_classifier as tcc

    print("toxic_comment_classifier OK ->", tcc.__file__)

    print("\n=== Config / paths ===")
    from toxic_comment_classifier.config import (
        PROJECT_ROOT,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
    )

    print("PROJECT_ROOT      :", PROJECT_ROOT)
    print("RAW_DATA_DIR      :", RAW_DATA_DIR, "exists:", RAW_DATA_DIR.exists())
    print(
        "PROCESSED_DATA_DIR:",
        PROCESSED_DATA_DIR,
        "exists:",
        PROCESSED_DATA_DIR.exists(),
    )
    print("MODELS_DIR        :", MODELS_DIR, "exists:", MODELS_DIR.exists())

    print("\n=== Bind mounts (cwd /app) ===")
    for p in ["data/raw", "data/processed", "models", "mlruns", "configs", "reports"]:
        full = os.path.join("/app", p)
        n = len(os.listdir(full)) if os.path.isdir(full) else "MISSING"
        print(f"  /app/{p:18s} entries={n}")

    print("\n=== Heavy deps ===")
    import torch
    import transformers
    import mlflow
    import pandas
    import sklearn

    print("torch       :", torch.__version__, "| cuda:", torch.cuda.is_available())
    print("transformers:", transformers.__version__)
    print("mlflow      :", mlflow.__version__)
    print("pandas      :", pandas.__version__)
    print("sklearn     :", sklearn.__version__)

    print("\n=== Data loader sanity ===")
    from toxic_comment_classifier.data import load_raw

    for filename in ("train.csv", "test.csv", "test_labels.csv"):
        df = load_raw(filename)
        print(f"  {filename:24s} rows={len(df):>7d} cols={len(df.columns)}")

    print("\nSMOKE TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
