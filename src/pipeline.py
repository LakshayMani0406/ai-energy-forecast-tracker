#!/usr/bin/env python3
"""
pipeline.py — runs the full pipeline end-to-end locally.
Equivalent to what GitHub Actions runs monthly.

Usage:  python src/pipeline.py
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY   = sys.executable

steps = [
    ("📡 Fetching data",          [PY, str(ROOT / "src" / "ingest" / "eia.py")]),
    ("🔬 Training model",          [PY, str(ROOT / "src" / "forecast" / "train.py")]),
    ("🏷️  Evaluating & promoting", [PY, str(ROOT / "src" / "forecast" / "evaluate.py")]),
]

for label, cmd in steps:
    print(f"\n{label}...")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"❌ Failed at: {label}")
        sys.exit(1)

print("\n✅ Pipeline complete.")
print("   View MLflow UI:  mlflow ui --backend-store-uri mlruns/")
print("   View dashboard:  streamlit run src/dashboard/app.py")
