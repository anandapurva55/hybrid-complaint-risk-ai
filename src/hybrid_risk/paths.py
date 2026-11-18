"""Single source of truth for where things live, so no module depends on the working directory.

Layout (relative to the repository root, or /app inside the container):
    data/raw/        the original dataset
    data/processed/  train.csv, val.csv, test.csv (generated, git-ignored)
    data/samples/    bundled example inputs and outputs
    models/          trained artifacts used by the service
    results/         evaluation outputs and the monitoring baseline
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "Womens Clothing E-Commerce Reviews.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
SAMPLES = ROOT / "data" / "samples"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
