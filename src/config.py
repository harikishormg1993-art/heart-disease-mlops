from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
MLRUNS = ROOT / "mlruns"

# Public mirror of the processed Cleveland dataset. Verify source/schema after download.
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/processed.cleveland.data"
)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
RANDOM_STATE = 42

for path in (DATA_RAW, DATA_PROCESSED, ARTIFACTS, FIGURES, MLRUNS):
    path.mkdir(parents=True, exist_ok=True)
