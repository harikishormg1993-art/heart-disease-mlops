"""Data acquisition, normalization, and dataset validation."""
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from src.config import DATASET_URL, DATA_PROCESSED, DATA_RAW, FEATURES

COLUMNS = FEATURES + ["target"]


def download_file(url: str, destination: Path, timeout: int = 30) -> None:
    """Download a URL to destination, raising a useful error on failure."""
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = response.read()
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not download dataset from {url}: {exc}") from exc
    if not payload:
        raise RuntimeError(f"Downloaded an empty file from {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a compatible processed Cleveland CSV and normalize target to binary."""
    df = pd.read_csv(path, na_values=["?", "NA", ""])
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"num": "target", "condition": "target", "output": "target"}
    df = df.rename(columns=aliases)

    if not set(FEATURES).issubset(df.columns):
        df = pd.read_csv(path, header=None, na_values=["?", "NA", ""])
        if df.shape[1] == 14:
            df.columns = COLUMNS
        elif df.shape[1] == 15:
            df = df.iloc[:, 1:]
            df.columns = COLUMNS
        else:
            raise ValueError(
                f"Expected 14 columns (13 features + target), got {df.shape[1]}"
            )

    if "id" in df.columns:
        df = df.drop(columns=["id"])
    target_col = "target" if "target" in df.columns else "num"
    if target_col not in df.columns:
        raise ValueError("Could not identify target column.")
    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    df = df[FEATURES + [target_col]].rename(columns={target_col: "target"})
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["target"])
    # Cleveland source target: 0 = absence, 1–4 = presence.
    df["target"] = (df["target"] > 0).astype(int)
    if len(df) < 50:
        raise ValueError(f"Dataset unexpectedly small ({len(df)} rows); verify source.")
    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError("Target is not binary after normalization.")
    return df


def prepare_dataset(source_url: str = DATASET_URL) -> pd.DataFrame:
    raw_path = DATA_RAW / "heart_disease_source.csv"
    download_file(source_url, raw_path)
    df = read_dataset(raw_path)
    output = DATA_PROCESSED / "heart_disease.csv"
    df.to_csv(output, index=False)
    metadata = {
        "source_url": source_url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "target_mapping": "original 0 -> 0; original values >0 -> 1",
        "missing_values_by_column": df.isna().sum().astype(int).to_dict(),
    }
    (DATA_PROCESSED / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return df


def load_dataset() -> pd.DataFrame:
    path = DATA_PROCESSED / "heart_disease.csv"
    return pd.read_csv(path) if path.exists() else prepare_dataset()


if __name__ == "__main__":
    data = prepare_dataset()
    print(f"Saved {DATA_PROCESSED / 'heart_disease.csv'}")
    print(f"Rows: {len(data)}")
    print("Target counts:\n" + data["target"].value_counts().sort_index().to_string())
