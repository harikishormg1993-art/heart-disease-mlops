"""Exploratory data analysis outputs."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FEATURES, FIGURES, ROOT
from src.data import load_dataset


def run_eda() -> dict:
    df = load_dataset()
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().astype(int).to_dict(),
        "target_distribution": df["target"].value_counts().sort_index().astype(int).to_dict(),
        "descriptive_statistics": df[FEATURES].describe().round(3).to_dict(),
    }

    df["target"].value_counts().sort_index().plot(kind="bar", rot=0)
    plt.title("Target class balance")
    plt.xlabel("Target (0 = absent, 1 = present)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES / "class_balance.png", dpi=160)
    plt.close()

    df[FEATURES].hist(figsize=(14, 10), bins=18)
    plt.suptitle("Feature distributions")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_histograms.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 9))
    sns.heatmap(df[FEATURES + ["target"]].corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES / "correlation_heatmap.png", dpi=160)
    plt.close()

    path = ROOT / "reports" / "eda_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"EDA figures saved to {FIGURES}")
    print(f"Summary saved to {path}")
    return summary


if __name__ == "__main__":
    run_eda()
