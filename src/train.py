"""Train, compare and log classification models with MLflow."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.config import ARTIFACTS, FEATURES, MLRUNS, RANDOM_STATE
from src.data import load_dataset
from src.pipeline import build_model

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


def train_all() -> dict:
    df = load_dataset()
    X, y = df[FEATURES], df["target"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mlflow.set_tracking_uri(MLRUNS.resolve().as_uri())
    mlflow.set_experiment("heart-disease-classification")
    results = []

    for name in ("logistic_regression", "random_forest"):
        pipeline = build_model(name)
        with mlflow.start_run(run_name=name) as run:
            cv_results = cross_validate(
                pipeline, X_train, y_train, cv=cv, scoring=SCORING,
                return_train_score=False, n_jobs=1,
            )
            fitted = build_model(name)
            fitted.fit(X_train, y_train)
            predictions = fitted.predict(X_test)
            probabilities = fitted.predict_proba(X_test)[:, 1]
            metrics = {
                f"cv_{metric}_mean": float(np.mean(cv_results[f"test_{metric}"]))
                for metric in SCORING
            }
            metrics.update({
                f"cv_{metric}_std": float(np.std(cv_results[f"test_{metric}"], ddof=1))
                for metric in SCORING
            })
            metrics.update({
                "test_accuracy": float(accuracy_score(y_test, predictions)),
                "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
                "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
            })
            mlflow.log_params({
                "model": name, "random_state": RANDOM_STATE, "test_size": 0.2,
                "cv_folds": 5, "selection_metric": "cv_roc_auc_mean",
                "features": ",".join(FEATURES),
            })
            mlflow.log_metrics(metrics)

            cm_disp = ConfusionMatrixDisplay(
                confusion_matrix(y_test, predictions), display_labels=["No disease", "Disease"]
            )
            cm_disp.plot(values_format="d")
            plt.tight_layout()
            cm_path = ARTIFACTS / f"{name}_confusion_matrix.png"
            plt.savefig(cm_path, dpi=150)
            plt.close()
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")

            RocCurveDisplay.from_predictions(y_test, probabilities)
            plt.tight_layout()
            roc_path = ARTIFACTS / f"{name}_roc_curve.png"
            plt.savefig(roc_path, dpi=150)
            plt.close()
            mlflow.log_artifact(str(roc_path), artifact_path="evaluation")
            mlflow.sklearn.log_model(fitted, artifact_path="model")

            results.append({
                "model": name, "run_id": run.info.run_id,
                "metrics": metrics, "cv_roc_auc_mean": metrics["cv_roc_auc_mean"],
            })

    best = max(results, key=lambda item: item["cv_roc_auc_mean"])
    final_model = build_model(best["model"])
    final_model.fit(X, y)
    import joblib
    joblib.dump(final_model, ARTIFACTS / "model.joblib")
    summary = {
        "selection_rule": "highest mean stratified 5-fold CV ROC-AUC on training split",
        "selected_model": best["model"],
        "runs": results,
        "final_model_fit": "selected pipeline refit on all rows for serving",
        "warning": "Educational dataset only; not a clinically validated diagnostic system.",
    }
    (ARTIFACTS / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train_all()
