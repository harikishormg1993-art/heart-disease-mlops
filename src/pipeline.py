"""Shared preprocessing and model construction."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import FEATURES, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Baseline treats numeric source encodings as numeric; document this limitation.
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(model_name: str, random_state: int = RANDOM_STATE):
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
