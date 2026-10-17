import numpy as np
import pandas as pd

from src.config import FEATURES
from src.pipeline import build_model


def sample_frame(n=40):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "age": rng.integers(30, 75, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n), "trestbps": rng.integers(100, 180, n),
        "chol": rng.integers(150, 320, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(90, 190, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.random(n) * 4,
        "slope": rng.integers(0, 3, n), "ca": rng.integers(0, 4, n),
        "thal": rng.integers(0, 4, n),
    })
    target = pd.Series([0, 1] * (n // 2))
    return frame[FEATURES], target


def test_both_models_fit_and_predict():
    X, y = sample_frame()
    for name in ("logistic_regression", "random_forest"):
        model = build_model(name)
        model.fit(X, y)
        prediction = model.predict(X.iloc[:3])
        probability = model.predict_proba(X.iloc[:3])
        assert len(prediction) == 3
        assert probability.shape == (3, 2)
        assert set(prediction).issubset({0, 1})
