"""FastAPI serving layer with request logging and Prometheus metrics."""
import logging
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.config import ARTIFACTS, FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart_disease_api")
REQUESTS = Counter("api_requests_total", "Total API requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["endpoint"])

app = FastAPI(
    title="Heart Disease Educational Classifier",
    version="1.0.0",
    description="Academic demonstration only. Not for clinical use.",
)


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(ge=0, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-10, le=20)
    slope: int = Field(ge=0, le=2)
    ca: float = Field(ge=0, le=4)
    thal: float = Field(ge=0, le=7)


_model = None


def get_model():
    global _model
    if _model is None:
        path = ARTIFACTS / "model.joblib"
        if not path.exists():
            raise RuntimeError(
                f"Model not found at {path}. Run `python -m src.data` and `python -m src.train` first."
            )
        _model = joblib.load(path)
    return _model


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    status = "500"
    endpoint = request.url.path
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception:
        logger.exception("Unhandled request exception")
        raise
    finally:
        elapsed = time.perf_counter() - start
        REQUESTS.labels(request.method, endpoint, status).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        logger.info(
            "request method=%s path=%s status=%s latency_seconds=%.6f",
            request.method, endpoint, status, elapsed,
        )


@app.get("/health")
def health():
    return {
        "status": "ok" if (ARTIFACTS / "model.joblib").exists() else "model_missing",
        "model_loaded": _model is not None,
    }


@app.get("/")
def root():
    return {"service": "heart-disease-mlops", "docs": "/docs", "warning": "Academic use only."}


@app.post("/predict")
def predict(payload: PatientFeatures):
    try:
        model = get_model()
        frame = pd.DataFrame([payload.model_dump()], columns=FEATURES)
        prediction = int(model.predict(frame)[0])
        probabilities = model.predict_proba(frame)[0]
        return {
            "prediction": prediction,
            "label": "disease_present" if prediction == 1 else "disease_absent",
            "model_estimated_probability": {
                "disease_absent": float(probabilities[0]),
                "disease_present": float(probabilities[1]),
            },
            "disclaimer": "Educational output only; not medical advice or a diagnosis.",
        }
    except RuntimeError as exc:
        logger.error("Prediction unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed; inspect server logs.") from exc


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
