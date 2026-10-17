# Screenshot and video checklist

Only capture evidence after running the relevant steps. Do not use mock screenshots.

## Screenshots
1. Dataset file and metadata showing source, row count, and target mapping.
2. Class balance chart, feature histograms, and correlation heatmap.
3. Model comparison and cross-validation metrics from training output.
4. MLflow UI with both model runs and artifacts.
5. Successful `pytest -q` and `ruff check .`.
6. Successful GitHub Actions run.
7. Docker image build and running container.
8. Swagger UI `/docs` and a successful `/predict` response using synthetic example input.
9. Kubernetes pods and service status, if using Kubernetes.
10. `/metrics` output or monitoring dashboard.

## Short video outline (3–5 minutes)
1. Show repository structure and README.
2. Run data ingestion and EDA.
3. Run training and open MLflow UI.
4. Run tests and show GitHub Actions.
5. Start API and send a sample prediction.
6. Show Docker and, if implemented, Kubernetes.
7. Show logs/metrics and summarize limitations.

Do not show real patient information, credentials, tokens, or secrets.
