# End-to-End MLOps for Heart Disease Classification

**Course:** AIMLCZG523 — Machine Learning Operations  
**Assignment:** 1  
**Student:** [Your name and ID]  
**Repository:** [Your GitHub URL]  
**Date:** [Submission date]

Replace bracketed text with your own implementation details. Insert only metrics and screenshots produced by your run.

## 1. Abstract and objective
This project implements an end-to-end MLOps workflow for an educational binary classification task using a processed Cleveland Heart Disease dataset. It covers data acquisition, EDA, preprocessing, model comparison, experiment tracking, reproducible packaging, automated tests, CI/CD, containerization, deployment, and monitoring. It is not a medical device and must not be used for diagnosis or treatment.

## 2. Dataset acquisition and EDA
Describe the exact source URL recorded in `data/processed/metadata.json`, retrieval date, row count, columns, missing values, target mapping, and data-quality checks. Explain original target 0 maps to class 0 and positive target values map to class 1. Insert EDA figures and interpret your own observations without implying causality.

## 3. Feature engineering and modelling
Explain feature selection, target exclusion, imputation, scaling, split strategy, random seed, and model hyperparameters. Describe Logistic Regression and Random Forest, stratified five-fold cross-validation, and holdout evaluation.

| Model | CV accuracy mean | CV precision mean | CV recall mean | CV ROC-AUC mean | Holdout ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | [observed value] | [observed value] | [observed value] | [observed value] | [observed value] |
| Random Forest | [observed value] | [observed value] | [observed value] | [observed value] | [observed value] |

Copy actual values from `artifacts/training_summary.json`. Discuss variability and why ROC-AUC was used for comparison. For a health-related problem, discuss recall and false negatives.

## 4. Experiment tracking with MLflow
Describe the experiment name, logged parameters, metrics, model artifacts, confusion matrices, and ROC curves. Insert an MLflow UI screenshot and identify the selected run under the documented selection rule.

## 5. Reproducibility and packaging
Explain the sklearn pipeline, saved model, Python version, dependency versions, and how data/model artifacts can be regenerated from scripts.

## 6. Automated testing and CI/CD
Describe unit tests, API validation tests, Ruff linting, and GitHub Actions. Insert real screenshots of a successful workflow and any failure/fix demonstration.

## 7. API and containerization
Describe `/health`, `/predict`, and `/metrics`, request schema, response fields, error handling, and logging. Include real Docker build and local test evidence.

## 8. Deployment and monitoring
Describe your selected deployment: Minikube, Docker Desktop Kubernetes, or cloud. Explain Deployment, Service, probes, resources, and actual evidence. Describe request count, latency, and logs/metrics.

## 9. Limitations, ethics, and conclusion
Discuss dataset size, source variation, potential bias, encoded feature semantics, limited external validity, and absence of clinical validation. Model-estimated probabilities are not necessarily calibrated medical risk estimates.

## Appendix: Reproduction instructions
List tested operating system, Python version, commands, repository URL, deployment access instructions, and video location.
