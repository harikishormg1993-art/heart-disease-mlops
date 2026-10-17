# Data dictionary

The processed Cleveland Heart Disease dataset is commonly distributed with 13 input columns and one target column.

| Feature | General meaning | Notes |
|---|---|---|
| age | Age in years | Numeric |
| sex | Dataset sex encoding | 0/1; check source documentation |
| cp | Chest-pain category | Encoded category |
| trestbps | Resting blood pressure | Numeric |
| chol | Serum cholesterol | Numeric |
| fbs | Fasting blood sugar indicator | 0/1 |
| restecg | Resting ECG category | Encoded category |
| thalach | Maximum heart rate achieved | Numeric |
| exang | Exercise-induced angina | 0/1 |
| oldpeak | ST depression induced by exercise relative to rest | Numeric |
| slope | Slope of peak exercise ST segment | Encoded category |
| ca | Number of major vessels | Missing values can occur |
| thal | Source-encoded thalassemia-related category | Encoding may vary |
| target | Presence of heart disease | Original 0 -> 0; original 1–4 -> 1 |

The current baseline uses median imputation and scaling in one sklearn pipeline. A more advanced analysis could treat encoded categorical columns as categorical and use one-hot encoding. Document the actual source URL in `data/processed/metadata.json` and verify schema and target counts after download.
