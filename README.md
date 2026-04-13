# HealthSpark — Distributed Healthcare Claims Analytics & ML Pipeline

[![CI](https://github.com/Deepak-Lingala/HealthSpark-Claims-Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Deepak-Lingala/HealthSpark-Claims-Analytics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PySpark](https://img.shields.io/badge/PySpark-3.5.3-orange?logo=apachespark)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU-red?logo=xgboost)
![SHAP](https://img.shields.io/badge/SHAP-Interpretability-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green)

End-to-end PySpark pipeline that ingests 500,000 synthetic healthcare claims, engineers 36 clinical features, trains and compares three models (Logistic Regression, Random Forest, GPU XGBoost) for 30-day readmission prediction, explains predictions with SHAP, selects an operating threshold from the Precision-Recall curve, and serves the winning model through FastAPI. Runs end-to-end on Google Colab Pro (A100) in **~5 minutes**.

All data is 100% synthetic and HIPAA-safe.

---

## Headline Result

On a Colab Pro A100 run with 500K claims (seed 42), **Logistic Regression wins on AUC-PR** against a tuned Random Forest and GPU XGBoost — by a 0.003 margin. That is the interesting finding: when a linear baseline ties a GPU-tuned gradient booster, the feature engineering has already absorbed the non-linearities, and the correct production call is the simpler model.

| Metric | LR Baseline | RF Tuned | XGBoost GPU |
|--------|-------------|----------|-------------|
| AUC-ROC | 0.7261 | 0.7258 | 0.7237 |
| AUC-PR | **0.3658** | 0.3625 | 0.3610 |
| F1 Score | 0.7201 | 0.7248 | **0.7345** |

**Bayes-optimal AUC-ROC on this synthetic data is ~0.745** (the label is generated from a logistic model over real clinical predictors plus noise). Real CMS HRRP readmission models typically land at 0.65–0.72, so these numbers are in the right clinical range.

See [Results Discussion](#results-discussion) below for the full interpretation.

---

## What This Project Demonstrates

- **Distributed ETL** — PySpark ingestion with explicit `StructType` schemas, broadcast joins, window functions (rolling 90-day costs, provider rankings), DataFrame API + Spark SQL shown side-by-side, strategic `.cache()` / `.unpersist()`
- **Feature engineering at scale** — 20 numeric + 4 one-hot categorical features including comorbidity indices, provider denial rates, patient-level aggregates, and temporal lags
- **Class imbalance handled correctly** — `weightCol` in MLlib (inverse class frequency) and `scale_pos_weight` in XGBoost, no SMOTE
- **Three-model comparison** — Logistic Regression (ElasticNet), Random Forest (CrossValidator, 4×3 grid), GPU XGBoost — trained on the same feature set for a fair comparison
- **Clinical interpretability** — SHAP global summary (bar + beeswarm) and per-patient waterfall plots
- **Threshold selection from the PR curve** — not "pick the best AUC", but "at what threshold do precision and recall match what the business can staff?"
- **Production serving** — FastAPI inference server loading the fitted `PipelineModel` for zero train/serve skew
- **CI/CD** — GitHub Actions runs pytest (11 integration tests against a real `SparkSession`) and ruff on every push
- **Reproducible in 5 minutes** — Google Colab Pro notebook clones, installs, generates data, runs the full pipeline, and produces all visualizations

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐
│  Synthetic   │    │  PySpark     │    │  Parquet Data Lake       │
│  Data Gen    │───▶│  Ingestion   │───▶│  (columnar, partitioned) │
│  (500K rows) │    │  + QA Checks │    │                          │
└──────────────┘    └──────────────┘    └────────────┬─────────────┘
                                                     │
                                                     ▼
                    ┌──────────────────────────────────────────────┐
                    │          PySpark Transformations             │
                    │  • Window functions (rolling 90d, ranks)     │
                    │  • Broadcast joins (patients → claims)       │
                    │  • DataFrame API + Spark SQL in parallel     │
                    └────────────────────┬─────────────────────────┘
                                         ▼
                    ┌──────────────────────────────────────────────┐
                    │          Feature Engineering                 │
                    │  • 36 features (numeric + one-hot)           │
                    │  • Class weights for imbalance handling      │
                    └────────────────────┬─────────────────────────┘
                                         ▼
                    ┌──────────────────────────────────────────────┐
                    │      Multi-Model Training & Comparison       │
                    │                                              │
                    │   MLlib LR  │  MLlib RF  │  XGBoost (GPU)    │
                    │  baseline   │  4×3 CV    │  scale_pos_weight │
                    │                                              │
                    │  AUC-ROC / AUC-PR / F1 / Precision / Recall  │
                    │  + SHAP values + PR curve + threshold table  │
                    └────────────────────┬─────────────────────────┘
                                         ▼
                    ┌──────────────────────────────────────────────┐
                    │             FastAPI Serving Layer            │
                    │  POST /predict → probability + risk tier     │
                    │  GET  /stats   → model comparison + FI       │
                    │  GET  /health  → service health              │
                    └──────────────────────────────────────────────┘

Infrastructure: Docker Compose (Spark master + 2 workers + API)
CI/CD:          GitHub Actions (pytest + ruff on every push)
```

---

## Quick Start

### Google Colab Pro (recommended — GPU-accelerated)

1. Open https://colab.research.google.com
2. **File → Open notebook → GitHub**, paste: `Deepak-Lingala/HealthSpark-Claims-Analytics`
3. Select `healthspark_colab.ipynb`
4. **Runtime → Change runtime type → T4 or A100 GPU**
5. **Runtime → Run all** (~5 minutes on A100, ~8 minutes on T4)

The notebook clones this repo, installs dependencies, generates data, runs the full pipeline, and renders all visualizations inline.

### Local (Docker)

```bash
git clone https://github.com/Deepak-Lingala/HealthSpark-Claims-Analytics.git
cd HealthSpark-Claims-Analytics
make docker-up
make run-pipeline
make run-api        # FastAPI on http://localhost:8000/docs
```

### Local (venv)

```bash
python -m venv .venv && source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate                             # Windows
pip install -r requirements.txt
python -m src.data_generation.generate_claims
python -m src.pipeline.ml_pipeline
pytest tests/ -v
```

---

## Dataset

| Table | Records | Description |
|-------|---------|-------------|
| `claims.csv` | 500,000 | Healthcare insurance claims with ICD-10, CPT, costs, outcomes |
| `patients.csv` | 50,000 | Patient demographics (age, gender, state, insurance type) |

Distributions calibrated to real-world benchmarks:

- **30-day readmission rate**: ~15% (CMS HRRP benchmark), generated from a logistic model over comorbidity count, age, high-risk diagnosis flag, length of stay, and facility type
- **Denial rate**: ~12% overall, varying by payer (Self-Pay 22%, Medicare 8%)
- **Age distribution**: Beta-skewed toward 55–75 (high-utilization population)
- **Top 20 ICD-10 codes** by prevalence (diabetes, hypertension, COPD, CHF, pneumonia, CKD, etc.)
- **Claim amounts**: log-normal by facility type (Inpatient ~$8K median, Outpatient ~$400 median)

---

## Model Comparison

Three models trained on the same feature set, with class weights handling the ~15% positive class rate:

| Model | Role | Tuning |
|-------|------|--------|
| Logistic Regression | Interpretable linear baseline | ElasticNet regularization |
| Random Forest (MLlib) | Distributed non-linear ensemble | CrossValidator 4 combos × 3 folds |
| XGBoost (GPU) | Gradient-boosted comparison | `scale_pos_weight`, `device='cuda'` |

### Evaluation (Colab Pro A100, 500K claims, seed 42)

| Metric | LR Baseline | RF Tuned | XGBoost GPU |
|--------|-------------|----------|-------------|
| AUC-ROC | 0.7261 | 0.7258 | 0.7237 |
| AUC-PR  | **0.3658** | 0.3625 | 0.3610 |
| F1 Score | 0.7201 | 0.7248 | **0.7345** |

XGBoost GPU training time: **3.9 seconds** on A100 (CUDA). Full metrics are saved to `data/models/model_results.json` at the end of every run.

> **Why AUC-PR matters more than AUC-ROC here**: With a ~15% positive class, AUC-ROC is inflated by the abundance of true negatives. AUC-PR (average precision) measures what the model does on the positive class specifically, which is what clinical deployment cares about.

### Results Discussion

The synthetic readmission label is generated from a logistic model over real clinical predictors (comorbidity count, age, high-risk diagnosis, length of stay, facility type) plus Gaussian noise on the logit. This gives a **Bayes-optimal AUC-ROC of ~0.745** — the ceiling any model can achieve on this data. Published CMS HRRP readmission models typically land between 0.65–0.72, so the dataset is deliberately calibrated to that range.

Key takeaways:

**1. Logistic Regression wins on AUC-PR — and that's the right answer.**
All three models land within 0.003 AUC-PR of each other. When a well-regularized logistic regression ties a GPU-tuned gradient booster on imbalanced data, the problem is approximately linear *after feature engineering* — `comorbidity_index`, `diagnosis_risk_score`, and `los_vs_expected_ratio` have already absorbed the non-linearities that trees would otherwise need to discover. The correct production call is the simpler model: LR is a few KB on disk, trains in seconds, serves at sub-millisecond latency, and its coefficients translate directly to log-odds ratios a clinician can reason about.

**2. The XGBoost comparison wasn't wasted.**
Running it is what *confirmed* there are no meaningful non-linear interactions left. Without that comparison the LR choice would be a guess; with it, the decision is defensible.

**3. XGBoost edges out on F1 (0.7345 vs 0.7201).**
If the downstream use case optimizes F1 rather than AUC-PR — for example, a fixed-threshold flagging system with equal weight on precision and recall — XGBoost is the better call. This is why reporting multiple metrics matters: the "winner" depends on which loss the business actually cares about.

**4. Feature importance is clinically sensible.**
MLlib Random Forest top 5:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `comorbidity_index` | 0.2534 |
| 2 | `comorbidity_count` | 0.1711 |
| 3 | `length_of_stay` | 0.1129 |
| 4 | `diagnosis_risk_score` | 0.0841 |
| 5 | `age` | 0.0738 |

Comorbidity burden, admission length, and diagnosis risk — exactly what a clinician would expect to drive 30-day readmission.

**5. Threshold matters more than AUC.**
At the F1-optimal threshold of **0.5443**, the best model yields **31.8% precision and 53.1% recall** — roughly 1 true positive for every 3 patients flagged, catching just over half of actual readmissions. That is the operating point a care-management team could realistically staff against a weekly intervention budget. The PR curve and threshold table (Step 7 in the notebook) are what a deployed model actually gets judged on.

---

## Clinical Interpretability (SHAP)

The notebook includes two types of SHAP visualizations on the XGBoost model:

1. **Global SHAP summary** — bar chart and beeswarm showing which features drive predictions across all patients. Answers *"What does the model care about in aggregate?"*
2. **Per-patient waterfall** — shows exactly how each feature pushed a specific patient's readmission risk up or down. Answers *"Why was this specific patient flagged high-risk?"*

In healthcare ML, interpretability is not optional. Clinicians need to justify every intervention, and regulators (FDA, CMS) increasingly require explainable models for high-stakes decisions.

---

## Project Structure

```
HealthSpark-Claims-Analytics/
├── README.md
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Spark cluster + API orchestration
├── Dockerfile                      # API container
├── Makefile                        # Build/run commands
├── healthspark_colab.ipynb         # Colab notebook (GPU XGBoost + SHAP)
├── .github/workflows/ci.yml        # GitHub Actions: pytest + ruff
├── data/
│   ├── raw/                        # Generated CSV files
│   ├── processed/                  # Parquet output
│   └── models/                     # Saved PipelineModel + results.json
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   └── 02_ml_pipeline.ipynb        # Step-by-step ML walkthrough
├── src/
│   ├── data_generation/generate_claims.py   # Synthetic data generator
│   ├── pipeline/
│   │   ├── ingestion.py            # CSV → schema → QA → Parquet
│   │   ├── transforms.py           # Joins, windows, aggregations
│   │   ├── feature_engineering.py  # 36 ML features
│   │   └── ml_pipeline.py          # LR + RF + class weights + CV
│   ├── api/main.py                 # FastAPI prediction server
│   └── utils/spark_session.py      # SparkSession factory
└── tests/test_transforms.py        # 11 PySpark integration tests
```

---

## Key PySpark Concepts Demonstrated

- **DataFrame API & Spark SQL** — every major transform shown both ways
- **Window functions** — `ROW_NUMBER`, `DENSE_RANK`, `LAG`, rolling aggregations
- **Broadcast joins** — small dimension tables broadcast to avoid shuffle
- **Strategic caching** — `.cache()` on reused DataFrames with `.unpersist()` cleanup
- **Parquet I/O** — columnar storage for data lake patterns
- **MLlib Pipelines** — end-to-end `Pipeline` with `CrossValidator` tuning
- **Class weights** — `weightCol` for imbalanced binary classification
- **Explicit schemas** — `StructType` for type-safe ingestion

---

## Databricks Compatibility

Built for local development but maps cleanly to Databricks:

| HealthSpark | Databricks Equivalent |
|-------------|----------------------|
| `spark_session.py` | Databricks auto-provisions `spark` |
| Parquet writes | Delta Lake tables |
| `cv_model.bestModel.save()` | `mlflow.spark.log_model()` |
| `model_results.json` | MLflow experiment tracking |
| Docker Spark cluster | Databricks cluster auto-scaling |
| XGBoost on GPU | Databricks ML Runtime GPU cluster |
| FastAPI | Databricks Model Serving endpoint |

---

## Testing

```bash
pytest tests/ -v
```

11 integration tests using a real `SparkSession` (not mocks), covering:

- Null handling in temporal features
- Schema correctness after transforms
- Feature count after engineering
- Aggregation correctness (denial rates by payer)
- Window function output validity
- Provider ranking order

Tests run automatically on every push via GitHub Actions CI (Python 3.10 and 3.11).

---

## Author

**Deepak Lingala** — MS Data Science, University of Arizona

[GitHub](https://github.com/Deepak-Lingala) · [LinkedIn](https://www.linkedin.com/in/deepak-lingala/)

---

## License

MIT
