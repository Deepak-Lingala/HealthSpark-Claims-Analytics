# HealthSpark-Claims-Analytics

**Distributed Healthcare Claims Analytics & ML Pipeline**

[![CI](https://github.com/Deepak-Lingala/HealthSpark-Claims-Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Deepak-Lingala/HealthSpark-Claims-Analytics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PySpark](https://img.shields.io/badge/PySpark-3.5.3-orange?logo=apachespark)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU-red?logo=xgboost)
![SHAP](https://img.shields.io/badge/SHAP-Interpretability-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![pytest](https://img.shields.io/badge/Tests-pytest-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

Production-grade pipeline for healthcare claims analytics, 30-day readmission prediction, and real-time ML serving. Built to demonstrate **Optum-level distributed healthcare analytics skills**: end-to-end ETL, multi-model comparison with PySpark MLlib + GPU XGBoost, SHAP-based clinical interpretability, threshold-tuned deployment, and FastAPI serving — all on synthetic, HIPAA-safe data.

---

## Highlights

- **500K synthetic claims** with realistic CMS-calibrated distributions (ICD-10, ~12% denial, ~15% readmission)
- **PySpark ETL pipeline**: schema validation, window functions, broadcast joins, Spark SQL
- **Class imbalance handled** via class weights (MLlib) and `scale_pos_weight` (XGBoost)
- **3-model comparison**: Logistic Regression baseline → tuned Random Forest → GPU XGBoost
- **SHAP interpretability**: global feature importance + per-patient waterfall explanations
- **Precision-Recall curve + threshold analysis** for clinical operating-point selection
- **FastAPI inference server** with `PipelineModel` — zero train/serve skew
- **Docker Compose cluster**: Spark master + 2 workers + API container
- **CI/CD via GitHub Actions**: pytest + ruff on every push
- **Google Colab Pro notebook** runs the full pipeline on GPU in ~5-8 minutes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  HealthSpark-Claims-Analytics Architecture               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │  Synthetic   │    │  PySpark     │    │  Parquet Data Lake       │   │
│  │  Data Gen    │───▶│  Ingestion   │───▶│  (Columnar Storage)      │   │
│  │  (500K rows) │    │  + QA Checks │    │                          │   │
│  └──────────────┘    └──────────────┘    └────────────┬─────────────┘   │
│                                                       │                 │
│                                                       ▼                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    PySpark Transformations                       │   │
│  │  • Window functions (rolling 90d costs, provider rankings)       │   │
│  │  • Broadcast joins (patients dim → claims fact)                  │   │
│  │  • Aggregations (denial rates, LOS by diagnosis)                 │   │
│  │  • DataFrame API + Spark SQL shown in parallel                   │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Feature Engineering                           │   │
│  │  • 20+ ML features (risk scores, LOS ratio, provider stats)      │   │
│  │  • Strategic .cache() on reused DataFrames                       │   │
│  │  • Class weights added for imbalance handling                    │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │               Multi-Model Training & Comparison                  │   │
│  │                                                                  │   │
│  │   ┌────────────────┐ ┌────────────────┐ ┌────────────────┐       │   │
│  │   │ MLlib LogReg   │ │ MLlib RF + CV  │ │ XGBoost (GPU)  │       │   │
│  │   │  (baseline)    │ │ (8 combos x 3) │ │ scale_pos_wt   │       │   │
│  │   └────────┬───────┘ └────────┬───────┘ └────────┬───────┘       │   │
│  │            └────────┬────────┘                   │               │   │
│  │                     ▼                            ▼               │   │
│  │   AUC-ROC / AUC-PR / F1 / Precision / Recall Comparison          │   │
│  │                     │                                            │   │
│  │                     ▼                                            │   │
│  │   SHAP values + Precision-Recall curve + Threshold analysis      │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Serving Layer                         │   │
│  │  POST /predict  → readmission probability + risk tier            │   │
│  │  GET  /stats    → model comparison, feature importances          │   │
│  │  GET  /health   → service health check                           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Infrastructure: Docker Compose (Spark Master + 2 Workers + API)        │
│  CI/CD: GitHub Actions (pytest on push + PR)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option 1 — Google Colab Pro (recommended, GPU-accelerated)

1. Go to [Google Colab](https://colab.research.google.com/)
2. **File → Open notebook → GitHub**, paste: `Deepak-Lingala/HealthSpark-Claims-Analytics`
3. Open `healthspark_colab.ipynb`
4. **Runtime → Change runtime type → T4 GPU**
5. Run all cells (~5-8 minutes)

The notebook clones the repo, installs dependencies, generates data, runs the full pipeline, and produces all visualizations.

### Option 2 — Local (Docker)

```bash
# Clone and build
git clone https://github.com/Deepak-Lingala/HealthSpark-Claims-Analytics.git
cd HealthSpark-Claims-Analytics
make docker-up

# Or run individual stages manually
make setup
make generate-data
make run-pipeline
make run-api       # FastAPI on http://localhost:8000/docs
```

### Option 3 — Local (venv)

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

All data is **100% synthetic** and **HIPAA-safe** — no real patient information is used.

| Table | Records | Description |
|-------|---------|-------------|
| `claims.csv` | 500,000 | Healthcare insurance claims with ICD-10, CPT, costs, outcomes |
| `patients.csv` | 50,000 | Patient demographics (age, gender, state, insurance type) |

Key distributions calibrated to real-world benchmarks:
- **Denial rate**: ~12% (varies by payer: Self-Pay 22%, Medicare 8%)
- **30-day readmission rate**: ~15% (CMS HRRP benchmark)
- **Age distribution**: Beta-skewed toward 55-75 (high-utilization population)
- **Top 20 ICD-10 codes** by prevalence (diabetes, hypertension, COPD, CHF, pneumonia, CKD)
- **Claim costs**: log-normal distribution by facility type

---

## Model Comparison

The pipeline trains and compares **three models** on the same feature set, with class weights to handle the 15% positive class rate:

| Model | Purpose | Hyperparameter Tuning |
|-------|---------|-----------------------|
| **Logistic Regression** | Interpretable baseline | ElasticNet regularization |
| **Random Forest (MLlib)** | Non-linear ensemble | CrossValidator 8 combos × 3 folds |
| **XGBoost (GPU)** | Production model | `scale_pos_weight`, early stopping |

**Evaluation metrics** (populated after running the pipeline):

| Metric | LR Baseline | RF Tuned | XGBoost GPU |
|--------|-------------|----------|-------------|
| AUC-ROC | *run pipeline* | *run pipeline* | *run pipeline* |
| AUC-PR  | *run pipeline* | *run pipeline* | *run pipeline* |
| F1 Score | *run pipeline* | *run pipeline* | *run pipeline* |
| Precision | *run pipeline* | *run pipeline* | *run pipeline* |
| Recall | *run pipeline* | *run pipeline* | *run pipeline* |

> **Note on AUC-PR**: For imbalanced healthcare data (15% positive), AUC-PR (Average Precision) is more informative than AUC-ROC. The Colab notebook reports both.

### Why three models?

- **Logistic Regression** establishes a linear baseline. If it performs well, the problem is mostly linear and a complex model adds little value.
- **Random Forest** captures non-linear interactions and is fully distributed in Spark for billion-row datasets.
- **XGBoost on GPU** is the production workhorse — consistently wins Kaggle healthcare competitions and trains 10-20x faster on GPU than CPU ensembles.

---

## Clinical Interpretability (SHAP)

The notebook includes two types of SHAP visualizations for the XGBoost model:

1. **Global SHAP summary** — Bar chart and beeswarm showing which features drive predictions across all patients. Answers: *"What does the model care about in aggregate?"*
2. **Per-patient waterfall** — Shows exactly how each feature pushed a specific patient's readmission risk up or down. Answers: *"Why was **this** patient flagged high-risk?"*

In healthcare ML, interpretability is not optional — clinicians need to justify every intervention, and regulators (FDA, CMS) increasingly require explainable models for high-stakes decisions.

---

## Threshold Analysis

A deployed readmission model's real impact depends on the **operating threshold**, not just AUC-ROC. The notebook produces a threshold table like this:

| Threshold | Precision | Recall | Flag Rate |
|-----------|-----------|--------|-----------|
| 0.15 | — | — | — |
| 0.25 | — | — | — |
| 0.35 | — | — | — |
| 0.50 | — | — | — |

This is the conversation a data scientist has with a care-management team: *"At threshold 0.30, we flag 25% of patients with 42% precision and 80% recall. Your team can handle 300 intervention cases per week — which operating point fits that budget?"*

The F1-optimal threshold is marked on the Precision-Recall curve.

---

## Project Structure

```
HealthSpark-Claims-Analytics/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies (incl. xgboost, shap)
├── docker-compose.yml                 # Spark cluster + API orchestration
├── Dockerfile                         # API container
├── Makefile                           # Build/run commands
├── healthspark_colab.ipynb            # Colab Pro notebook (GPU XGBoost + SHAP)
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions: pytest + ruff
├── data/
│   ├── raw/                           # Generated CSV files
│   ├── processed/                     # Parquet output
│   └── models/                        # Saved PipelineModel + results.json
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis (8 charts)
│   └── 02_ml_pipeline.ipynb           # Step-by-step ML walkthrough
├── src/
│   ├── data_generation/
│   │   └── generate_claims.py         # Synthetic claims + patients generator
│   ├── pipeline/
│   │   ├── ingestion.py               # CSV → schema → QA → Parquet
│   │   ├── transforms.py              # Joins, windows, aggregations (DF API + SQL)
│   │   ├── feature_engineering.py     # 20+ ML features
│   │   └── ml_pipeline.py             # LR + RF comparison + class weights
│   ├── api/
│   │   └── main.py                    # FastAPI prediction server
│   └── utils/
│       └── spark_session.py           # SparkSession factory
└── tests/
    └── test_transforms.py             # 11 pytest + PySpark integration tests
```

---

## Key PySpark Concepts Demonstrated

- **DataFrame API & Spark SQL** — Every major transform shown both ways
- **Window Functions** — `ROW_NUMBER`, `DENSE_RANK`, `LAG`, rolling aggregations
- **Broadcast Joins** — Small dimension tables broadcast to avoid shuffle
- **Strategic Caching** — `.cache()` on reused DataFrames with `.unpersist()` cleanup
- **Parquet I/O** — Columnar storage for data lake patterns
- **MLlib Pipelines** — End-to-end `Pipeline` with `CrossValidator` tuning
- **Class Weights** — `weightCol` for imbalanced binary classification
- **Explicit Schemas** — `StructType` schemas for type-safe ingestion

---

## Databricks Compatibility

This project is designed for local development but maps directly to Databricks:

| HealthSpark | Databricks Equivalent |
|-------------|----------------------|
| `spark_session.py` | Databricks auto-provisions `spark` |
| Parquet writes | Delta Lake tables |
| `cv_model.bestModel.save()` | `mlflow.spark.log_model()` |
| JSON results file | MLflow experiment tracking |
| Docker Spark cluster | Databricks cluster auto-scaling |
| XGBoost on GPU | Databricks ML Runtime GPU cluster |
| FastAPI | Databricks Model Serving endpoint |

---

## Testing

```bash
pytest tests/ -v
```

11 integration tests using a real `SparkSession` (not mocks) covering:
- Null handling in temporal features
- Schema correctness after transforms
- Feature count after engineering
- Aggregation correctness (denial rates by payer)
- Window function output validity
- Provider ranking order

Tests run automatically on every push via GitHub Actions CI.

---

## Author

**Deepak Lingala** — MS Data Science, University of Arizona

[GitHub](https://github.com/Deepak-Lingala) • [LinkedIn](https://www.linkedin.com/in/deepak-lingala/)

---

## License

MIT License — see LICENSE file for details.
