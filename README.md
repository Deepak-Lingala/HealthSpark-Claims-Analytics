# HealthSpark — Distributed Healthcare Claims Analytics & ML Pipeline

**Production-grade PySpark pipeline for healthcare claims analytics, readmission prediction, and real-time ML serving.**

Built to demonstrate Optum-level distributed healthcare analytics skills: end-to-end ETL, feature engineering with PySpark MLlib, cross-validated model training, and FastAPI serving — all on synthetic, HIPAA-safe data.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HealthSpark Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │  Synthetic    │    │  PySpark     │    │  Parquet Data Lake       │   │
│  │  Data Gen     │───▶│  Ingestion   │───▶│  (Columnar Storage)      │   │
│  │  (500K rows)  │    │  + QA Checks │    │                          │   │
│  └──────────────┘    └──────────────┘    └────────────┬─────────────┘   │
│                                                        │                 │
│                                                        ▼                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    PySpark Transformations                        │   │
│  │  • Window Functions (rolling costs, provider rankings)           │   │
│  │  • Joins (broadcast for small tables, sort-merge for large)      │   │
│  │  • Aggregations (denial rates, LOS by diagnosis)                 │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                                │                                         │
│                                ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Feature Engineering                            │   │
│  │  • 15+ ML features via PySpark DataFrame API                     │   │
│  │  • VectorAssembler, StringIndexer, OneHotEncoder                 │   │
│  │  • Strategic .cache() on reused DataFrames                       │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                                │                                         │
│                                ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                 PySpark MLlib Pipeline                            │   │
│  │  StringIndexer → OHE → VectorAssembler → Scaler → RandomForest  │   │
│  │  • CrossValidator + ParamGridBuilder (27 combos)                 │   │
│  │  • BinaryClassificationEvaluator (AUC-ROC)                      │   │
│  │  • Model saved to data/models/                                   │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                                │                                         │
│                                ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Serving Layer                          │   │
│  │  POST /predict  → readmission probability + risk tier            │   │
│  │  GET  /stats    → pipeline metrics, feature importances          │   │
│  │  GET  /health   → service health check                           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Infrastructure: Docker Compose (Spark Master + 2 Workers + API)       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PySpark](https://img.shields.io/badge/PySpark-3.5.x-orange?logo=apachespark)
![MLlib](https://img.shields.io/badge/MLlib-Classification-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![Parquet](https://img.shields.io/badge/Storage-Parquet-purple)
![pytest](https://img.shields.io/badge/Tests-pytest-yellow)

---

## Quick Start

```bash
# 1. Install dependencies and set up directories
make setup

# 2. Generate synthetic data, run ETL + ML pipeline, start API
make generate-data
make run-pipeline
make run-api

# Or run everything with Docker:
make docker-up
```

---

## Dataset

All data is **100% synthetic** and **HIPAA-safe** — no real patient information is used.

| Table | Records | Description |
|-------|---------|-------------|
| `claims.csv` | 500,000 | Healthcare insurance claims with ICD-10 codes, costs, outcomes |
| `patients.csv` | ~50,000 | Patient demographics (age, gender, state, insurance type) |

Key distributions:
- **Denial rate**: ~12% (realistic for commercial + Medicare mix)
- **30-day readmission rate**: ~15% (aligned with CMS benchmarks)
- **Age distribution**: Skewed toward 45–75 (typical claims population)
- **Diagnoses**: Top 20 ICD-10 codes (diabetes, hypertension, COPD, CHF, pneumonia, etc.)

---

## Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | _Run pipeline to populate_ |
| F1 Score | _Run pipeline to populate_ |
| Best numTrees | _Run pipeline to populate_ |
| Best maxDepth | _Run pipeline to populate_ |
| Records processed | 500,000 |

---

## Project Structure

```
healthspark/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── docker-compose.yml               # Spark cluster + API orchestration
├── Makefile                         # Build/run commands
├── data/
│   ├── raw/                         # Generated CSV files land here
│   ├── processed/                   # Parquet output from ETL
│   └── models/                      # Saved PySpark MLlib pipeline model
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis (8+ charts)
│   └── 02_ml_pipeline.ipynb         # Step-by-step ML walkthrough
├── src/
│   ├── data_generation/
│   │   └── generate_claims.py       # Synthetic claims + patients generator
│   ├── pipeline/
│   │   ├── ingestion.py             # CSV → validated Spark DF → Parquet
│   │   ├── transforms.py            # Joins, windows, aggregations
│   │   ├── feature_engineering.py   # 15+ ML features via PySpark API
│   │   └── ml_pipeline.py           # MLlib Pipeline + CrossValidator
│   ├── api/
│   │   └── main.py                  # FastAPI prediction server
│   └── utils/
│       └── spark_session.py         # SparkSession factory
└── tests/
    └── test_transforms.py           # pytest + PySpark integration tests
```

---

## Key PySpark Concepts Demonstrated

- **DataFrame API & Spark SQL**: Every major transform shown both ways
- **Window Functions**: `ROW_NUMBER`, `DENSE_RANK`, `LAG`, rolling aggregations
- **Broadcast Joins**: Small dimension tables broadcast to avoid shuffle
- **Strategic Caching**: `.cache()` on reused DataFrames with `.unpersist()` cleanup
- **Parquet I/O**: Columnar storage for production-grade data lake patterns
- **MLlib Pipelines**: End-to-end `Pipeline` with `CrossValidator` tuning
- **Explicit Schemas**: `StructType` schemas for type-safe ingestion

---

## Databricks Compatibility

This project is designed for local development but maps directly to Databricks:
- `spark_session.py` → Databricks provides `spark` automatically
- Parquet writes → Delta Lake tables
- `model.save()` → MLflow model registry
- Docker Spark cluster → Databricks cluster auto-scaling

---

*Built by Deepak Lingala — MS Data Science, University of Arizona*
