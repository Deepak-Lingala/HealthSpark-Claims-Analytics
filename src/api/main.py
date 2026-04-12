"""
HealthSpark — FastAPI Prediction Server
==========================================
Serves the trained PySpark MLlib readmission model via REST API.

Endpoints:
  POST /predict  — accepts patient claim features, returns readmission probability
  GET  /stats    — returns pipeline metrics and top feature importances
  GET  /health   — service health check

The model is loaded once at startup as a PipelineModel, which includes
all preprocessing stages (StringIndexer, OHE, VectorAssembler, Scaler, RF).
This eliminates train/serve skew — the same transformations run at inference.

# Databricks equivalent: MLflow model serving or Databricks Model Serving endpoint
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "models", "readmission_model"),
)
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models", "model_results.json")

# Global references (initialized at startup)
spark: Optional[SparkSession] = None
model: Optional[PipelineModel] = None
model_results: Optional[dict] = None


# ──────────────────────────────────────────────
# Pydantic Models — Input Validation
# ──────────────────────────────────────────────

class ClaimInput(BaseModel):
    """Input schema for a single claim prediction request.

    All fields have sensible defaults or are validated to realistic ranges.
    """
    claim_amount: float = Field(..., gt=0, description="Total billed amount in USD")
    paid_amount: float = Field(..., ge=0, description="Amount paid by insurer in USD")
    length_of_stay: int = Field(..., ge=0, le=365, description="Days in facility")
    age: int = Field(..., ge=18, le=120, description="Patient age in years")
    gender: str = Field(..., pattern="^(M|F)$", description="M or F")
    comorbidity_count: int = Field(..., ge=0, le=15, description="Number of comorbidities")
    diagnosis_code: str = Field(..., description="ICD-10 diagnosis code")
    procedure_code: str = Field(..., description="CPT procedure code")
    facility_type: str = Field(..., description="Inpatient, Outpatient, Emergency, Ambulatory, or SNF")
    payer_type: str = Field(..., description="Medicare, Medicaid, Commercial, or Self-Pay")
    insurance_type: str = Field(..., description="HMO, PPO, EPO, POS, HDHP, Medicare Advantage, or Medicaid Managed Care")
    state: str = Field(..., min_length=2, max_length=2, description="US state code")

    model_config = {"json_schema_extra": {
        "examples": [{
            "claim_amount": 12500.00,
            "paid_amount": 9800.00,
            "length_of_stay": 5,
            "age": 67,
            "gender": "M",
            "comorbidity_count": 3,
            "diagnosis_code": "I50.9",
            "procedure_code": "99223",
            "facility_type": "Inpatient",
            "payer_type": "Medicare",
            "insurance_type": "Medicare Advantage",
            "state": "AZ",
        }]
    }}


class PredictionOutput(BaseModel):
    """Output schema for prediction responses."""
    readmission_probability: float
    risk_tier: str
    risk_score: float
    input_summary: dict


class StatsOutput(BaseModel):
    """Output schema for pipeline statistics."""
    model_metrics: dict
    best_params: dict
    top_features: list
    records_processed: int


class HealthOutput(BaseModel):
    """Output schema for health check."""
    status: str
    spark_active: bool
    model_loaded: bool


# ──────────────────────────────────────────────
# Risk Score Mapping (same as feature engineering)
# ──────────────────────────────────────────────
DIAGNOSIS_RISK_MAP = {
    "I50.9": 5, "J44.1": 4, "N18.9": 4, "I48.91": 4, "J18.9": 3,
    "I25.10": 3, "E11.9": 3, "I10": 2, "E78.5": 2, "F32.9": 2,
    "J45.909": 2, "N39.0": 2, "M54.5": 1, "M17.11": 1, "K21.0": 1,
    "E03.9": 1, "G47.33": 1, "J06.9": 1, "K59.00": 1, "R10.9": 1,
}

EXPECTED_LOS = {
    "E11.9": 3, "I10": 2, "J44.1": 5, "I50.9": 6, "J18.9": 5,
    "M54.5": 2, "I25.10": 4, "N18.9": 4, "J06.9": 1, "E78.5": 1,
    "F32.9": 3, "K21.0": 2, "G47.33": 1, "M17.11": 3, "E03.9": 1,
    "J45.909": 2, "I48.91": 4, "N39.0": 3, "K59.00": 1, "R10.9": 2,
}


def classify_risk_tier(probability: float) -> str:
    """Map readmission probability to clinical risk tier."""
    if probability >= 0.5:
        return "High"
    elif probability >= 0.25:
        return "Medium"
    return "Low"


# ──────────────────────────────────────────────
# Application Lifecycle
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Spark and load model at startup, clean up at shutdown."""
    global spark, model, model_results

    print("Starting HealthSpark API...")

    # Initialize SparkSession for inference (lightweight config)
    spark = (
        SparkSession.builder
        .appName("HealthSpark-API")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")  # Disable Spark UI in API mode
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print("  SparkSession initialized.")

    # Load the saved PipelineModel
    if os.path.exists(MODEL_PATH):
        model = PipelineModel.load(MODEL_PATH)
        print(f"  Model loaded from {MODEL_PATH}")
    else:
        print(f"  WARNING: Model not found at {MODEL_PATH}. Run the pipeline first.")

    # Load model results
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            model_results = json.load(f)
        print(f"  Model results loaded.")

    yield

    # Shutdown
    if spark:
        spark.stop()
        print("SparkSession stopped.")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="HealthSpark API",
    description="Readmission prediction API powered by PySpark MLlib",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allows frontend dashboards (Power BI, React, etc.) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthOutput)
def health_check():
    """Service health check endpoint."""
    return HealthOutput(
        status="healthy",
        spark_active=spark is not None and spark.sparkContext._jsc is not None,
        model_loaded=model is not None,
    )


@app.get("/stats", response_model=StatsOutput)
def get_stats():
    """Return pipeline statistics, model metrics, and top feature importances."""
    if model_results is None:
        raise HTTPException(
            status_code=503,
            detail="Model results not available. Run the pipeline first.",
        )

    return StatsOutput(
        model_metrics=model_results.get("metrics", {}),
        best_params=model_results.get("best_params", {}),
        top_features=model_results.get("feature_importances", [])[:5],
        records_processed=500_000,
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(claim: ClaimInput):
    """Predict 30-day readmission probability for a single claim.

    The model expects the same features as the training pipeline.
    We construct derived features here to match the training schema.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the pipeline first with `make run-pipeline`.",
        )

    # Compute derived features (same logic as feature_engineering.py)
    diagnosis_risk = DIAGNOSIS_RISK_MAP.get(claim.diagnosis_code, 1)
    expected_los = EXPECTED_LOS.get(claim.diagnosis_code, 2.0)
    los_ratio = claim.length_of_stay / expected_los if expected_los > 0 else 1.0
    comorbidity_index = claim.comorbidity_count * 0.5 + diagnosis_risk * 0.5

    # Age bucket
    age = claim.age
    if age < 30:
        age_bucket = "18-29"
    elif age < 45:
        age_bucket = "30-44"
    elif age < 55:
        age_bucket = "45-54"
    elif age < 65:
        age_bucket = "55-64"
    elif age < 75:
        age_bucket = "65-74"
    else:
        age_bucket = "75+"

    # Build a single-row Spark DataFrame matching the training schema
    schema = StructType([
        StructField("claim_amount",             DoubleType(),  True),
        StructField("paid_amount",              DoubleType(),  True),
        StructField("length_of_stay",           IntegerType(), True),
        StructField("age",                      IntegerType(), True),
        StructField("comorbidity_count",        IntegerType(), True),
        StructField("diagnosis_risk_score",     IntegerType(), True),
        StructField("los_vs_expected_ratio",    DoubleType(),  True),
        StructField("provider_denial_rate",     DoubleType(),  True),
        StructField("provider_claim_volume",    IntegerType(), True),
        StructField("payer_approval_rate",      DoubleType(),  True),
        StructField("patient_avg_claim",        DoubleType(),  True),
        StructField("patient_total_claims",     IntegerType(), True),
        StructField("cost_ratio_to_patient_avg", DoubleType(), True),
        StructField("comorbidity_index",        DoubleType(),  True),
        StructField("rolling_cost_90d",         DoubleType(),  True),
        StructField("claim_count_90d",          IntegerType(), True),
        StructField("days_since_last_claim",    IntegerType(), True),
        StructField("prev_denial_flag",         IntegerType(), True),
        StructField("prev_claim_amount",        DoubleType(),  True),
        StructField("claim_sequence",           IntegerType(), True),
        StructField("facility_type",            StringType(),  True),
        StructField("insurance_type",           StringType(),  True),
        StructField("gender",                   StringType(),  True),
        StructField("age_bucket",               StringType(),  True),
    ])

    row_data = [(
        float(claim.claim_amount),
        float(claim.paid_amount),
        claim.length_of_stay,
        claim.age,
        claim.comorbidity_count,
        diagnosis_risk,
        los_ratio,
        0.12,          # provider_denial_rate (population average as default)
        100,           # provider_claim_volume (median estimate)
        0.88,          # payer_approval_rate (population average)
        float(claim.claim_amount),  # patient_avg_claim (use current as estimate)
        1,             # patient_total_claims
        1.0,           # cost_ratio_to_patient_avg
        comorbidity_index,
        float(claim.claim_amount),  # rolling_cost_90d
        1,             # claim_count_90d
        -1,            # days_since_last_claim (unknown)
        0,             # prev_denial_flag (unknown)
        0.0,           # prev_claim_amount (unknown)
        1,             # claim_sequence
        claim.facility_type,
        claim.insurance_type,
        claim.gender,
        age_bucket,
    )]

    input_df = spark.createDataFrame(row_data, schema=schema)

    # Run prediction through the full pipeline model
    prediction_df = model.transform(input_df)
    result = prediction_df.select("probability", "prediction").first()

    # probability is a DenseVector: [prob_class_0, prob_class_1]
    readmission_prob = float(result["probability"][1])
    risk_tier = classify_risk_tier(readmission_prob)

    return PredictionOutput(
        readmission_probability=round(readmission_prob, 4),
        risk_tier=risk_tier,
        risk_score=round(readmission_prob * 100, 1),
        input_summary={
            "age": claim.age,
            "diagnosis": claim.diagnosis_code,
            "facility": claim.facility_type,
            "claim_amount": claim.claim_amount,
            "comorbidities": claim.comorbidity_count,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
