"""
HealthSpark — Feature Engineering
====================================
Engineers 15+ ML features using PySpark DataFrame API and MLlib transformers.

All features are built using PySpark-native operations (not pandas/sklearn)
to demonstrate distributed feature engineering that scales to billions of rows.

Feature categories:
  1. Temporal: days_since_last_claim, claim_frequency_90d
  2. Cost: avg_claim_amount_90d, cost_ratio_to_mean
  3. Clinical: diagnosis_risk_score, comorbidity_index, LOS_vs_expected
  4. Provider: provider_denial_rate_historical
  5. Payer: payer_approval_rate
  6. Demographic: age_bucket, one-hot encoded categoricals

# Databricks equivalent: Feature Store tables (databricks.feature_store)
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import (
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)


# ──────────────────────────────────────────────
# ICD-10 → Risk Score Mapping
# ──────────────────────────────────────────────
# Maps diagnosis codes to CMS-style risk tiers (1=low, 5=high)
# Higher risk scores indicate diagnoses associated with higher
# healthcare utilization and worse outcomes.

DIAGNOSIS_RISK_MAP = {
    "I50.9":   5,  # Heart failure — highest risk, frequent readmissions
    "J44.1":   4,  # COPD exacerbation — high risk
    "N18.9":   4,  # Chronic kidney disease
    "I48.91":  4,  # Atrial fibrillation
    "J18.9":   3,  # Pneumonia
    "I25.10":  3,  # Coronary artery disease
    "E11.9":   3,  # Type 2 diabetes
    "I10":     2,  # Hypertension — common, moderate risk
    "E78.5":   2,  # Hyperlipidemia
    "F32.9":   2,  # Depression
    "J45.909": 2,  # Asthma
    "N39.0":   2,  # UTI
    "M54.5":   1,  # Low back pain — low risk
    "M17.11":  1,  # Knee osteoarthritis
    "K21.0":   1,  # GERD
    "E03.9":   1,  # Hypothyroidism
    "G47.33":  1,  # Sleep apnea
    "J06.9":   1,  # Upper respiratory infection
    "K59.00":  1,  # Constipation
    "R10.9":   1,  # Abdominal pain
}

# Expected LOS by diagnosis (used for LOS ratio feature)
EXPECTED_LOS = {
    "E11.9": 3, "I10": 2, "J44.1": 5, "I50.9": 6, "J18.9": 5,
    "M54.5": 2, "I25.10": 4, "N18.9": 4, "J06.9": 1, "E78.5": 1,
    "F32.9": 3, "K21.0": 2, "G47.33": 1, "M17.11": 3, "E03.9": 1,
    "J45.909": 2, "I48.91": 4, "N39.0": 3, "K59.00": 1, "R10.9": 2,
}


def add_diagnosis_risk_score(df: DataFrame) -> DataFrame:
    """Map ICD-10 diagnosis codes to numerical risk scores.

    Uses a UDF-free approach via create_map — more efficient than UDFs
    because it stays within Spark's optimized execution engine (Catalyst).
    """
    # Build a Spark map column from the Python dict
    mapping_expr = F.create_map(
        *[item for code, score in DIAGNOSIS_RISK_MAP.items()
          for item in (F.lit(code), F.lit(score))]
    )

    df = df.withColumn(
        "diagnosis_risk_score",
        F.coalesce(mapping_expr[F.col("diagnosis_code")], F.lit(1))
    )
    return df


def add_los_ratio(df: DataFrame) -> DataFrame:
    """Compute length_of_stay / expected_LOS for each diagnosis.

    A ratio > 1.0 means the patient stayed longer than expected,
    which is a strong signal for complications and readmission risk.
    """
    mapping_expr = F.create_map(
        *[item for code, los in EXPECTED_LOS.items()
          for item in (F.lit(code), F.lit(float(los)))]
    )

    df = df.withColumn(
        "expected_los",
        F.coalesce(mapping_expr[F.col("diagnosis_code")], F.lit(2.0))
    ).withColumn(
        "los_vs_expected_ratio",
        F.when(F.col("expected_los") > 0,
               F.col("length_of_stay") / F.col("expected_los"))
        .otherwise(1.0)
    ).drop("expected_los")

    return df


def add_provider_denial_rate(df: DataFrame) -> DataFrame:
    """Add historical denial rate per provider as a feature.

    This is a target-encoding-style feature: it uses aggregate statistics
    from the provider's history. In production, you'd compute this on a
    training-only window to avoid data leakage.

    We use a Window function rather than a separate groupBy + join to
    keep the pipeline concise and avoid an extra shuffle.
    """
    provider_window = Window.partitionBy("provider_id")

    df = df.withColumn(
        "provider_denial_rate",
        F.avg("denial_flag").over(provider_window)
    ).withColumn(
        "provider_claim_volume",
        F.count("claim_id").over(provider_window)
    )
    return df


def add_payer_approval_rate(df: DataFrame) -> DataFrame:
    """Add payer-level approval rate (1 - denial_rate) as a feature."""
    payer_window = Window.partitionBy("payer_type")

    df = df.withColumn(
        "payer_approval_rate",
        1.0 - F.avg("denial_flag").over(payer_window)
    )
    return df


def add_age_bucket(df: DataFrame) -> DataFrame:
    """Bin age into clinical age groups.

    Binning continuous variables reduces noise and captures non-linear
    relationships (e.g., readmission risk increases sharply after 65).
    """
    df = df.withColumn(
        "age_bucket",
        F.when(F.col("age") < 30, "18-29")
        .when(F.col("age") < 45, "30-44")
        .when(F.col("age") < 55, "45-54")
        .when(F.col("age") < 65, "55-64")
        .when(F.col("age") < 75, "65-74")
        .otherwise("75+")
    )
    return df


def add_cost_features(df: DataFrame) -> DataFrame:
    """Add cost-based features relative to patient and global averages."""
    # Patient-level average claim cost
    patient_window = Window.partitionBy("patient_id")

    df = df.withColumn(
        "patient_avg_claim",
        F.avg("claim_amount").over(patient_window)
    ).withColumn(
        "patient_total_claims",
        F.count("claim_id").over(patient_window)
    )

    # Cost ratio: this claim vs patient's average
    df = df.withColumn(
        "cost_ratio_to_patient_avg",
        F.when(F.col("patient_avg_claim") > 0,
               F.col("claim_amount") / F.col("patient_avg_claim"))
        .otherwise(1.0)
    )

    return df


def add_comorbidity_index(df: DataFrame) -> DataFrame:
    """Compute a weighted comorbidity index.

    Combines comorbidity count with diagnosis risk for a richer signal
    than raw comorbidity count alone. Inspired by the Charlson index.
    """
    df = df.withColumn(
        "comorbidity_index",
        (F.col("comorbidity_count") * 0.5 + F.col("diagnosis_risk_score") * 0.5)
    )
    return df


def build_feature_vector(
    df: DataFrame,
) -> tuple[DataFrame, list[str], list]:
    """Assemble all features into a single vector using MLlib transformers.

    Pipeline stages:
      1. StringIndexer: convert categoricals to numeric indices
      2. OneHotEncoder: create sparse binary vectors from indices
      3. VectorAssembler: combine all features into one vector column

    Why MLlib transformers instead of pandas get_dummies?
      - Scales to billions of rows (distributed across workers)
      - Integrates with MLlib Pipeline for model serialization
      - Transformers are saved with the model — no train/serve skew

    # Databricks equivalent: same MLlib API, runs on cluster automatically
    """
    # Categorical columns to encode
    cat_cols = ["facility_type", "insurance_type", "gender", "age_bucket"]
    indexed_cols = [f"{c}_idx" for c in cat_cols]
    ohe_cols = [f"{c}_ohe" for c in cat_cols]

    # Numeric feature columns (already computed above)
    numeric_cols = [
        "claim_amount",
        "paid_amount",
        "length_of_stay",
        "age",
        "comorbidity_count",
        "diagnosis_risk_score",
        "los_vs_expected_ratio",
        "provider_denial_rate",
        "provider_claim_volume",
        "payer_approval_rate",
        "patient_avg_claim",
        "patient_total_claims",
        "cost_ratio_to_patient_avg",
        "comorbidity_index",
        "rolling_cost_90d",
        "claim_count_90d",
    ]

    # Fill nulls in temporal features (first claims per patient have no lag)
    df = df.fillna({
        "days_since_last_claim": -1,
        "prev_denial_flag": 0,
        "prev_claim_amount": 0.0,
        "rolling_cost_90d": 0.0,
        "claim_count_90d": 0,
    })

    # Add lag features to numeric columns
    numeric_cols.extend([
        "days_since_last_claim",
        "prev_denial_flag",
        "prev_claim_amount",
        "claim_sequence",
    ])

    # Stage 1: StringIndexer for each categorical column
    indexers = [
        StringIndexer(
            inputCol=col, outputCol=idx_col,
            handleInvalid="keep"  # Handle unseen categories at inference time
        )
        for col, idx_col in zip(cat_cols, indexed_cols)
    ]

    # Stage 2: OneHotEncoder for indexed columns
    encoder = OneHotEncoder(
        inputCols=indexed_cols,
        outputCols=ohe_cols,
        dropLast=True,  # Avoid multicollinearity by dropping last category
    )

    # Stage 3: VectorAssembler — combine everything into one feature vector
    assembler_inputs = numeric_cols + ohe_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="skip",  # Skip rows with nulls in feature columns
    )

    # Fit and transform indexers
    for indexer in indexers:
        model = indexer.fit(df)
        df = model.transform(df)

    # Fit and transform encoder
    encoder_model = encoder.fit(df)
    df = encoder_model.transform(df)

    # Assemble feature vector
    df = assembler.transform(df)

    # Collect pipeline stages for the ML Pipeline (used in ml_pipeline.py)
    stages = indexers + [encoder, assembler]

    print(f"  Feature vector assembled: {len(assembler_inputs)} input features")
    print(f"  Numeric features: {len(numeric_cols)}")
    print(f"  Categorical features (OHE): {len(ohe_cols)}")

    return df, assembler_inputs, stages


def engineer_features(
    spark: SparkSession,
    enriched_df: DataFrame,
) -> DataFrame:
    """Run the full feature engineering pipeline.

    Returns a DataFrame with all features added and a 'features' vector column
    ready for MLlib model training.
    """
    print("=" * 60)
    print("HealthSpark — Feature Engineering Pipeline")
    print("=" * 60)

    # Apply feature transforms sequentially — each builds on prior columns
    print("\n[1/8] Adding diagnosis risk scores...")
    features_df = add_diagnosis_risk_score(enriched_df)

    print("[2/8] Computing LOS vs expected ratio...")
    features_df = add_los_ratio(features_df)

    print("[3/8] Adding provider denial rate...")
    features_df = add_provider_denial_rate(features_df)

    print("[4/8] Adding payer approval rate...")
    features_df = add_payer_approval_rate(features_df)

    print("[5/8] Binning age into clinical groups...")
    features_df = add_age_bucket(features_df)

    print("[6/8] Computing cost features...")
    features_df = add_cost_features(features_df)

    print("[7/8] Computing comorbidity index...")
    features_df = add_comorbidity_index(features_df)

    # Cache before vector assembly — this DataFrame is read multiple times
    # during indexer.fit(), encoder.fit(), and assembler.transform()
    # Without caching, Spark would recompute ALL prior stages for each action
    features_df.cache()
    cached_count = features_df.count()
    print(f"  Cached {cached_count:,} records for vector assembly")

    print("[8/8] Assembling feature vector (StringIndexer -> OHE -> VectorAssembler)...")
    features_df, feature_names, stages = build_feature_vector(features_df)

    print(f"\nFeature engineering complete. Total columns: {len(features_df.columns)}")
    print(f"Feature names: {feature_names}\n")

    return features_df
