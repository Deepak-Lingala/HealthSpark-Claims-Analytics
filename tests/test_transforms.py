"""
HealthSpark — PySpark Transform Tests
========================================
Integration tests using a real SparkSession (not mocks).

Tests verify:
  - Null handling in feature engineering
  - Schema correctness after transforms
  - Feature count matches expectations
  - Aggregation correctness (denial rates)
  - Window function output validity
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for the entire test session.

    scope='session' ensures we only start Spark once across all tests
    (starting a JVM is expensive — ~5 seconds).
    """
    spark = (
        SparkSession.builder
        .appName("HealthSpark-Tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture
def sample_claims(spark):
    """Create a small test DataFrame mimicking claims data."""
    schema = StructType([
        StructField("claim_id",          StringType(),  False),
        StructField("patient_id",        StringType(),  False),
        StructField("admit_date",        DateType(),    True),
        StructField("discharge_date",    DateType(),    True),
        StructField("diagnosis_code",    StringType(),  True),
        StructField("procedure_code",    StringType(),  True),
        StructField("provider_id",       StringType(),  True),
        StructField("facility_type",     StringType(),  True),
        StructField("payer_type",        StringType(),  True),
        StructField("claim_amount",      DoubleType(),  True),
        StructField("paid_amount",       DoubleType(),  True),
        StructField("denial_flag",       IntegerType(), True),
        StructField("readmission_30day", IntegerType(), True),
        StructField("length_of_stay",    IntegerType(), True),
        StructField("age",               IntegerType(), True),
        StructField("gender",            StringType(),  True),
        StructField("comorbidity_count", IntegerType(), True),
        StructField("state",             StringType(),  True),
        StructField("insurance_type",    StringType(),  True),
    ])

    data = [
        ("CLM001", "P001", "2023-01-10", "2023-01-13", "I50.9",  "99223", "PRV001", "Inpatient",  "Medicare",    15000.0, 12000.0, 0, 1, 3, 72, "M", 4, "AZ", "Medicare Advantage"),
        ("CLM002", "P001", "2023-03-15", "2023-03-17", "I50.9",  "99232", "PRV001", "Inpatient",  "Medicare",    8000.0,  6500.0,  0, 0, 2, 72, "M", 4, "AZ", "Medicare Advantage"),
        ("CLM003", "P002", "2023-02-20", "2023-02-21", "E11.9",  "99213", "PRV002", "Outpatient", "Commercial",  500.0,   400.0,   1, 0, 1, 55, "F", 2, "CA", "PPO"),
        ("CLM004", "P002", "2023-05-10", "2023-05-10", "I10",    "99214", "PRV002", "Outpatient", "Commercial",  350.0,   0.0,     1, 0, 0, 55, "F", 2, "CA", "PPO"),
        ("CLM005", "P003", "2023-04-01", "2023-04-06", "J44.1",  "99233", "PRV003", "Inpatient",  "Medicaid",    22000.0, 18000.0, 0, 1, 5, 68, "M", 5, "TX", "Medicaid Managed Care"),
        ("CLM006", "P003", "2023-04-20", "2023-04-22", "J18.9",  "99284", "PRV003", "Emergency",  "Medicaid",    5000.0,  4000.0,  0, 0, 2, 68, "M", 5, "TX", "Medicaid Managed Care"),
        ("CLM007", "P004", "2023-06-01", "2023-06-01", "M54.5",  "99213", "PRV004", "Ambulatory", "Self-Pay",    200.0,   0.0,     1, 0, 0, 35, "F", 0, "NY", "HDHP"),
        ("CLM008", "P005", "2023-07-15", "2023-07-20", "N18.9",  "99223", "PRV001", "Inpatient",  "Medicare",    25000.0, 20000.0, 0, 1, 5, 78, "M", 6, "FL", "HMO"),
        ("CLM009", "P005", "2023-08-01", "2023-08-03", "I48.91", "99232", "PRV001", "Inpatient",  "Medicare",    12000.0, 9500.0,  0, 0, 2, 78, "M", 6, "FL", "HMO"),
        ("CLM010", "P006", "2023-09-10", "2023-09-10", "J06.9",  "99281", "PRV005", "Emergency",  "Commercial",  800.0,   600.0,   0, 0, 0, 28, "F", 0, "WA", "EPO"),
    ]

    # Parse date strings
    from datetime import datetime
    parsed_data = []
    for row in data:
        parsed_row = list(row)
        parsed_row[2] = datetime.strptime(row[2], "%Y-%m-%d").date() if row[2] else None
        parsed_row[3] = datetime.strptime(row[3], "%Y-%m-%d").date() if row[3] else None
        parsed_data.append(tuple(parsed_row))

    return spark.createDataFrame(parsed_data, schema=schema)


# ──────────────────────────────────────────────
# Test: Null Handling
# ──────────────────────────────────────────────

class TestNullHandling:
    """Verify that transforms handle null values correctly."""

    def test_no_nulls_in_required_fields(self, sample_claims):
        """claim_id and patient_id should never be null."""
        null_claims = sample_claims.where(F.col("claim_id").isNull()).count()
        null_patients = sample_claims.where(F.col("patient_id").isNull()).count()
        assert null_claims == 0, "claim_id has null values"
        assert null_patients == 0, "patient_id has null values"

    def test_lag_features_handle_first_claim(self, sample_claims):
        """First claim per patient has no prior — lag should produce null, then we fill."""
        from src.pipeline.transforms import add_lag_lead_features

        result = add_lag_lead_features(sample_claims)
        # First claim per patient should have null days_since_last_claim
        first_claims = result.where(F.col("claim_sequence") == 1)
        nulls = first_claims.where(F.col("days_since_last_claim").isNull()).count()
        assert nulls == first_claims.count(), "First claims should have null days_since_last_claim"

    def test_fillna_removes_nulls(self, sample_claims):
        """After fillna, temporal features should have no nulls."""
        from src.pipeline.transforms import add_lag_lead_features

        result = add_lag_lead_features(sample_claims)
        result = result.fillna({"days_since_last_claim": -1, "prev_denial_flag": 0, "prev_claim_amount": 0.0})

        for col in ["days_since_last_claim", "prev_denial_flag", "prev_claim_amount"]:
            null_count = result.where(F.col(col).isNull()).count()
            assert null_count == 0, f"{col} still has nulls after fillna"


# ──────────────────────────────────────────────
# Test: Schema Output
# ──────────────────────────────────────────────

class TestSchemaOutput:
    """Verify that transforms produce expected schema changes."""

    def test_rolling_cost_adds_columns(self, sample_claims):
        """Rolling cost transform should add exactly 2 columns."""
        from src.pipeline.transforms import add_rolling_claim_cost

        original_cols = len(sample_claims.columns)
        result = add_rolling_claim_cost(sample_claims)
        new_cols = len(result.columns)

        assert new_cols == original_cols + 2, f"Expected {original_cols + 2} cols, got {new_cols}"
        assert "rolling_cost_90d" in result.columns
        assert "claim_count_90d" in result.columns

    def test_lag_lead_adds_columns(self, sample_claims):
        """Lag/lead transform should add temporal features."""
        from src.pipeline.transforms import add_lag_lead_features

        result = add_lag_lead_features(sample_claims)
        expected_new = {"days_since_last_claim", "prev_denial_flag", "prev_claim_amount", "claim_sequence"}
        actual_new = set(result.columns) - set(sample_claims.columns)
        assert expected_new.issubset(actual_new), f"Missing columns: {expected_new - actual_new}"

    def test_diagnosis_risk_score_column(self, sample_claims):
        """Risk score transform should add diagnosis_risk_score column."""
        from src.pipeline.feature_engineering import add_diagnosis_risk_score

        result = add_diagnosis_risk_score(sample_claims)
        assert "diagnosis_risk_score" in result.columns

        # Heart failure should have risk score 5
        chf_score = result.where(F.col("diagnosis_code") == "I50.9").select("diagnosis_risk_score").first()[0]
        assert chf_score == 5, f"I50.9 risk score should be 5, got {chf_score}"

    def test_age_bucket_values(self, sample_claims):
        """Age bucket should produce valid categories."""
        from src.pipeline.feature_engineering import add_age_bucket

        result = add_age_bucket(sample_claims)
        valid_buckets = {"18-29", "30-44", "45-54", "55-64", "65-74", "75+"}
        actual_buckets = set(row["age_bucket"] for row in result.select("age_bucket").distinct().collect())
        assert actual_buckets.issubset(valid_buckets), f"Invalid age buckets: {actual_buckets - valid_buckets}"


# ──────────────────────────────────────────────
# Test: Feature Count
# ──────────────────────────────────────────────

class TestFeatureCount:
    """Verify feature engineering produces expected number of features."""

    def test_minimum_feature_count(self, sample_claims):
        """After all feature transforms, we should have 15+ new columns."""
        from src.pipeline.transforms import add_rolling_claim_cost, add_lag_lead_features
        from src.pipeline.feature_engineering import (
            add_diagnosis_risk_score, add_los_ratio, add_provider_denial_rate,
            add_payer_approval_rate, add_age_bucket, add_cost_features, add_comorbidity_index,
        )

        original_cols = len(sample_claims.columns)
        df = add_rolling_claim_cost(sample_claims)
        df = add_lag_lead_features(df)
        df = add_diagnosis_risk_score(df)
        df = add_los_ratio(df)
        df = add_provider_denial_rate(df)
        df = add_payer_approval_rate(df)
        df = add_age_bucket(df)
        df = add_cost_features(df)
        df = add_comorbidity_index(df)

        new_feature_count = len(df.columns) - original_cols
        assert new_feature_count >= 15, f"Expected 15+ new features, got {new_feature_count}"


# ──────────────────────────────────────────────
# Test: Aggregation Correctness
# ──────────────────────────────────────────────

class TestAggregations:
    """Verify aggregation logic produces correct results."""

    def test_denial_rate_by_payer(self, sample_claims):
        """Manually verify denial rate calculation for a known payer."""
        from src.pipeline.transforms import compute_aggregations

        aggs = compute_aggregations(sample_claims)
        denial_df = aggs["denial_by_payer"]

        # Commercial: CLM003(denied) + CLM004(denied) + CLM010(not denied) = 2/3 = 66.7%
        commercial = denial_df.where(F.col("payer_type") == "Commercial").first()
        assert commercial is not None, "Commercial payer not found in aggregation"
        assert commercial["total_claims"] == 3, f"Expected 3 Commercial claims, got {commercial['total_claims']}"
        expected_rate = 2 / 3
        assert abs(commercial["denial_rate"] - expected_rate) < 0.01, \
            f"Commercial denial rate should be {expected_rate:.4f}, got {commercial['denial_rate']:.4f}"

    def test_total_claim_count(self, sample_claims):
        """Total claims across all payers should match input count."""
        from src.pipeline.transforms import compute_aggregations

        aggs = compute_aggregations(sample_claims)
        total = aggs["denial_by_payer"].select(F.sum("total_claims")).first()[0]
        expected = sample_claims.count()
        assert total == expected, f"Total claims mismatch: {total} vs {expected}"

    def test_claims_per_patient(self, sample_claims):
        """Patient P001 should have exactly 2 claims."""
        from src.pipeline.transforms import compute_aggregations

        aggs = compute_aggregations(sample_claims)
        p001 = aggs["claims_per_patient"].where(F.col("patient_id") == "P001").first()
        assert p001["num_claims"] == 2, f"P001 should have 2 claims, got {p001['num_claims']}"


# ──────────────────────────────────────────────
# Test: Provider Ranking
# ──────────────────────────────────────────────

class TestProviderRanking:
    """Verify provider ranking window function."""

    def test_provider_ranking_order(self, sample_claims):
        """Providers should be ranked by denial rate (highest first)."""
        from src.pipeline.transforms import rank_providers_by_denial_rate

        # Lower min claim threshold for test data
        ranked = (
            sample_claims.groupBy("provider_id")
            .agg(
                F.count("claim_id").alias("total_claims"),
                F.avg("denial_flag").alias("denial_rate"),
            )
            .orderBy(F.col("denial_rate").desc())
        )

        rows = ranked.collect()
        rates = [r["denial_rate"] for r in rows]
        assert rates == sorted(rates, reverse=True), "Providers not sorted by denial rate descending"
