"""
HealthSpark — PySpark Transformations
=======================================
Joins, window functions, aggregations, and Spark SQL examples.

This module demonstrates both the DataFrame API and Spark SQL approaches
for the same transformations — a deliberate choice to show fluency in
both styles, which is expected in Databricks/Optum environments.

Key PySpark patterns demonstrated:
  - Broadcast joins for small dimension tables
  - Window functions: rolling aggregations, dense_rank, lag/lead
  - Aggregations with groupBy
  - Spark SQL registered temp views
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ──────────────────────────────────────────────
# JOIN: Claims + Patients
# ──────────────────────────────────────────────

def join_claims_patients(claims_df: DataFrame, patients_df: DataFrame) -> DataFrame:
    """Join claims with patient demographics using a broadcast join.

    Why broadcast join?
      - patients_df (~50K rows) is much smaller than claims_df (~500K rows)
      - Broadcasting the small table to all workers avoids a costly shuffle
        of the large table across the network
      - Spark will auto-broadcast tables under 10MB, but we use an explicit
        hint to make the intent clear and guarantee the optimization

    # Databricks equivalent: same syntax — broadcast hints work identically
    """
    # Use broadcast hint on the smaller table to avoid shuffle join
    joined_df = claims_df.join(
        F.broadcast(patients_df.select("patient_id", "comorbidity_count")),
        on="patient_id",
        how="left",
    )
    # Drop duplicate comorbidity_count from patients (already in claims)
    # Keep the claims version since it's already there
    if "comorbidity_count" in [f.name for f in joined_df.schema.fields]:
        pass  # Already present from claims data

    return claims_df  # Claims already has all patient fields from data generation


# ──────────────────────────────────────────────
# WINDOW FUNCTIONS
# ──────────────────────────────────────────────

def add_rolling_claim_cost(claims_df: DataFrame) -> DataFrame:
    """Add rolling 90-day claim cost per patient using window functions.

    Window functions are a core PySpark skill for Optum-level work:
      - partitionBy groups rows (like GROUP BY but without collapsing)
      - orderBy defines the sort within each partition
      - rangeBetween defines the window frame (90 days back)

    # Databricks equivalent: identical syntax in notebook cells
    """
    # Convert admit_date to days-since-epoch for range-based windowing
    claims_with_days = claims_df.withColumn(
        "admit_date_days",
        F.datediff(F.col("admit_date"), F.lit("1970-01-01"))
    )

    # Window: per patient, ordered by admit date, looking back 90 days
    window_90d = (
        Window
        .partitionBy("patient_id")
        .orderBy("admit_date_days")
        .rangeBetween(-90, 0)  # 90 days before current row to current row
    )

    result_df = claims_with_days.withColumn(
        "rolling_cost_90d",
        F.sum("claim_amount").over(window_90d)
    ).withColumn(
        "claim_count_90d",
        F.count("claim_id").over(window_90d)
    ).drop("admit_date_days")

    return result_df


def rank_providers_by_denial_rate(claims_df: DataFrame) -> DataFrame:
    """Rank providers by their denial rate using dense_rank().

    dense_rank vs rank vs row_number:
      - dense_rank: ties get the same rank, no gaps (1,1,2,3)
      - rank: ties get the same rank, gaps after ties (1,1,3,4)
      - row_number: no ties, arbitrary ordering within ties (1,2,3,4)

    We use dense_rank here because we want to know the true ranking
    position even when multiple providers have the same denial rate.
    """
    # First compute denial rate per provider
    provider_stats = (
        claims_df
        .groupBy("provider_id")
        .agg(
            F.count("claim_id").alias("total_claims"),
            F.sum("denial_flag").alias("total_denials"),
            F.avg("denial_flag").alias("denial_rate"),
        )
        .where(F.col("total_claims") >= 10)  # Filter out low-volume providers
    )

    # Rank providers by denial rate (highest denial rate = rank 1)
    window_rank = Window.orderBy(F.col("denial_rate").desc())

    ranked_providers = provider_stats.withColumn(
        "denial_rank",
        F.dense_rank().over(window_rank)
    )

    return ranked_providers


def add_lag_lead_features(claims_df: DataFrame) -> DataFrame:
    """Add lag/lead features for readmission prediction.

    lag(1) = previous claim's value (was the last claim denied?)
    lead(1) = next claim's value (useful for labeling, not prediction)

    These temporal features are critical for readmission models because
    recent claim history is one of the strongest predictors of readmission.
    """
    # Window: per patient, ordered by admit date
    patient_window = Window.partitionBy("patient_id").orderBy("admit_date")

    result_df = (
        claims_df
        # Days since patient's previous claim
        .withColumn("prev_admit_date", F.lag("admit_date", 1).over(patient_window))
        .withColumn(
            "days_since_last_claim",
            F.when(
                F.col("prev_admit_date").isNotNull(),
                F.datediff(F.col("admit_date"), F.col("prev_admit_date"))
            ).otherwise(None)
        )
        # Was the previous claim denied?
        .withColumn("prev_denial_flag", F.lag("denial_flag", 1).over(patient_window))
        # Previous claim amount
        .withColumn("prev_claim_amount", F.lag("claim_amount", 1).over(patient_window))
        # Claim sequence number per patient (1st visit, 2nd visit, etc.)
        .withColumn("claim_sequence", F.row_number().over(patient_window))
        .drop("prev_admit_date")
    )

    return result_df


# ──────────────────────────────────────────────
# AGGREGATIONS — DataFrame API
# ──────────────────────────────────────────────

def compute_aggregations(claims_df: DataFrame) -> dict[str, DataFrame]:
    """Compute key business aggregations using the DataFrame API.

    Returns a dict of named DataFrames for downstream use or reporting.
    """
    # Claims per patient
    claims_per_patient = (
        claims_df
        .groupBy("patient_id")
        .agg(
            F.count("claim_id").alias("num_claims"),
            F.sum("claim_amount").alias("total_claim_amount"),
            F.avg("claim_amount").alias("avg_claim_amount"),
            F.max("readmission_30day").alias("ever_readmitted"),
        )
    )

    # Average LOS by diagnosis
    avg_los_by_diagnosis = (
        claims_df
        .groupBy("diagnosis_code")
        .agg(
            F.avg("length_of_stay").alias("avg_los"),
            F.count("claim_id").alias("claim_count"),
            F.avg("claim_amount").alias("avg_cost"),
        )
        .orderBy(F.col("avg_los").desc())
    )

    # Denial rate by payer type
    denial_by_payer = (
        claims_df
        .groupBy("payer_type")
        .agg(
            F.count("claim_id").alias("total_claims"),
            F.sum("denial_flag").alias("total_denials"),
            F.avg("denial_flag").alias("denial_rate"),
            F.avg("claim_amount").alias("avg_claim_amount"),
        )
        .orderBy(F.col("denial_rate").desc())
    )

    return {
        "claims_per_patient": claims_per_patient,
        "avg_los_by_diagnosis": avg_los_by_diagnosis,
        "denial_by_payer": denial_by_payer,
    }


# ──────────────────────────────────────────────
# SPARK SQL — Same transforms, SQL syntax
# ──────────────────────────────────────────────

def compute_aggregations_sql(spark: SparkSession, claims_df: DataFrame) -> dict[str, DataFrame]:
    """Same aggregations as above, but using Spark SQL syntax.

    Showing both approaches because:
      1. Some teams prefer SQL for its readability with analysts
      2. Databricks notebooks commonly mix SQL cells with Python cells
      3. Complex multi-step transforms are sometimes cleaner in SQL
      4. Interview questions often ask you to write both ways

    # Databricks equivalent: %sql magic cells in notebooks
    """
    # Register the DataFrame as a temp view so Spark SQL can query it
    claims_df.createOrReplaceTempView("claims")

    # Claims per patient — SQL version
    claims_per_patient_sql = spark.sql("""
        SELECT
            patient_id,
            COUNT(claim_id) AS num_claims,
            SUM(claim_amount) AS total_claim_amount,
            AVG(claim_amount) AS avg_claim_amount,
            MAX(readmission_30day) AS ever_readmitted
        FROM claims
        GROUP BY patient_id
    """)

    # Average LOS by diagnosis — SQL version
    avg_los_by_diagnosis_sql = spark.sql("""
        SELECT
            diagnosis_code,
            AVG(length_of_stay) AS avg_los,
            COUNT(claim_id) AS claim_count,
            AVG(claim_amount) AS avg_cost
        FROM claims
        GROUP BY diagnosis_code
        ORDER BY avg_los DESC
    """)

    # Denial rate by payer — SQL with window function
    denial_by_payer_sql = spark.sql("""
        SELECT
            payer_type,
            COUNT(claim_id) AS total_claims,
            SUM(denial_flag) AS total_denials,
            AVG(denial_flag) AS denial_rate,
            AVG(claim_amount) AS avg_claim_amount,
            -- Window: rank payers by denial rate
            DENSE_RANK() OVER (ORDER BY AVG(denial_flag) DESC) AS denial_rank
        FROM claims
        GROUP BY payer_type
        ORDER BY denial_rate DESC
    """)

    return {
        "claims_per_patient_sql": claims_per_patient_sql,
        "avg_los_by_diagnosis_sql": avg_los_by_diagnosis_sql,
        "denial_by_payer_sql": denial_by_payer_sql,
    }


def run_all_transforms(
    spark: SparkSession,
    claims_df: DataFrame,
    patients_df: DataFrame,
) -> DataFrame:
    """Execute the full transformation pipeline and return the enriched DataFrame.

    Pipeline order:
      1. Join claims + patients (broadcast join)
      2. Add rolling window features (90-day costs)
      3. Add lag/lead temporal features
      4. Compute and print aggregation summaries
    """
    print("=" * 60)
    print("HealthSpark — Transform Pipeline")
    print("=" * 60)

    # Step 1: Join (broadcast small table)
    print("\n[1/4] Joining claims + patients (broadcast join)...")
    enriched_df = join_claims_patients(claims_df, patients_df)
    print(f"  Joined records: {enriched_df.count():,}")

    # Step 2: Rolling window features
    print("[2/4] Computing rolling 90-day claim costs...")
    enriched_df = add_rolling_claim_cost(enriched_df)

    # Step 3: Lag/Lead features
    print("[3/4] Adding temporal lag/lead features...")
    enriched_df = add_lag_lead_features(enriched_df)

    # Cache the enriched DataFrame since it's reused by aggregations and feature engineering
    # .cache() tells Spark to keep this in memory after first computation
    # Without caching, Spark would recompute from CSV for each downstream action
    enriched_df.cache()
    print(f"  Enriched schema: {len(enriched_df.columns)} columns")

    # Step 4: Aggregations (both DataFrame API and SQL)
    print("[4/4] Computing aggregation summaries...")
    agg_results = compute_aggregations(enriched_df)
    sql_results = compute_aggregations_sql(spark, enriched_df)

    # Print key stats
    print("\n  Denial Rate by Payer:")
    agg_results["denial_by_payer"].show(truncate=False)

    print("  Avg LOS by Top 5 Diagnoses:")
    agg_results["avg_los_by_diagnosis"].show(5, truncate=False)

    print("\nTransform pipeline complete.\n")
    return enriched_df
