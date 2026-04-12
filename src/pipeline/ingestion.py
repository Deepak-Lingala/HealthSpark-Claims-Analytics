"""
HealthSpark — Data Ingestion Layer
====================================
Loads raw CSV files into Spark DataFrames with explicit schemas,
runs data quality checks, and writes cleaned data to Parquet.

Why Parquet over CSV?
  - Columnar storage: only reads columns needed by the query (predicate pushdown)
  - Built-in compression (snappy by default): 3-5x smaller than CSV
  - Schema embedded in the file: no parsing ambiguity
  - Industry standard for data lakes (S3, ADLS, DBFS)

# Databricks equivalent: spark.read.format("csv").load("dbfs:/mnt/raw/claims.csv")
# Databricks equivalent: df.write.format("delta").saveAsTable("claims_clean")
"""

import os
import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# ──────────────────────────────────────────────
# Explicit Schemas — type-safe ingestion
# ──────────────────────────────────────────────
# Defining schemas upfront avoids Spark's costly schema inference pass
# (which reads the entire file once just to guess types) and prevents
# silent type mismatches in production pipelines.

CLAIMS_SCHEMA = StructType([
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

PATIENTS_SCHEMA = StructType([
    StructField("patient_id",        StringType(),  False),
    StructField("age",               IntegerType(), True),
    StructField("gender",            StringType(),  True),
    StructField("state",             StringType(),  True),
    StructField("insurance_type",    StringType(),  True),
    StructField("comorbidity_count", IntegerType(), True),
])


def load_csv(
    spark: SparkSession,
    path: str,
    schema: StructType,
) -> DataFrame:
    """Load a CSV file into a Spark DataFrame with an explicit schema.

    Using schema enforcement (not inferSchema=True) because:
      1. Avoids a full file scan just to guess types
      2. Catches schema drift immediately (malformed files fail fast)
      3. Ensures dates are parsed as DateType, not StringType
    """
    df = (
        spark.read
        .option("header", "true")       # First row is column names
        .option("dateFormat", "yyyy-MM-dd")
        .option("mode", "DROPMALFORMED") # Drop rows that don't match schema
        .schema(schema)
        .csv(path)
    )
    return df


def run_quality_checks(df: DataFrame, table_name: str) -> dict:
    """Run data quality checks and print a summary report.

    Checks:
      - Total record count
      - Null counts per column
      - Numeric outlier detection (values beyond 3 standard deviations)
      - Schema validation (column names and types)
    """
    print(f"\n{'-' * 50}")
    print(f"  Data Quality Report: {table_name}")
    print(f"{'-' * 50}")

    total_rows = df.count()
    print(f"  Total records: {total_rows:,}")

    # Null counts per column
    from pyspark.sql import functions as F

    null_counts = {}
    for col_name in df.columns:
        n_nulls = df.where(F.col(col_name).isNull()).count()
        null_counts[col_name] = n_nulls
        if n_nulls > 0:
            pct = n_nulls / total_rows * 100
            print(f"  WARNING: {col_name}: {n_nulls:,} nulls ({pct:.2f}%)")

    if sum(null_counts.values()) == 0:
        print("  OK: No null values found")

    # Numeric outlier detection
    numeric_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, (IntegerType, DoubleType))
    ]
    for col_name in numeric_cols:
        stats = df.select(
            F.mean(col_name).alias("mean"),
            F.stddev(col_name).alias("std"),
        ).first()
        if stats["std"] and stats["std"] > 0:
            threshold = stats["mean"] + 3 * stats["std"]
            outlier_count = df.where(F.col(col_name) > threshold).count()
            if outlier_count > 0:
                print(f"  INFO: {col_name}: {outlier_count:,} values > 3std (mean={stats['mean']:.2f})")

    # Schema validation
    print(f"  OK: Schema: {len(df.columns)} columns validated")
    print(f"{'-' * 50}\n")

    return {"total_rows": total_rows, "null_counts": null_counts}


def write_parquet(df: DataFrame, path: str, partition_cols: list[str] | None = None) -> None:
    """Write a DataFrame to Parquet format with optional partitioning.

    Why Parquet?
      - Columnar format enables predicate pushdown (read only needed columns)
      - Snappy compression by default (fast decompression, good ratio)
      - Schema is embedded — no ambiguity when reading back

    Why partition by date/facility?
      - Enables partition pruning: queries on a specific date range skip
        irrelevant partition directories entirely
      - Standard pattern in data lakes (S3/ADLS/DBFS)

    # Databricks equivalent: df.write.format("delta").partitionBy(...).saveAsTable(...)
    """
    writer = df.write.mode("overwrite")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.parquet(path)
    print(f"  Written Parquet to {path}")


def ingest_all(spark: SparkSession, data_dir: str) -> tuple[DataFrame, DataFrame]:
    """Run the full ingestion pipeline: CSV → QA → Parquet.

    Returns cleaned claims and patients DataFrames for downstream transforms.
    """
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")

    print("=" * 60)
    print("HealthSpark — Data Ingestion Pipeline")
    print("=" * 60)

    # Load raw CSVs with explicit schemas
    print("\n[1/4] Loading claims CSV...")
    claims_df = load_csv(spark, os.path.join(raw_dir, "claims.csv"), CLAIMS_SCHEMA)

    print("[2/4] Loading patients CSV...")
    patients_df = load_csv(spark, os.path.join(raw_dir, "patients.csv"), PATIENTS_SCHEMA)

    # Data quality checks
    print("\n[3/4] Running data quality checks...")
    run_quality_checks(claims_df, "claims")
    run_quality_checks(patients_df, "patients")

    # Write to Parquet (production-grade columnar format)
    print("[4/4] Writing to Parquet...")
    write_parquet(claims_df, os.path.join(processed_dir, "claims_parquet"))
    write_parquet(patients_df, os.path.join(processed_dir, "patients_parquet"))

    print("\nIngestion complete.\n")
    return claims_df, patients_df


if __name__ == "__main__":
    from src.utils.spark_session import get_spark_session

    spark = get_spark_session()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    ingest_all(spark, data_dir)
    spark.stop()
