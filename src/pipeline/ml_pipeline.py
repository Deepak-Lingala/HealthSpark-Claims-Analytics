"""
HealthSpark — PySpark MLlib Pipeline
=======================================
End-to-end ML pipeline for 30-day readmission prediction using PySpark MLlib.

Pipeline stages:
  StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler → RandomForestClassifier

Hyperparameter tuning:
  CrossValidator with ParamGridBuilder (numTrees × maxDepth × maxBins = 27 combos)
  Evaluated with BinaryClassificationEvaluator (AUC-ROC)

Why PySpark MLlib (not scikit-learn)?
  - Trains on distributed data — scales to billions of rows without sampling
  - Pipeline object serializes all stages together (no train/serve skew)
  - Integrates with Spark's Catalyst optimizer for predicate pushdown
  - Standard at Optum/UHG, Cigna, Humana, and other large healthcare orgs

# Databricks equivalent: MLflow integration for experiment tracking + model registry
"""

import json
import os
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def build_ml_pipeline() -> tuple[Pipeline, list[str], list[str]]:
    """Construct the full MLlib Pipeline with all preprocessing + model stages.

    Returns:
        pipeline: Unfitted Pipeline object
        cat_cols: List of categorical column names
        numeric_cols: List of numeric feature column names
    """
    # ── Categorical columns ──
    cat_cols = ["facility_type", "insurance_type", "gender", "age_bucket"]
    indexed_cols = [f"{c}_idx" for c in cat_cols]
    ohe_cols = [f"{c}_ohe" for c in cat_cols]

    # ── Numeric columns (engineered in feature_engineering.py) ──
    numeric_cols = [
        "claim_amount", "paid_amount", "length_of_stay", "age",
        "comorbidity_count", "diagnosis_risk_score", "los_vs_expected_ratio",
        "provider_denial_rate", "provider_claim_volume", "payer_approval_rate",
        "patient_avg_claim", "patient_total_claims", "cost_ratio_to_patient_avg",
        "comorbidity_index", "rolling_cost_90d", "claim_count_90d",
        "days_since_last_claim", "prev_denial_flag", "prev_claim_amount",
        "claim_sequence",
    ]

    # Stage 1: StringIndexer — convert categorical strings to numeric indices
    indexers = [
        StringIndexer(inputCol=col, outputCol=idx_col, handleInvalid="keep")
        for col, idx_col in zip(cat_cols, indexed_cols)
    ]

    # Stage 2: OneHotEncoder — create sparse binary vectors from indices
    encoder = OneHotEncoder(inputCols=indexed_cols, outputCols=ohe_cols, dropLast=True)

    # Stage 3: VectorAssembler — combine all features into one vector
    all_feature_cols = numeric_cols + ohe_cols
    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol="raw_features",
        handleInvalid="skip",
    )

    # Stage 4: StandardScaler — normalize features for better convergence
    # withMean=False because sparse vectors don't support mean centering
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="scaled_features",
        withMean=False,
        withStd=True,
    )

    # Stage 5: RandomForestClassifier — ensemble of decision trees
    # Why RandomForest?
    #   - Handles mixed feature types well (numeric + one-hot)
    #   - Robust to outliers (common in claims data)
    #   - Provides feature importance scores (critical for clinical interpretability)
    #   - Scales well in distributed Spark (each tree trained on a partition)
    rf = RandomForestClassifier(
        featuresCol="scaled_features",
        labelCol="readmission_30day",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="raw_prediction",
        seed=42,
    )

    # Assemble pipeline: all stages execute in order
    pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, rf])

    return pipeline, cat_cols, numeric_cols


def tune_and_train(
    pipeline: Pipeline,
    train_df: DataFrame,
) -> tuple:
    """Run hyperparameter tuning with CrossValidator.

    Grid search over:
      - numTrees: [50, 100, 200]  (more trees = better generalization, slower)
      - maxDepth: [5, 10, 15]     (deeper = more complex, risk of overfitting)
      - maxBins: [32, 64, 128]    (more bins = finer splits for continuous features)

    Total combinations: 3 × 3 × 3 = 27 parameter sets × 3 folds = 81 model fits

    # Databricks equivalent: MLflow autologging captures all runs automatically
    """
    # Get the RandomForestClassifier stage from the pipeline
    rf_stage = pipeline.getStages()[-1]

    # Build parameter grid
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf_stage.numTrees, [50, 100, 200])
        .addGrid(rf_stage.maxDepth, [5, 10, 15])
        .addGrid(rf_stage.maxBins, [32, 64, 128])
        .build()
    )

    # Evaluator: AUC-ROC for binary classification
    evaluator = BinaryClassificationEvaluator(
        labelCol="readmission_30day",
        rawPredictionCol="raw_prediction",
        metricName="areaUnderROC",
    )

    # CrossValidator: 3-fold cross-validation
    # parallelism=2 runs two parameter combos in parallel on the driver
    cross_validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2,
        seed=42,
    )

    print("  Starting CrossValidator (27 param combos × 3 folds = 81 fits)...")
    print("  This may take several minutes on local mode...\n")

    start_time = time.time()
    cv_model = cross_validator.fit(train_df)
    elapsed = time.time() - start_time

    print(f"  Training complete in {elapsed:.1f} seconds")

    return cv_model, evaluator


def evaluate_model(cv_model, test_df: DataFrame, evaluator) -> dict:
    """Evaluate the best model on the test set with multiple metrics.

    Metrics:
      - AUC-ROC: overall discriminative ability (>0.5 = better than random)
      - F1 Score: harmonic mean of precision and recall
      - Accuracy: overall correct predictions
      - Precision/Recall: for the positive class (readmitted)
    """
    # Get predictions from best model
    predictions = cv_model.transform(test_df)

    # AUC-ROC
    auc_roc = evaluator.evaluate(predictions)

    # F1 Score
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="readmission_30day",
        predictionCol="prediction",
        metricName="f1",
    )
    f1_score = f1_evaluator.evaluate(predictions)

    # Accuracy
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="readmission_30day",
        predictionCol="prediction",
        metricName="accuracy",
    )
    accuracy = accuracy_evaluator.evaluate(predictions)

    # Precision and Recall
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="readmission_30day",
        predictionCol="prediction",
        metricName="weightedPrecision",
    )
    precision = precision_evaluator.evaluate(predictions)

    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="readmission_30day",
        predictionCol="prediction",
        metricName="weightedRecall",
    )
    recall = recall_evaluator.evaluate(predictions)

    metrics = {
        "auc_roc": round(auc_roc, 4),
        "f1_score": round(f1_score, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }

    print(f"\n{'-' * 50}")
    print(f"  Model Evaluation (Test Set)")
    print(f"{'-' * 50}")
    for metric, value in metrics.items():
        print(f"  {metric:>15}: {value:.4f}")
    print(f"{'-' * 50}")

    return metrics, predictions


def extract_feature_importances(cv_model, numeric_cols: list[str], cat_cols: list[str]) -> list[dict]:
    """Extract and rank feature importances from the best RandomForest model.

    Feature importance in RandomForest = mean decrease in impurity (Gini)
    across all trees for each feature.
    """
    # Navigate to the best model's RandomForest stage
    best_model = cv_model.bestModel
    rf_model = best_model.stages[-1]  # Last stage is the RF classifier

    importances = rf_model.featureImportances.toArray()

    # Build feature name list (numeric + OHE categories)
    feature_names = list(numeric_cols)
    for cat in cat_cols:
        feature_names.append(f"{cat}_ohe")

    # Pad or truncate to match importance vector length
    while len(feature_names) < len(importances):
        feature_names.append(f"feature_{len(feature_names)}")
    feature_names = feature_names[:len(importances)]

    # Rank by importance
    importance_list = [
        {"feature": name, "importance": round(float(imp), 6)}
        for name, imp in zip(feature_names, importances)
    ]
    importance_list.sort(key=lambda x: x["importance"], reverse=True)

    print(f"\n  Top 10 Feature Importances:")
    for i, feat in enumerate(importance_list[:10]):
        bar = "#" * int(feat["importance"] * 100)
        print(f"  {i+1:>3}. {feat['feature']:<30} {feat['importance']:.4f} {bar}")

    return importance_list


def get_best_params(cv_model) -> dict:
    """Extract the best hyperparameters from the CrossValidator model."""
    best_model = cv_model.bestModel
    rf_model = best_model.stages[-1]

    params = {
        "numTrees": rf_model.getNumTrees,
        "maxDepth": rf_model.getOrDefault("maxDepth"),
        "maxBins": rf_model.getOrDefault("maxBins"),
    }
    return params


def save_results(metrics: dict, importances: list, best_params: dict, output_dir: str) -> None:
    """Save all metrics, feature importances, and best params to JSON.

    # Databricks equivalent: mlflow.log_metrics(), mlflow.log_params()
    """
    results = {
        "metrics": metrics,
        "best_params": best_params,
        "feature_importances": importances[:20],  # Top 20
    }

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "model_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")


def run_ml_pipeline(spark: SparkSession, features_df: DataFrame, data_dir: str) -> None:
    """Execute the full ML training pipeline.

    Steps:
      1. Prepare data (fill nulls, split train/test)
      2. Build MLlib Pipeline
      3. Tune with CrossValidator
      4. Evaluate on test set
      5. Extract feature importances
      6. Save model and results
    """
    print("=" * 60)
    print("HealthSpark — ML Training Pipeline")
    print("=" * 60)

    models_dir = os.path.join(data_dir, "models")
    model_path = os.path.join(models_dir, "readmission_model")

    # ── Step 1: Prepare data ──
    print("\n[1/6] Preparing data...")

    # Fill nulls in temporal features
    features_df = features_df.fillna({
        "days_since_last_claim": -1,
        "prev_denial_flag": 0,
        "prev_claim_amount": 0.0,
        "rolling_cost_90d": 0.0,
        "claim_count_90d": 0,
        "claim_sequence": 1,
        "provider_denial_rate": 0.0,
        "provider_claim_volume": 0,
        "payer_approval_rate": 0.85,
        "patient_avg_claim": 0.0,
        "patient_total_claims": 1,
        "cost_ratio_to_patient_avg": 1.0,
    })

    # Ensure label column exists and is integer
    features_df = features_df.withColumn(
        "readmission_30day",
        F.col("readmission_30day").cast("integer")
    )

    # Drop columns that the ML Pipeline will recreate (from feature_engineering's
    # build_feature_vector). The Pipeline's own StringIndexer/OHE stages need to
    # produce these from scratch so the fitted stages are saved with the model.
    cols_to_drop = [c for c in features_df.columns
                    if c.endswith("_idx") or c.endswith("_ohe")
                    or c in ("features", "raw_features", "scaled_features")]
    if cols_to_drop:
        features_df = features_df.drop(*cols_to_drop)

    # Train/test split: 80/20 with fixed seed for reproducibility
    train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=42)

    train_count = train_df.count()
    test_count = test_df.count()
    pos_rate = train_df.select(F.avg("readmission_30day")).first()[0]
    print(f"  Train: {train_count:,} rows | Test: {test_count:,} rows")
    print(f"  Positive class rate (train): {pos_rate:.1%}")

    # ── Step 2: Build Pipeline ──
    print("\n[2/6] Building MLlib Pipeline...")
    pipeline, cat_cols, numeric_cols = build_ml_pipeline()
    print(f"  Pipeline stages: {len(pipeline.getStages())}")

    # ── Step 3: CrossValidator tuning ──
    print("\n[3/6] Running hyperparameter tuning...")
    cv_model, evaluator = tune_and_train(pipeline, train_df)

    # ── Step 4: Evaluate ──
    print("\n[4/6] Evaluating on test set...")
    metrics, predictions = evaluate_model(cv_model, test_df, evaluator)

    # ── Step 5: Feature importances ──
    print("\n[5/6] Extracting feature importances...")
    importances = extract_feature_importances(cv_model, numeric_cols, cat_cols)

    # ── Step 6: Save model and results ──
    print("\n[6/6] Saving model and results...")
    best_params = get_best_params(cv_model)
    print(f"  Best params: {best_params}")

    # Save the full pipeline model (all stages: indexers + encoder + scaler + RF)
    # This allows loading a single object for inference — no preprocessing code needed
    # Databricks equivalent: mlflow.spark.log_model(cv_model.bestModel, "model")
    cv_model.bestModel.write().overwrite().save(model_path)
    print(f"  Model saved to {model_path}")

    save_results(metrics, importances, best_params, models_dir)

    print("\n" + "=" * 60)
    print("ML Pipeline complete!")
    print("=" * 60)


# ──────────────────────────────────────────────
# Main entry point — run full pipeline end-to-end
# ──────────────────────────────────────────────

def main():
    """Run the entire HealthSpark pipeline: Ingest → Transform → Features → ML."""
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.utils.spark_session import get_spark_session
    from src.pipeline.ingestion import ingest_all
    from src.pipeline.transforms import run_all_transforms
    from src.pipeline.feature_engineering import engineer_features

    data_dir = os.path.join(project_root, "data")

    # Initialize Spark
    spark = get_spark_session(app_name="HealthSpark-ML-Pipeline")

    try:
        # Stage 1: Ingest raw CSV → Parquet
        claims_df, patients_df = ingest_all(spark, data_dir)

        # Stage 2: Transforms (joins, windows, aggregations)
        enriched_df = run_all_transforms(spark, claims_df, patients_df)

        # Stage 3: Feature engineering
        features_df = engineer_features(spark, enriched_df)

        # Stage 4: ML Pipeline (train, tune, evaluate, save)
        run_ml_pipeline(spark, features_df, data_dir)

    finally:
        spark.stop()
        print("\nSparkSession stopped.")


if __name__ == "__main__":
    main()
