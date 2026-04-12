"""
HealthSpark — PySpark MLlib Pipeline
=======================================
End-to-end ML pipeline for 30-day readmission prediction using PySpark MLlib.

Model comparison:
  - Logistic Regression (baseline): fast, interpretable, strong baseline
  - Random Forest (tuned): non-linear, handles mixed features, feature importances

Pipeline stages:
  StringIndexer -> OneHotEncoder -> VectorAssembler -> StandardScaler -> Classifier

Hyperparameter tuning:
  CrossValidator with ParamGridBuilder for Random Forest
  Evaluated with BinaryClassificationEvaluator (AUC-ROC)

Class imbalance:
  Handled via weighted column (inverse class frequency) — readmission is ~15%,
  so positive class gets higher weight to avoid the model ignoring the minority.

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
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
)
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


# ──────────────────────────────────────────────
# Preprocessing stages (shared across models)
# ──────────────────────────────────────────────

# Categorical and numeric column definitions
CAT_COLS = ["facility_type", "insurance_type", "gender", "age_bucket"]
INDEXED_COLS = [f"{c}_idx" for c in CAT_COLS]
OHE_COLS = [f"{c}_ohe" for c in CAT_COLS]

NUMERIC_COLS = [
    "claim_amount", "paid_amount", "length_of_stay", "age",
    "comorbidity_count", "diagnosis_risk_score", "los_vs_expected_ratio",
    "provider_denial_rate", "provider_claim_volume", "payer_approval_rate",
    "patient_avg_claim", "patient_total_claims", "cost_ratio_to_patient_avg",
    "comorbidity_index", "rolling_cost_90d", "claim_count_90d",
    "days_since_last_claim", "prev_denial_flag", "prev_claim_amount",
    "claim_sequence",
]


def _build_preprocessing_stages():
    """Build shared preprocessing stages used by all models.

    Returns indexers, encoder, assembler, scaler, and the list of
    assembler input column names.
    """
    indexers = [
        StringIndexer(inputCol=col, outputCol=idx_col, handleInvalid="keep")
        for col, idx_col in zip(CAT_COLS, INDEXED_COLS)
    ]
    encoder = OneHotEncoder(inputCols=INDEXED_COLS, outputCols=OHE_COLS, dropLast=True)

    all_feature_cols = NUMERIC_COLS + OHE_COLS
    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol="raw_features",
        handleInvalid="skip",
    )
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="scaled_features",
        withMean=False,  # False because sparse vectors don't support mean centering
        withStd=True,
    )
    return indexers, encoder, assembler, scaler


def add_class_weights(df: DataFrame, label_col: str = "readmission_30day") -> DataFrame:
    """Add a weight column to handle class imbalance.

    Weights are inversely proportional to class frequency:
      - Majority class (not readmitted): weight = 1.0
      - Minority class (readmitted): weight = (count_negative / count_positive)

    This forces the model to pay more attention to the underrepresented
    readmission cases, improving recall without oversampling (SMOTE).
    """
    pos_count = df.where(F.col(label_col) == 1).count()
    neg_count = df.where(F.col(label_col) == 0).count()
    balance_ratio = neg_count / max(pos_count, 1)

    df = df.withColumn(
        "class_weight",
        F.when(F.col(label_col) == 1, balance_ratio).otherwise(1.0),
    )
    print(f"  Class balance: {neg_count:,} negative / {pos_count:,} positive (ratio {balance_ratio:.2f})")
    print(f"  Weight for positive class: {balance_ratio:.2f}")
    return df


# ──────────────────────────────────────────────
# Model 1: Logistic Regression (baseline)
# ──────────────────────────────────────────────

def build_lr_pipeline():
    """Build a Logistic Regression pipeline as the interpretable baseline.

    Why LR as baseline?
      - Fast to train (minutes vs. hours for ensembles at scale)
      - Coefficients are directly interpretable (log-odds ratios)
      - Strong baseline: if LR does well, the problem is mostly linear
      - If RF/XGBoost outperforms LR significantly, non-linear patterns exist
    """
    indexers, encoder, assembler, scaler = _build_preprocessing_stages()

    lr = LogisticRegression(
        featuresCol="scaled_features",
        labelCol="readmission_30day",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="raw_prediction",
        weightCol="class_weight",
        maxIter=100,
        regParam=0.01,       # L2 regularization to prevent overfitting
        elasticNetParam=0.1, # Mix of L1 and L2 (0=L2, 1=L1)
    )

    pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, lr])
    return pipeline


def train_logistic_regression(pipeline: Pipeline, train_df: DataFrame) -> tuple:
    """Train the logistic regression baseline model."""
    evaluator = BinaryClassificationEvaluator(
        labelCol="readmission_30day",
        rawPredictionCol="raw_prediction",
        metricName="areaUnderROC",
    )

    start = time.time()
    lr_model = pipeline.fit(train_df)
    elapsed = time.time() - start
    print(f"  Logistic Regression trained in {elapsed:.1f}s")

    return lr_model, evaluator


# ──────────────────────────────────────────────
# Model 2: Random Forest (tuned)
# ──────────────────────────────────────────────

def build_rf_pipeline():
    """Build a Random Forest pipeline with CrossValidator tuning.

    Why Random Forest?
      - Handles mixed feature types well (numeric + one-hot)
      - Robust to outliers (common in claims data)
      - Provides feature importance scores (critical for clinical interpretability)
      - Scales well in distributed Spark (each tree trained on a partition)
    """
    indexers, encoder, assembler, scaler = _build_preprocessing_stages()

    rf = RandomForestClassifier(
        featuresCol="scaled_features",
        labelCol="readmission_30day",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="raw_prediction",
        weightCol="class_weight",
        seed=42,
    )

    pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, rf])
    return pipeline, rf


def tune_and_train_rf(pipeline: Pipeline, rf_stage, train_df: DataFrame) -> tuple:
    """Run hyperparameter tuning with CrossValidator.

    Grid search over:
      - numTrees: [100, 150]   (more trees = better generalization)
      - maxDepth: [8, 10]      (capped at 10 — deeper trees OOM the driver
                                on local[*] when MLlib collects per-node
                                histograms during split finding)
      - maxBins:  [48]         (fixed at 48 — 64+ combined with deep trees
                                and 500k rows kills the 8g driver)

    Total: 2 x 2 x 1 = 4 combos x 3 folds = 12 model fits.
    We deliberately keep RF lightweight here because the real production model
    is the GPU XGBoost downstream (Step 5b) — RF serves as a distributed,
    interpretable ensemble baseline.

    # Databricks equivalent: MLflow autologging captures all runs automatically
    """
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf_stage.numTrees, [100, 150])
        .addGrid(rf_stage.maxDepth, [8, 10])
        .addGrid(rf_stage.maxBins, [48])
        .build()
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol="readmission_30day",
        rawPredictionCol="raw_prediction",
        metricName="areaUnderROC",
    )

    cross_validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        # parallelism=1: train folds sequentially. parallelism=2 doubles peak
        # driver memory (two RFs building histograms at once) and was the root
        # cause of the SparkContext shutdown on Colab.
        parallelism=1,
        seed=42,
    )

    print(f"  Starting CrossValidator ({len(param_grid)} param combos x 3 folds = {len(param_grid)*3} fits)...")
    print("  This may take several minutes...\n")

    start = time.time()
    cv_model = cross_validator.fit(train_df)
    elapsed = time.time() - start
    print(f"  Random Forest tuning complete in {elapsed:.1f}s")

    return cv_model, evaluator


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate_model(model, test_df: DataFrame, model_name: str) -> tuple[dict, DataFrame]:
    """Evaluate a model on the test set with multiple metrics.

    Metrics:
      - AUC-ROC: overall discriminative ability
      - AUC-PR: precision-recall tradeoff (better for imbalanced data)
      - F1 Score: harmonic mean of precision and recall
      - Accuracy: overall correct predictions
      - Precision/Recall: for the positive class (readmitted)
    """
    predictions = model.transform(test_df)

    # AUC-ROC
    auc_roc = BinaryClassificationEvaluator(
        labelCol="readmission_30day", rawPredictionCol="raw_prediction",
        metricName="areaUnderROC",
    ).evaluate(predictions)

    # AUC-PR (more informative than AUC-ROC for imbalanced datasets)
    auc_pr = BinaryClassificationEvaluator(
        labelCol="readmission_30day", rawPredictionCol="raw_prediction",
        metricName="areaUnderPR",
    ).evaluate(predictions)

    # F1, Accuracy, Precision, Recall
    f1 = MulticlassClassificationEvaluator(
        labelCol="readmission_30day", predictionCol="prediction", metricName="f1",
    ).evaluate(predictions)

    accuracy = MulticlassClassificationEvaluator(
        labelCol="readmission_30day", predictionCol="prediction", metricName="accuracy",
    ).evaluate(predictions)

    precision = MulticlassClassificationEvaluator(
        labelCol="readmission_30day", predictionCol="prediction", metricName="weightedPrecision",
    ).evaluate(predictions)

    recall = MulticlassClassificationEvaluator(
        labelCol="readmission_30day", predictionCol="prediction", metricName="weightedRecall",
    ).evaluate(predictions)

    metrics = {
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }

    print(f"\n  {model_name} Metrics:")
    print(f"  {'-' * 40}")
    for metric, value in metrics.items():
        print(f"    {metric:>15}: {value:.4f}")

    return metrics, predictions


def extract_feature_importances(cv_model, model_type: str = "rf") -> list[dict]:
    """Extract and rank feature importances from the best model.

    For RandomForest: mean decrease in impurity (Gini) across all trees.
    For LogisticRegression: absolute coefficient values.
    """
    best_model = cv_model.bestModel if hasattr(cv_model, "bestModel") else cv_model
    classifier = best_model.stages[-1]

    feature_names = list(NUMERIC_COLS) + [f"{c}_ohe" for c in CAT_COLS]

    if model_type == "rf":
        importances = classifier.featureImportances.toArray()
    else:
        importances = [abs(float(v)) for v in classifier.coefficients.toArray()]

    # Pad or truncate names to match vector length
    while len(feature_names) < len(importances):
        feature_names.append(f"feature_{len(feature_names)}")
    feature_names = feature_names[: len(importances)]

    importance_list = [
        {"feature": name, "importance": round(float(imp), 6)}
        for name, imp in zip(feature_names, importances)
    ]
    importance_list.sort(key=lambda x: x["importance"], reverse=True)

    print(f"\n  Top 10 Feature Importances ({model_type.upper()}):")
    for i, feat in enumerate(importance_list[:10]):
        bar = "#" * int(feat["importance"] * 100)
        print(f"  {i+1:>3}. {feat['feature']:<30} {feat['importance']:.4f} {bar}")

    return importance_list


def get_best_params(cv_model) -> dict:
    """Extract the best hyperparameters from the CrossValidator model."""
    best_model = cv_model.bestModel
    rf_model = best_model.stages[-1]
    return {
        "numTrees": rf_model.getNumTrees,
        "maxDepth": rf_model.getOrDefault("maxDepth"),
        "maxBins": rf_model.getOrDefault("maxBins"),
    }


def save_results(
    model_comparison: dict,
    rf_importances: list,
    lr_importances: list,
    best_params: dict,
    output_dir: str,
) -> None:
    """Save all metrics, feature importances, and best params to JSON.

    # Databricks equivalent: mlflow.log_metrics(), mlflow.log_params()
    """
    results = {
        "model_comparison": model_comparison,
        "best_params": best_params,
        "feature_importances": rf_importances[:20],
        "lr_feature_importances": lr_importances[:20],
        # Keep backward-compatible 'metrics' key pointing to best model
        "metrics": model_comparison.get("random_forest", model_comparison.get("logistic_regression", {})),
    }

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "model_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")


# ──────────────────────────────────────────────
# Main pipeline entry point
# ──────────────────────────────────────────────

def run_ml_pipeline(spark: SparkSession, features_df: DataFrame, data_dir: str) -> None:
    """Execute the full ML training pipeline with model comparison.

    Steps:
      1. Prepare data (fill nulls, add class weights, split train/test)
      2. Train Logistic Regression baseline
      3. Train Random Forest with CrossValidator tuning
      4. Compare models on test set
      5. Extract feature importances from both models
      6. Save best model and results
    """
    print("=" * 60)
    print("HealthSpark — ML Training Pipeline")
    print("=" * 60)

    models_dir = os.path.join(data_dir, "models")
    model_path = os.path.join(models_dir, "readmission_model")

    # ── Step 1: Prepare data ──
    print("\n[1/6] Preparing data...")

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

    features_df = features_df.withColumn(
        "readmission_30day", F.col("readmission_30day").cast("integer")
    )

    # Drop columns that the ML Pipeline will recreate (from feature_engineering's
    # build_feature_vector). The Pipeline's own StringIndexer/OHE stages need to
    # produce these from scratch so the fitted stages are saved with the model.
    cols_to_drop = [
        c for c in features_df.columns
        if c.endswith("_idx") or c.endswith("_ohe")
        or c in ("features", "raw_features", "scaled_features")
    ]
    if cols_to_drop:
        features_df = features_df.drop(*cols_to_drop)

    # Add class weights to handle imbalance
    features_df = add_class_weights(features_df)

    # Train/test split: 80/20 with fixed seed for reproducibility
    train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()
    test_df.cache()

    train_count = train_df.count()
    test_count = test_df.count()
    pos_rate = train_df.select(F.avg("readmission_30day")).first()[0]
    print(f"  Train: {train_count:,} rows | Test: {test_count:,} rows")
    print(f"  Positive class rate (train): {pos_rate:.1%}")

    # ── Step 2: Logistic Regression baseline ──
    print("\n[2/6] Training Logistic Regression baseline...")
    lr_pipeline = build_lr_pipeline()
    lr_model, lr_evaluator = train_logistic_regression(lr_pipeline, train_df)

    # ── Step 3: Random Forest with CrossValidator ──
    print("\n[3/6] Building & tuning Random Forest...")
    rf_pipeline, rf_stage = build_rf_pipeline()
    cv_model, rf_evaluator = tune_and_train_rf(rf_pipeline, rf_stage, train_df)

    # ── Step 4: Evaluate both models ──
    print("\n[4/6] Evaluating models on test set...")
    lr_metrics, lr_predictions = evaluate_model(lr_model, test_df, "Logistic Regression")
    rf_metrics, rf_predictions = evaluate_model(cv_model, test_df, "Random Forest")

    model_comparison = {
        "logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
    }

    # Print comparison table
    print(f"\n{'=' * 60}")
    print(f"  MODEL COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<20} {'LR (Baseline)':>15} {'RF (Tuned)':>15}")
    print(f"  {'-' * 50}")
    for metric in lr_metrics:
        lr_val = lr_metrics[metric]
        rf_val = rf_metrics[metric]
        winner = " <--" if rf_val > lr_val else ""
        print(f"  {metric:<20} {lr_val:>15.4f} {rf_val:>15.4f}{winner}")
    print(f"{'=' * 60}")

    # ── Step 5: Feature importances ──
    print("\n[5/6] Extracting feature importances...")
    rf_importances = extract_feature_importances(cv_model, "rf")
    lr_importances = extract_feature_importances(lr_model, "lr")

    # ── Step 6: Save model and results ──
    print("\n[6/6] Saving model and results...")
    best_params = get_best_params(cv_model)
    print(f"  Best RF params: {best_params}")

    # Save the best RF pipeline model (all stages serialized together)
    # Databricks equivalent: mlflow.spark.log_model(cv_model.bestModel, "model")
    cv_model.bestModel.write().overwrite().save(model_path)
    print(f"  Model saved to {model_path}")

    save_results(model_comparison, rf_importances, lr_importances, best_params, models_dir)

    # Unpersist cached DataFrames
    train_df.unpersist()
    test_df.unpersist()

    print("\n" + "=" * 60)
    print("ML Pipeline complete!")
    print("=" * 60)


# ──────────────────────────────────────────────
# Main entry point — run full pipeline end-to-end
# ──────────────────────────────────────────────

def main():
    """Run the entire HealthSpark pipeline: Ingest -> Transform -> Features -> ML."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.utils.spark_session import get_spark_session
    from src.pipeline.ingestion import ingest_all
    from src.pipeline.transforms import run_all_transforms
    from src.pipeline.feature_engineering import engineer_features

    data_dir = os.path.join(project_root, "data")

    spark = get_spark_session(app_name="HealthSpark-ML-Pipeline")

    try:
        claims_df, patients_df = ingest_all(spark, data_dir)
        enriched_df = run_all_transforms(spark, claims_df, patients_df)
        features_df = engineer_features(spark, enriched_df)
        run_ml_pipeline(spark, features_df, data_dir)
    finally:
        spark.stop()
        print("\nSparkSession stopped.")


if __name__ == "__main__":
    main()
