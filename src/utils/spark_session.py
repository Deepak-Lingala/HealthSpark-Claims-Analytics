"""
HealthSpark — SparkSession Factory
====================================
Reusable SparkSession builder with configs for local dev and cluster modes.

In Databricks, you never need this — the `spark` variable is pre-configured.
# Databricks equivalent: spark = SparkSession.builder.getOrCreate()  (auto-provided)
"""

import os

from pyspark.sql import SparkSession

# Set HADOOP_HOME for Windows if winutils.exe exists and env is not already set
_WINUTILS_DIR = "C:/hadoop"
if os.name == "nt" and os.path.isfile(os.path.join(_WINUTILS_DIR, "bin", "winutils.exe")):
    os.environ["HADOOP_HOME"] = _WINUTILS_DIR
    os.environ["hadoop.home.dir"] = _WINUTILS_DIR
    # Prepend hadoop/bin to PATH so hadoop.dll is found by the JVM native loader
    _bin = os.path.join(_WINUTILS_DIR, "bin")
    if _bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")


def get_spark_session(
    app_name: str = "HealthSpark",
    master: str = "local[*]",
    shuffle_partitions: int = 8,
    enable_hive: bool = False,
) -> SparkSession:
    """Create or retrieve a SparkSession with production-tuned defaults.

    Args:
        app_name: Application name visible in the Spark UI.
        master: Spark master URL. Use 'local[*]' for development,
                'spark://host:7077' for standalone cluster.
        shuffle_partitions: Number of partitions after shuffle operations.
            Default 200 is too high for local mode — we use 8 for faster dev.
            In production Databricks clusters, use 200+ based on data volume.
        enable_hive: Enable Hive metastore support for Spark SQL tables.
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        # Reduce shuffle partitions for local dev (default 200 is for large clusters)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        # Enable adaptive query execution — Spark auto-tunes joins and partitions at runtime
        # Databricks equivalent: enabled by default on DBR 7.3+
        .config("spark.sql.adaptive.enabled", "true")
        # Broadcast join threshold: tables under 10MB are broadcast to all workers
        # avoiding expensive shuffle joins for small dimension tables
        .config("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB
        # Arrow optimization for toPandas() — critical for notebook plotting
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # Limit driver memory for local mode (increase on cluster)
        .config("spark.driver.memory", "4g")
        # Windows: tell the JVM where to find hadoop.dll for native IO
        .config("spark.driver.extraJavaOptions",
                f"-Djava.library.path=C:/hadoop/bin -Dhadoop.home.dir=C:/hadoop")
    )

    if enable_hive:
        builder = builder.enableHiveSupport()

    spark = builder.getOrCreate()

    # Set log level to reduce noise in local dev
    spark.sparkContext.setLogLevel("WARN")

    return spark
