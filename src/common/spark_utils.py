from pyspark.sql import SparkSession
import os

def get_spark(app_name: str = "CityEats-CA"):
    # Use forward slashes so JVM sees an absolute path on Windows
    HADOOP_DIR = "C:/hadoop"

    os.environ.setdefault("HADOOP_HOME", HADOOP_DIR)
    os.environ.setdefault("hadoop.home.dir", HADOOP_DIR)

    return (
        SparkSession.builder
        .appName(app_name)
        # also pass to JVM in case env vars are missed
        .config("spark.driver.extraJavaOptions", "-Dhadoop.home.dir=C:/hadoop")
        .config("spark.executor.extraJavaOptions", "-Dhadoop.home.dir=C:/hadoop")

        .config("spark.python.worker.reuse", "true")  # reuse Python workers
        .config("spark.sql.shuffle.partitions", "4")  # fewer partitions for small/local data
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")  # disable Arrow for safety
        .config("spark.ui.showConsoleProgress", "false")  # cleaner logs
        .getOrCreate()
    )
