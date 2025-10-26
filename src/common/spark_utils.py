from pyspark.sql import SparkSession
import os

def get_spark(app_name: str = "CityEats-CA"):
    # Normalize and validate Hadoop home
    HADOOP_DIR = os.path.abspath(r"C:\hadoop")  # ensure absolute + canonical
    if not os.path.isdir(HADOOP_DIR):
        # Soft warning; Spark will still run with RawLocalFS but warns about winutils
        print(f"[warn] HADOOP_HOME not found at {HADOOP_DIR}. "
              f"Create it and (optionally) put winutils.exe under {HADOOP_DIR}\\bin to silence warnings.")

    os.environ.setdefault("HADOOP_HOME", HADOOP_DIR)
    os.environ.setdefault("hadoop.home.dir", HADOOP_DIR)

    bin_path = os.path.join(HADOOP_DIR, "bin")
    if os.path.isdir(bin_path) and bin_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")

    spark = (
        SparkSession.builder
        .appName(app_name)
        # Tell JVM explicitly
        .config("spark.driver.extraJavaOptions", f"-Dhadoop.home.dir={HADOOP_DIR}")
        .config("spark.executor.extraJavaOptions", f"-Dhadoop.home.dir={HADOOP_DIR}")
        # Local/dev settings
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.python.worker.reuse", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    # Avoid NativeIO on Windows (keep ALSModel.load() FS flip in train_als_local.py)
    jconf = spark._jsc.hadoopConfiguration()
    jconf.set("fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
    jconf.set("fs.file.impl.disable.cache", "true")
    return spark
