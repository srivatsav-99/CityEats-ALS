# jobs/prepare_explicit_local.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
import os
# Point to your Hadoop folder if you have it; otherwise leave as-is for now
os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ["PATH"] = os.environ.get("PATH","") + os.pathsep + os.environ.get("HADOOP_HOME", r"C:\hadoop") + r"\bin"


spark = SparkSession.builder.getOrCreate()
# Fallback to pure-Java local FS to avoid NativeIO on Windows
spark._jsc.hadoopConfiguration().set("fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
spark._jsc.hadoopConfiguration().set("fs.file.impl.disable.cache", "true")

bronze = r".\data\bronze_reviews.parquet"
silver = r".\data\silver_explicit.parquet"

df = spark.read.parquet(bronze)

# Filter users with >= 3 ratings to reduce cold-start problems
users_ok = (df.groupBy("userId").agg(count("*").alias("n"))
              .where(col("n") >= 3)
              .select("userId"))

df2 = (df.join(users_ok, "userId")
         .select(col("userId").cast("string").alias("user_id"),
                 col("movieId").cast("string").alias("item_id"),
                 col("rating").cast("double").alias("rating"))
         .where(col("rating").isNotNull() & (col("rating") > 0)))

print("rows (silver):", df2.count())
df2.write.mode("overwrite").parquet(silver)
print("wrote silver:", silver)
spark.stop()
