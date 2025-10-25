# jobs/ingest_movielens_local.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

src = r".\data\movielens\ml-20m\ratings.csv"       # CSV with headers: userId,movieId,rating,timestamp
dst = r".\data\bronze_reviews.parquet"

df = (spark.read.option("header", True).csv(src)
      .select(col("userId").cast("int"),
              col("movieId").cast("int"),
              col("rating").cast("double")))

print("rows (raw):", df.count())
df.write.mode("overwrite").parquet(dst)
print("wrote bronze:", dst)
spark.stop()
