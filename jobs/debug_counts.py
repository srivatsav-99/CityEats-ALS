#jobs/debug_counts.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet(r".\data\bronze_reviews.parquet")

print("Rows:", df.count())
for c in ["user_id","item_id","rating","timestamp"]:
    print(c, "exists:", c in df.columns)

df.select(countDistinct("user_id").alias("users"),
          countDistinct("item_id").alias("items")).show()

#showing a tiny sample to verify columns
df.select("user_id","item_id","rating").limit(10).show()
spark.stop()
