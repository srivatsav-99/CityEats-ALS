#jobs/score_als_local.py
import argparse
from pyspark.sql import functions as F
from src.common.spark_utils import get_spark
from pyspark.ml.recommendation import ALSModel

def main(a):
    spark = get_spark("ALS-Score")
    ui = spark.read.parquet(a.out + "_ui_map")
    bi = spark.read.parquet(a.out + "_bi_map")
    model = ALSModel.load(a.out + "_als_model")

    k = a.k
    recs = model.recommendForAllUsers(k)  #user_idx, recommendations(array<struct<biz_idx,rating>>)

    #explode & map back to ids
    recs = (recs.select("user_idx", F.posexplode("recommendations").alias("rank0", "r"))
                .select("user_idx",
                        (F.col("rank0")+1).alias("rank"),
                        F.col("r.biz_idx").alias("biz_idx"),
                        F.col("r.rating").alias("score"))
                .join(ui, "user_idx", "inner")
                .join(bi, "biz_idx", "inner")
                .select("user_id", "business_id", "score", "rank"))

    #optionally excluding seen
    if a.exclude_seen:
        df = spark.read.parquet(a.in_path).select("user_id","business_id")
        recs = recs.join(df, ["user_id","business_id"], how="left_anti")

    #save
    (recs.orderBy("user_id","rank")
         .write.mode("overwrite")
         .option("header", "true")
         .csv(a.out + "_recs_csv") if a.format=="csv"
     else (recs.write.mode("overwrite").json(a.out + "_recs_json")))

    print(f"[info] wrote batch recs -> {a.out}_{'recs_csv' if a.format=='csv' else 'recs_json'}")
    spark.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="same reviews parquet used for training")
    p.add_argument("--out", dest="out", required=True, help="base path of saved ALS artifacts")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--exclude-seen", action="store_true")
    p.add_argument("--format", choices=["json","csv"], default="json")
    main(p.parse_args())
