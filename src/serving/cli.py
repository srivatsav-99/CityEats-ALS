import argparse, json
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ui-map", required=True)
    ap.add_argument("--bi-map", required=True)
    ap.add_argument("--seen", required=False)
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-exclude-seen", action="store_true")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("CityEats-CLI").getOrCreate()
    ui = spark.read.parquet(args.ui_map)   # user_id, user_idx
    bi = spark.read.parquet(args.bi_map)   # business_id, biz_idx
    model = ALSModel.load(args.model-dir if hasattr(args, 'model-dir') else args.model_dir)

    u = ui.filter(F.col("user_id") == args.user_id).select("user_idx")
    if u.limit(1).count() == 0:
        print(json.dumps({"msg": f"unknown user_id {args.user_id}"}))
        spark.stop()
        return

    recs = (model.recommendForUserSubset(u, args.k)
            .selectExpr("explode(recommendations) as r")
            .select(F.col("r.biz_idx").alias("biz_idx"),
                    F.col("r.rating").alias("score"))
            .join(bi, "biz_idx", "inner")
            .select("business_id","score"))

    if args.seen and not args.no_exclude_seen:
        seen = spark.read.parquet(args.seen)
        recs = recs.join(
            seen.filter(F.col("user_id")==args.user_id).select("business_id"),
            "business_id","left_anti"
        )

    out = [{"business_id": r["business_id"], "score": float(r["score"])}
           for r in recs.orderBy(F.desc("score")).collect()]
    print(json.dumps({"user_id": args.user_id, "k": args.k, "items": out}, indent=2))
    spark.stop()

if __name__ == "__main__":
    main()
