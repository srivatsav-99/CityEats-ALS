import argparse, json, os
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

def _abspath(p: str) -> str:
    #Spark on Windows is happiest with absolute local paths
    return os.path.abspath(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, dest="model_dir")
    ap.add_argument("--ui-map",    required=True, dest="ui_map")
    ap.add_argument("--bi-map",    required=True, dest="bi_map")
    ap.add_argument("--seen",      required=False)
    ap.add_argument("--user-id",   required=True, dest="user_id")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-exclude-seen", action="store_true")
    args = ap.parse_args()

    #windows safe Spark session
    spark = (
        SparkSession.builder
        .appName("CityEats-CLI")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.sql.warehouse.dir", "/tmp")
        .getOrCreate()
    )

    try:
        ui_path = _abspath(args.ui_map)
        bi_path = _abspath(args.bi_map)
        model_path = _abspath(args.model_dir)

        ui = spark.read.parquet(ui_path)  #expects : user_id, user_idx
        bi = spark.read.parquet(bi_path)  #expects : business_id, biz_idx
        model = ALSModel.load(model_path)

        # get this user's internal index
        u = ui.filter(F.col("user_id") == args.user_id).select("user_idx")
        if u.limit(1).count() == 0:
            print(json.dumps({"msg": f"unknown user_id {args.user_id}"}))
            return

        recs = (
            model.recommendForUserSubset(u, args.k)
            .selectExpr("explode(recommendations) as r")
            .select(
                F.col("r.biz_idx").alias("biz_idx"),
                F.col("r.rating").alias("score")
            )
            .join(bi, "biz_idx", "inner")
            .select("business_id", "score")
        )

        if args.seen and not args.no_exclude_seen:
            seen = spark.read.parquet(_abspath(args.seen))
            recs = recs.join(
                seen.filter(F.col("user_id") == args.user_id).select("business_id"),
                "business_id",
                "left_anti",
            )

        out = [
            {"business_id": r["business_id"], "score": float(r["score"])}
            for r in recs.orderBy(F.desc("score")).collect()
        ]
        print(json.dumps({"user_id": args.user_id, "k": args.k, "items": out}, indent=2))

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
