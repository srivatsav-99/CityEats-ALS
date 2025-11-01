import argparse, json, os, pathlib
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

def _abs_norm(p: str) -> str:
    p = pathlib.Path(p).resolve()
    return "file:///" + str(p).replace("\\", "/").lstrip("/")

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

    spark = (
        SparkSession.builder
        .appName("CityEats-CLI")
        #Hadoop conf flip below is the key
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.sql.warehouse.dir", "/tmp")
        .getOrCreate()
    )

    jconf = spark._jsc.hadoopConfiguration()
    jconf.set("fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
    jconf.set("fs.AbstractFileSystem.file.impl", "org.apache.hadoop.fs.local.Local")
    jconf.set("fs.file.impl.disable.cache", "true")
    jconf.set("io.native.lib.available", "false")


  
    jconf = spark._jsc.hadoopConfiguration()
    jconf.set("fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
    jconf.set("fs.AbstractFileSystem.file.impl", "org.apache.hadoop.fs.local.Local")

    try:
        ui_path    = _abs_norm(args.ui_map)
        bi_path    = _abs_norm(args.bi_map)
        model_path = _abs_norm(args.model_dir)
        seen_path  = _abs_norm(args.seen) if args.seen else None

        ui = spark.read.parquet(ui_path)   # expects : user_id, user_idx
        bi = spark.read.parquet(bi_path)   # expects : business_id, biz_idx
        model = ALSModel.load(model_path)

        u = ui.filter(F.col("user_id") == args.user_id).select("user_idx")
        if u.limit(1).count() == 0:
            print(json.dumps({"msg": f"unknown user_id {args.user_id}"}))
            return

        recs = (
            model.recommendForUserSubset(u, args.k)
            .selectExpr("explode(recommendations) as r")
            .select(F.col("r.biz_idx").alias("biz_idx"),
                    F.col("r.rating").alias("score"))
            .join(bi, "biz_idx", "inner")
            .select("business_id", "score")
        )

        if seen_path and not args.no_exclude_seen:
            seen = spark.read.parquet(seen_path)
            recs = recs.join(
                seen.filter(F.col("user_id")==args.user_id).select("business_id"),
                "business_id", "left_anti"
            )

        out = [{"business_id": r["business_id"], "score": float(r["score"])}
               for r in recs.orderBy(F.desc("score")).collect()]
        print(json.dumps({"user_id": args.user_id, "k": args.k, "items": out}, indent=2))

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
