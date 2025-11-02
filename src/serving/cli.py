import argparse, json, os, pathlib
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

import glob
import platform
import pandas as pd

def _read_map_as_spark_df(spark, dir_path: str, cols: list[str]):
    import re
    is_windows = platform.system().lower().startswith("win")


    def _local(p: str) -> str:
        return p.replace("file:///", "") if p.startswith("file:///") else p

    local_dir = _local(dir_path)

    #pick up any *.parquet part rather than "part-*.parquet" only
    parts = sorted(glob.glob(os.path.join(local_dir, "*.parquet")))
    if not parts:
        raise FileNotFoundError(f"No parquet files found under {local_dir}")

    if is_windows:
        #pandas to spark
        pdfs = [pd.read_parquet(p, engine="pyarrow", columns=cols) for p in parts]
        pdf = pd.concat(pdfs, ignore_index=True)
        return spark.createDataFrame(pdf[cols])
    else:
        return spark.read.parquet(*parts).select(*cols)




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

        ui = _read_map_as_spark_df(spark, ui_path, ["user_id", "user_idx"])
        bi = _read_map_as_spark_df(spark, bi_path, ["item_id", "item_idx"])
        model = ALSModel.load(model_path)

        u = ui.filter(F.col("user_id") == args.user_id).select("user_idx")
        if u.limit(1).count() == 0:
            print(json.dumps({"msg": f"unknown user_id {args.user_id}"}))
            return

        # ALS recommendForUserSubset returns a struct with item index and rating.
        # In our bundle/maps: item index column is "item_idx", external id is "item_id".
        recs = (
            model.recommendForUserSubset(u, args.k)
            .selectExpr("explode(recommendations) as r")
            .select(
                F.col("r.item_idx").alias("item_idx"),
                F.col("r.rating").alias("score"),
            )
            .join(bi, "item_idx", "inner")
            .select("item_id", "score")
        )

        if seen_path and not args.no_exclude_seen:
            seen = spark.read.parquet(seen_path)
            recs = recs.join(
                seen.filter(F.col("user_id") == args.user_id).select("item_id"),
                "item_id",
                "left_anti",
            )

        out = [{"item_id": r["item_id"], "score": float(r["score"])}
               for r in recs.orderBy(F.desc("score")).collect()]
        print(json.dumps({"user_id": args.user_id, "k": args.k, "items": out}, indent=2))

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
