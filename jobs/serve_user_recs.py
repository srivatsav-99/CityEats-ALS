import argparse, json, os, sys
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F
from src.common.spark_utils import get_spark

# ---- Local copy to avoid import cycles; identical signature to train_als_local.py ----
def print_user_recs(spark, model_path, ui_map_path, bi_map_path,
                    user_id, k=10, seen_df=None, format="text", outfile=None):
    import pyspark.sql.functions as F
    import json

    ui = spark.read.parquet(ui_map_path)
    bi = spark.read.parquet(bi_map_path)

    subset = ui.filter(F.col("user_id") == user_id).select("user_idx").limit(1)
    if subset.count() == 0:
        msg = f"[warn] Unknown user_id '{user_id}'. No recommendations."
        print(msg)
        if outfile:
            with open(outfile, "w", encoding="utf-8") as f:
                f.write(msg + "\n")
        return

    model = ALSModel.load(model_path)
    recs = (model.recommendForUserSubset(subset, k)
                 .selectExpr("explode(recommendations) as r")
                 .select(F.col("r.biz_idx").alias("biz_idx"),
                         F.col("r.rating").alias("score"))
                 .join(bi, "biz_idx", "inner")
                 .select("business_id", "score"))

    if seen_df is not None:
        recs = recs.join(
            seen_df.filter(F.col("user_id") == user_id).select("business_id"),
            on="business_id", how="left_anti"
        )

    rows = [(r["business_id"], float(r["score"]))
            for r in recs.orderBy(F.desc("score")).collect()]

    if not rows:
        msg = f"[info] No recs produced for user '{user_id}'."
        print(msg)
        if outfile:
            with open(outfile, "w", encoding="utf-8") as f:
                f.write(msg + "\n")
        return

    # Build output once
    if format == "json":
        payload = {
            "user_id": user_id,
            "k": k,
            "recommendations": [{"business_id": b, "score": s} for b, s in rows],
        }
        out = json.dumps(payload, indent=2)
    elif format == "csv":
        lines = ["user_id,business_id,score"]
        lines += [f"{user_id},{b},{s:.6f}" for b, s in rows]
        out = "\n".join(lines)
    else:
        lines = [f"\nTop-{k} for '{user_id}':"]
        lines += [f"{i:>2}. {b}  (score={s:.4f})" for i, (b, s) in enumerate(rows, 1)]
        out = "\n".join(lines)

    print(out)
    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(out + ("\n" if not out.endswith("\n") else ""))

# -----------------------------------------------------------------------------

def _assert_dir(pth, label):
    if not os.path.isdir(pth):
        sys.exit(f"[serve] missing {label}: {pth}")

def main(args):
    # Decide base run folder
    if args.use_best:
        base = "data/best_run"
    elif args.run_name:
        base = os.path.join(args.run_dir, args.run_name)
    else:
        if os.path.isdir("data/best_run"):
            base = "data/best_run"
            print("[serve] --run-name not provided; defaulting to data/best_run")
        else:
            sys.exit("[serve] Provide --run-name or create data/best_run (run tools/pick_best_run.py)")

    # Artifacts
    model_dir = os.path.join(base, "model")
    ui_dir    = os.path.join(base, "ui_map")
    bi_dir    = os.path.join(base, "bi_map")
    seen_path = os.path.join(base, "seen.parquet")

    # Sanity checks before spinning up Spark jobs
    _assert_dir(model_dir, "model_dir")
    _assert_dir(ui_dir, "ui_map")
    _assert_dir(bi_dir, "bi_map")

    spark = get_spark("Serve-ALS")

    # Optional seen filter
    seen_df = None
    if args.exclude_seen and os.path.isdir(seen_path):
        seen_df = spark.read.parquet(seen_path).select("user_id", "business_id").distinct()

    # Print recs
    print_user_recs(
        spark,
        model_path=model_dir,
        ui_map_path=ui_dir,
        bi_map_path=bi_dir,
        user_id=args.user,
        k=args.k,
        seen_df=seen_df,
        format=args.format,
        outfile=args.outfile,
    )

    spark.stop()

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="data/runs",
                   help="Parent directory containing run folders")
    p.add_argument("--run-name",
                   help="Run folder name under --run-dir (e.g., 'base', 'sweepA'). "
                        "If omitted, defaults to data/best_run/ when present.")
    p.add_argument("--use-best", action="store_true",
                   help="Force serving from data/best_run/")
    p.add_argument("--user", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--exclude-seen", action="store_true",
                   help="Filter out items the user interacted with in TRAIN (if seen.parquet exists)")
    p.add_argument("--outfile", help="Write output to this file (csv|json|text)")
    p.add_argument("--format", choices=["text","json","csv"], default="text")
    main(p.parse_args())
