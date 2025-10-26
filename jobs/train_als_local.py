import argparse, yaml, random
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F, Window
from pyspark.sql import types as T
from pyspark.sql.functions import broadcast
from src.common.spark_utils import get_spark
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALSModel
import json, os, time

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

    # Build output string once
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
        lines = [f"\nTop-{k} recommendations for user '{user_id}':"]
        lines += [f"{i:>2}. {b}  (score={s:.4f})" for i, (b, s) in enumerate(rows, 1)]
        out = "\n".join(lines)

    # Print and/or write
    print(out)
    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(out + ("\n" if not out.endswith("\n") else ""))




def precision_at_k(pred, truth, k=10):
    # pred: user_id, recs (array<struct<business_id:string, rating:double>>)
    # truth: user_id, business_id (held-out positives)
    exploded = pred.select("user_id", F.expr(f"slice(recommendations, 1, {k}) as topk"))
    exploded = exploded.select("user_id", F.explode("topk.business_id").alias("business_id"))
    joined = exploded.join(truth, on=["user_id","business_id"], how="left_semi")
    prec = joined.groupBy().agg((F.count("*")/F.countDistinct("user_id")).alias("precision_at_k")).collect()[0][0]
    return prec



def main(args):
    spark = get_spark("ALS-Local")
    spark.conf.set("spark.sql.codegen.wholeStage", "false")
    #spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    spark.sparkContext.setLogLevel("ERROR")  #quieter logs
    with open("conf/config.yaml","r") as f:
        conf = yaml.safe_load(f)

    # evaluation params
    k_cfg = conf["params"]["eval"]["k"]
    pos_thresh = conf["params"]["eval"]["pos_thresh"]  #updated to read from yaml rather than hard coding

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or ts
    run_dir = os.path.join(args.run_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Derive paths for all artifacts
    model_dir = os.path.join(run_dir, "model")
    ui_dir = os.path.join(run_dir, "ui_map")
    bi_dir = os.path.join(run_dir, "bi_map")
    recs_dir = os.path.join(run_dir, "recs_json")
    metrics_p = os.path.join(run_dir, "metrics.json")
    leaderboard_path = os.path.join(run_dir, "leaderboard.csv")

    batch_dir = os.path.join(run_dir, "batch_csv")  # optional

    # load
    #BEFORE
    #df = spark.read.parquet(args.in_path).select("user_id","business_id","stars").dropna()

    # AFTER — handle both schemas
    raw = spark.read.parquet(args.in_path)
    cols = set(raw.columns)

    if {"user_id", "business_id", "stars"}.issubset(cols):
        df = raw.select("user_id", "business_id", "stars").dropna()
    elif {"user_id", "item_id", "rating"}.issubset(cols):
        df = raw.selectExpr(
            "cast(user_id as int) as user_id",
            "cast(item_id as int) as business_id",
            "cast(rating as double) as stars"
        ).dropna()
    else:
        raise ValueError(f"Input schema mismatch. Found columns: {raw.columns}")

    # integer indices for ALS
    #u_index = df.select("user_id").distinct()\
    #    .withColumn("user_idx", F.row_number().over(Window.orderBy("user_id")) - 1)
    #b_index = df.select("business_id").distinct()\
    #    .withColumn("biz_idx", F.row_number().over(Window.orderBy("business_id")) - 1)
    #dfi = df.join(u_index, "user_id").join(b_index, "business_id")
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx_d", handleInvalid="skip")
    biz_indexer = StringIndexer(inputCol="business_id", outputCol="biz_idx_d", handleInvalid="skip")

    ui_model = user_indexer.fit(df)
    bi_model = biz_indexer.fit(df)

    dfi = (ui_model.transform(df)
           .transform(lambda x: bi_model.transform(x))
           .withColumn("user_idx", F.col("user_idx_d").cast("int"))
           .withColumn("biz_idx", F.col("biz_idx_d").cast("int"))
           .drop("user_idx_d", "biz_idx_d"))

    dfi = dfi.persist()

    # --- NEW: keep only users with enough history to ensure non-empty truth/preds ---
    user_stats = dfi.groupBy("user_idx").agg(F.count("*").alias("n"))
    dfi = dfi.join(user_stats.where(F.col("n") >= 5).select("user_idx"), "user_idx", "inner")
    # --- end NEW ---

    # cap k for tiny data
    num_items = dfi.select("biz_idx").distinct().count()
    k = int(min(args.k or k_cfg, max(1, num_items)))

    # split by users (so test users also appear in train)

    if args.split_mode == "peruser":
        # (original behavior; OK for small data, costly for ML-20M)
        w = Window.partitionBy("user_idx").orderBy(F.rand(args.split_seed))
        dfi_ranked = dfi.withColumn("rn", F.row_number().over(w))

        base_train = dfi_ranked.filter(F.col("rn") == 1)

        rest = dfi_ranked.filter(F.col("rn") > 1).withColumn("r", F.rand(args.split_seed + 1))
        train = base_train.unionByName(rest.filter(F.col("r") <= 0.8).drop("r")).drop("rn")
        test = rest.filter(F.col("r") > 0.8).drop("r", "rn")

        if test.limit(1).count() == 0:
            if rest.limit(1).count() > 0:
                sample = rest.orderBy(F.rand(args.split_seed + 2)).limit(1)
                test = sample
                train = train.join(sample.select("user_idx", "biz_idx"),
                                   on=["user_idx", "biz_idx"], how="left_anti")
    else:
        # FAST global random split (recommended for ML-20M)
        splits = dfi.randomSplit([args.global_split, 1.0 - args.global_split], seed=args.split_seed)
        train, test = splits[0], splits[1]

    als = ALS(
        userCol="user_idx",
        itemCol="biz_idx",
        ratingCol="stars",
        # allow negative factors → often lifts recall on explicit ratings
        nonnegative=False,
        coldStartStrategy=conf["params"]["als"]["coldStartStrategy"],
        rank=32,  # ↑ capacity
        regParam=0.02,  # ↓ lighter regularization
        maxIter=15,  # a few more iters
        numUserBlocks=20,
        numItemBlocks=20,
        checkpointInterval=2,
        seed=1,
    )

    # mapping tables (need these for sweep as well)
    ui = dfi.select("user_id", "user_idx").distinct()
    bi = dfi.select("business_id", "biz_idx").distinct()

    # truth mapped back to string IDs (thresholded)
    truth = (
        test.filter(F.col("stars") >= pos_thresh)
        .select("user_idx", "biz_idx")
        .join(ui, "user_idx", "inner")
        .join(bi, "biz_idx", "inner")
        .select("user_id", "business_id")
    )

    # items each user already interacted with in TRAIN (string IDs)
    seen = (
        train.select("user_idx", "biz_idx")
        .join(ui, "user_idx")
        .join(bi, "biz_idx")
        .select("user_id", "business_id")
        .distinct()
    )

    if args.sweep:
        import itertools, csv

        ranks = [8, 12, 16]  # was [8, 16, 32]
        regs = [0.05, 0.1]  # was [0.01, 0.05, 0.1]
        iters = [10]  # was [10, 20]

        lb_rows, best, best_model = [], None, None

        # precompute once
        truth_set = truth.groupBy("user_id").agg(F.collect_set("business_id").alias("truth_set"))

        for rnk, reg, it in itertools.product(ranks, regs, iters):
            tmp_model = ALS(
                userCol="user_idx", itemCol="biz_idx", ratingCol="stars",
                nonnegative=True,
                coldStartStrategy=conf["params"]["als"]["coldStartStrategy"],
                rank=rnk, regParam=reg, maxIter=it,
                numUserBlocks=200,
                numItemBlocks=200,
                checkpointInterval=2,
                seed=1
            ).fit(train)

            # recs -> explode -> map to business_id and user_id
            tmp_recs = tmp_model.recommendForAllUsers(k)
            tmp_exploded = (
                tmp_recs
                .select("user_idx", F.explode("recommendations").alias("r"))
                .select("user_idx", F.col("r.biz_idx").alias("biz_idx"), F.col("r.rating").alias("pred"))
            )

            tmp_candidates = (
                tmp_exploded
                .join(bi, "biz_idx", "inner")
                .join(ui, "user_idx", "inner")
                .select("user_id", "business_id", "pred")
            )

            # drop seen, build top-k lists
            tmp_recs_topk = (
                tmp_candidates.join(seen, ["user_id", "business_id"], "left_anti")
                .groupBy("user_id")
                .agg(
                    F.reverse(
                        F.array_sort(
                            F.collect_list(F.struct(F.col("pred"), F.col("business_id")))
                        )
                    ).alias("arr")
                )
                .select("user_id", F.expr(f"transform(slice(arr, 1, {k}), x -> x.business_id)").alias("pred_topk"))
            )

            # precision / recall
            tmp_per_user = (
                tmp_recs_topk.join(truth_set, "user_id", "inner")
                .select(
                    "user_id",
                    F.size(F.array_intersect(F.col("pred_topk"), F.col("truth_set"))).alias("hits"),
                    F.size(F.col("truth_set")).alias("truth_size"),
                )
                .withColumn("prec", F.col("hits") / F.lit(k))
                .withColumn("recall",
                            F.when(F.col("truth_size") > 0, F.col("hits") / F.col("truth_size")).otherwise(F.lit(0.0)))
            )
            tmp_p = float(tmp_per_user.agg(F.mean("prec")).first()[0] or 0.0)
            tmp_r = float(tmp_per_user.agg(F.mean("recall")).first()[0] or 0.0)

            # NDCG@K (JVM-only)
            ndcg_base = tmp_recs_topk.join(truth_set, "user_id", "inner")
            dcg_df = ndcg_base.select(
                "user_id",
                F.aggregate(
                    F.transform(
                        F.sequence(F.lit(0), F.size(F.col("pred_topk")) - F.lit(1)),
                        lambda i: F.when(
                            F.array_contains(F.col("truth_set"), F.element_at(F.col("pred_topk"), i + 1)),
                            F.lit(1.0) / F.log2(i + F.lit(2.0))
                        ).otherwise(F.lit(0.0))
                    ),
                    F.lit(0.0),
                    lambda acc, x: acc + x
                ).alias("dcg"),
                F.least(F.size(F.col("truth_set")), F.lit(k)).alias("ideal_hits")
            )
            idcg_df = dcg_df.select(
                "user_id", "dcg", "ideal_hits",
                F.when(
                    F.col("ideal_hits") > 0,
                    F.aggregate(
                        F.sequence(F.lit(0), F.col("ideal_hits") - F.lit(1)),
                        F.lit(0.0),
                        lambda acc, i: acc + (F.lit(1.0) / F.log2(i + F.lit(2.0)))
                    )
                ).otherwise(F.lit(0.0)).alias("idcg")
            )
            ndcg_mean = float(
                idcg_df.select(
                    F.when(F.col("idcg") > 0, F.col("dcg") / F.col("idcg")).otherwise(F.lit(0.0)).alias("ndcg")
                ).agg(F.mean("ndcg")).first()[0] or 0.0
            )

            row = {"rank": rnk, "regParam": reg, "maxIter": it, "precision_at_k": tmp_p, "recall_at_k": tmp_r,
                   "ndcg_at_k": ndcg_mean}
            lb_rows.append(row)
            if best is None or (ndcg_mean, tmp_p) > (best["ndcg_at_k"], best["precision_at_k"]):
                best, best_model = row, tmp_model

        # save leaderboard + best model
        lb_path = leaderboard_path
        with open(lb_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(lb_rows[0].keys()))
            w.writeheader()
            w.writerows(lb_rows)
        print(f"[sweep] wrote {lb_path}")
        best_model.write().overwrite().save(model_dir)
        print(f"[sweep] best by NDCG@K -> rank={best['rank']} reg={best['regParam']} iters={best['maxIter']}")

        # use best model downstream
        model = best_model
    else:
        if args.eval_only:
            # --- Windows-only workaround: flip FS for model load ---
            jconf = spark._jsc.hadoopConfiguration()
            prev_fs = jconf.get("fs.file.impl")  # remember current setting
            # DefaultParamsReader expects LocalFileSystem when reading metadata
            jconf.set("fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
            try:
                model = ALSModel.load(model_dir)
            finally:
                # switch back to RawLocalFileSystem so parquet IO still works on Windows
                jconf.set("fs.file.impl", prev_fs or "org.apache.hadoop.fs.RawLocalFileSystem")
            # --- end workaround ---
        else:
            model = als.fit(train)
            model.write().overwrite().save(model_dir)

    #model = als.fit(train)

    #model.write().overwrite().save(args.out_path + "_als_model")

    # mapping tables

    ui.write.mode("overwrite").parquet(ui_dir)
    bi.write.mode("overwrite").parquet(bi_dir)

    # Optional: print top-K for a specific user from disk artifacts

    seen_for_cli = (train.select("user_idx", "biz_idx")
                    .join(ui, "user_idx")
                    .join(bi, "biz_idx")
                    .select("user_id", "business_id")
                    .distinct())

    # persist seen items for serving
    seen_for_cli.write.mode("overwrite").parquet(os.path.join(run_dir, "seen.parquet"))

    if args.user_id:
        print_user_recs(
            spark,
            model_path=model_dir,
            ui_map_path=ui_dir,
            bi_map_path=bi_dir,
            user_id=args.user_id,
            k=k,
            seen_df=seen_for_cli if args.exclude_seen else None,
            format=args.format,  # <- add this
            outfile=args.outfile,
        )

    # recommendations: [user_idx, recommendations(array<struct<biz_idx, rating>>)]
    recs = model.recommendForAllUsers(k)

    # truth mapped back to string IDs with ONE consistent threshold


    # convert recs to string IDs with scores
    recs_ids = recs.join(ui, on="user_idx", how="inner")\
                   .select("user_id", "recommendations")

    exploded = (
        recs_ids
        .select("user_id", F.explode("recommendations").alias("r"))
        .select("user_id",
                F.col("r.biz_idx").alias("biz_idx"),
                F.col("r.rating").alias("pred"))
    )

    recs_mapped = (
        exploded.join(bi, on="biz_idx", how="inner")
                .select("user_id", "business_id", "pred")
                .groupBy("user_id")
                .agg(F.collect_list(F.struct(F.col("business_id"), F.col("pred")))
                     .alias("recommendations"))
    )

    biz_df = None
    if args.biz:
        if not os.path.exists(args.biz):
            print(f"[warn] --biz path not found: {args.biz}. Skipping business metadata join.")
        else:
            biz_df = spark.read.parquet(args.biz)
            biz_df = biz_df.withColumn(
                "categories_arr",
                F.filter(
                    F.transform(
                        F.split(F.coalesce(F.col("categories"), F.lit("")), r",\s*"),
                        lambda x: F.lower(F.trim(x))
                    ),
                    lambda x: x != F.lit("")
                )
            )

    # Flatten per-user recs into (user_id, business_id, pred)
    flat_recs = exploded.join(bi, "biz_idx", "inner") \
        .select("user_id", "business_id", "pred")

    enriched = flat_recs
    if biz_df is not None:
        enriched = enriched.join(biz_df, on="business_id", how="left")

    # Filters
    if args.min_score is not None:
        enriched = enriched.filter(F.col("pred") >= F.lit(args.min_score))

    if args.city:
        city_key = args.city.strip().lower()
        if "city" in enriched.columns:
            enriched = enriched.filter(F.lower(F.col("city")) == F.lit(city_key))

    if args.categories:
        keys = [k.strip().lower() for k in args.categories.split(",") if k.strip()]
        if keys:
            arr = F.array([F.lit(k) for k in keys])
            enriched = enriched.filter(F.array_overlap(F.col("categories_arr"), arr))

    # Nice ordering
    enriched = enriched.orderBy(F.desc("pred"))

    # ----- Batch CSV export (optional) -----
    batch_out = None if args.disable_batch else (args.batch_out_csv or batch_dir)

    if batch_out:
        # choose columns to export; add/remove as you like
        export_cols = ["user_id", "business_id", "pred"]
        # include metadata columns when available
        if "name" in enriched.columns:  export_cols.append("name")
        if "city" in enriched.columns:  export_cols.append("city")
        if "categories_arr" in enriched.columns:
            # convert ARRAY<STRING> -> STRING so CSV can handle it
            enriched = enriched.withColumn("categories_csv", F.concat_ws("|", F.col("categories_arr")))
            export_cols.append("categories_csv")

        if args.per_user_topk:
            from pyspark.sql import Window as W
            w_top = W.partitionBy("user_id").orderBy(F.desc("pred"))
            enriched = (enriched
                        .withColumn("rn", F.row_number().over(w_top))
                        .filter(F.col("rn") <= args.per_user_topk)
                        .drop("rn"))

        final_sel = [F.col(c) if c != "categories_csv" else F.col("categories_csv").alias("categories")
                     for c in export_cols]

        (enriched
         .select(*final_sel)
         # .coalesce(1)   # Consider removing on ML-20M to avoid memory pressure
         .write
         .mode("overwrite")
         .option("header", "true")
         .csv(batch_out))

        print(f"[batch] wrote CSV folder at {batch_out}")

    # ----- Precision@K & Recall@K -----

    # drop seen items from top-k before computing P@K
    cand = exploded.join(bi, on="biz_idx", how="inner")  # [user_id, business_id, pred]
    if args.exclude_seen:
        cand = cand.join(seen, on=["user_id", "business_id"], how="left_anti")

    # --- NEW: popularity blend ---
    pop = (train
           .groupBy("biz_idx")
           .agg(F.count("*").alias("cnt"))
           .withColumn("pop_score", F.log1p(F.col("cnt"))))
    cand_scored = (cand
                   .join(pop.select("biz_idx", "pop_score"), on="biz_idx", how="left")
                   .fillna({"pop_score": 0.0})
                   .withColumn("final_pred", F.col("pred") + F.lit(args.pop_alpha) * F.col("pop_score")))
    # Use final_pred for ranking
    rank_col = F.col("final_pred") if args.pop_alpha and args.pop_alpha > 0 else F.col("pred")
    # --- end NEW ---

    source = cand_scored if (args.pop_alpha and args.pop_alpha > 0) else cand
    recs_topk = (
        source.groupBy("user_id")
        .agg(
            F.reverse(
                F.array_sort(
                    F.collect_list(F.struct(rank_col.alias("score"), F.col("business_id")))
                )
            ).alias("arr")
        )
        .select("user_id",
                F.expr(f"transform(slice(arr, 1, {k}), x -> x.business_id)").alias("pred_topk"))
    )

    truth_set = truth.groupBy("user_id").agg(F.collect_set("business_id").alias("truth_set"))

    if 0.0 < args.metrics_sample_frac < 1.0:
        sampled_users = truth.select("user_id").distinct() \
            .sample(False, args.metrics_sample_frac, args.split_seed)
        recs_topk = recs_topk.join(sampled_users, "user_id", "inner")
        truth_set = truth_set.join(sampled_users, "user_id", "inner")

    # keep truth_set (or at least its size) in the frame
    per_user = (
        recs_topk.join(truth_set, on="user_id", how="inner")
        .filter(F.size("truth_set") > 0)
        .filter(F.size("pred_topk") > 0)
        .select(
            "user_id",
            F.size(F.array_intersect(F.col("pred_topk"), F.col("truth_set"))).alias("hits"),
            F.size(F.col("truth_set")).alias("truth_size"),
            F.size("pred_topk").alias("pred_size")
        )
        .withColumn("prec", F.col("hits") / F.least(F.lit(k), F.col("pred_size")))  # precision denom = min(k, |pred|)
        .withColumn("recall", F.col("hits") / F.col("truth_size"))
        .cache()
    )

    # ---- DEBUG: inspect a few users' recs vs truth
    sample_dbg = (
        recs_topk
        .join(truth_set, "user_id", "inner")
        .select(
            "user_id",
            F.col("pred_topk"),
            F.col("truth_set"),
            F.size(F.array_intersect(F.col("pred_topk"), F.col("truth_set"))).alias("hits"),
            F.size("truth_set").alias("truth_sz")
        )
        .orderBy(F.desc("hits"), F.desc("truth_sz"))
        .limit(10)
        .toPandas()
    )
    print("[debug] top-10 users by hits:\n", sample_dbg)

    users_evalled_count = per_user.count()
    users_with_hit_count = per_user.filter(F.col("hits") > 0).count()
    mean_hits = per_user.select(F.avg("hits")).first()[0] or 0.0

    print("[metrics] users_evalled:", users_evalled_count)
    print("[metrics] users_with_any_hit:", users_with_hit_count)
    print("[metrics] mean_hits_per_user:", f"{float(mean_hits):.6f}")

    if test.filter(F.col("stars") >= pos_thresh).limit(1).count() == 0:
        print(f"No positives in test at threshold >= {pos_thresh}. Skipping P@{k}/R@{k}.")
        p_at_k = r_at_k = 0.0

    else:
        metrics = per_user.agg(F.mean("prec").alias("p"), F.mean("recall").alias("r")).first()
        p_at_k = 0.0 if metrics is None or metrics["p"] is None else float(metrics["p"])
        r_at_k = 0.0 if metrics is None or metrics["r"] is None else float(metrics["r"])
        denom_precision = per_user.select(F.sum(F.when(F.col("pred_size") > 0, F.lit(1)).otherwise(F.lit(0)))).first()[
                              0] or 0
        denom_ndcg = denom_precision  # same eligibility criterion here

        def fmt(x):
            return f"{x:.6f}"

        print(f"[metrics] denom_precision: {int(denom_precision)}  denom_ndcg: {int(denom_ndcg)}")
        print(f"Precision@{k}: {fmt(p_at_k)}")
        print(f"Recall@{k}:    {fmt(r_at_k)}")

    # ----- NDCG@K (JVM-only, no Python UDF) -----
    # We already have:
    #  - recs_topk: user_id, pred_topk (array<string>)
    #  - truth_set: user_id, truth_set (array<string>)

    ndcg_base = recs_topk.join(truth_set, on="user_id", how="inner")

    # DCG: sum over positions i (0-based):
    #   gain_i = 1/log2(i+2) if pred_topk[i] ∈ truth_set else 0
    dcg_df = ndcg_base.select(
        "user_id",
        F.aggregate(
            F.transform(
                F.sequence(F.lit(0), F.size(F.col("pred_topk")) - F.lit(1)),
                lambda i: F.when(
                    F.array_contains(F.col("truth_set"), F.element_at(F.col("pred_topk"), i + 1)),
                    F.lit(1.0) / F.log2(i + F.lit(2.0))
                ).otherwise(F.lit(0.0))
            ),
            F.lit(0.0),
            lambda acc, x: acc + x
        ).alias("dcg"),
        F.least(F.size(F.col("truth_set")), F.lit(k)).alias("ideal_hits")
    )

    # IDCG: best possible DCG with 'ideal_hits' relevant items at the top
    idcg_df = dcg_df.select(
        "user_id", "dcg", "ideal_hits",
        F.when(
            F.col("ideal_hits") > 0,
            F.aggregate(
                F.sequence(F.lit(0), F.col("ideal_hits") - F.lit(1)),
                F.lit(0.0),
                lambda acc, i: acc + (F.lit(1.0) / F.log2(i + F.lit(2.0)))
            )
        ).otherwise(F.lit(0.0)).alias("idcg")
    )

    ndcg_df = idcg_df.select(
        "user_id",
        F.when(F.col("idcg") > 0, F.col("dcg") / F.col("idcg")).otherwise(F.lit(0.0)).alias("ndcg")
    )

    if test.filter(F.col("stars") >= pos_thresh).limit(1).count() == 0:
        print(f"No positives in test at threshold >= {pos_thresh}. Skipping P@{k}/R@{k}/NDCG@{k}.")
        ndcg = 0.0
    else:
        ndcg = float(ndcg_df.agg(F.mean("ndcg").alias("ndcg")).first()["ndcg"] or 0.0)
        print(f"NDCG@{k}: {ndcg:.6f}")
    # --------------------------------------------

    # ---- Persist run metrics ----
    metrics_path = args.metrics_out or metrics_p
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    als_cfg = conf["params"]["als"]

    m = model._java_obj.parent()

    summary = {
        "ts": int(time.time()),
        "in_path": args.in_path,
        "run_dir": run_dir,
        "k": k,
        "pos_thresh": pos_thresh,
        "metrics": {
            "precision_at_k": p_at_k,
            "recall_at_k": r_at_k,
            "ndcg_at_k": ndcg,
        },
        "als_hparams": {
            "rank": int(m.getRank()),
            "regParam": float(m.getRegParam()),
            "maxIter": int(m.getMaxIter()),
            "coldStartStrategy": str(m.getColdStartStrategy()),
            "nonnegative": bool(m.getNonnegative())
        },
        "counts": {
            "n_users": df.select("user_id").distinct().count(),
            "n_items": df.select("business_id").distinct().count(),
            "n_rows": df.count(),
            "train_size": train.count(),
            "test_size": test.count(),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[metrics] wrote {metrics_path}")

    manifest = {
        "run_dir": run_dir,
        "artifacts": {
            "model_dir": model_dir,
            "ui_map": ui_dir,
            "bi_map": bi_dir,
            "recs_json": recs_dir,
            "leaderboard_csv": leaderboard_path if args.sweep else None,
            "batch_csv_dir": args.batch_out_csv or batch_dir,
            "metrics_json": metrics_path,
            "config_yaml": os.path.abspath("conf/config.yaml"),
        },
        "params": {
            "k": k,
            "pos_thresh": pos_thresh,
            "als": conf["params"]["als"],
            "filters": {
                "city": args.city,
                "categories": args.categories,
                "min_score": args.min_score,
                "per_user_topk": args.per_user_topk,
            },
            "split_seed": args.split_seed,
            "eval_only": args.eval_only,
        },
        "data": {
            "ratings_parquet": os.path.abspath(args.in_path),
            "biz_parquet": os.path.abspath(args.biz) if args.biz else None,
        },
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] wrote {os.path.join(run_dir, 'manifest.json')}")

    # ----- DEBUG (use SAME threshold) -----
    print("counts:",
          "users =", df.select("user_id").distinct().count(),
          "items =", df.select("business_id").distinct().count(),
          "rows  =", df.count())

    print("train/test sizes:", train.count(), test.count())

    test_pos = test.filter(F.col("stars") >= pos_thresh)
    print("test positives (rows) =", test_pos.count(),
          "users with any positives =", test_pos.select("user_idx").distinct().count())

    print("users with recs =", recs_mapped.count())
    print("sample recs:", recs_mapped.limit(1).toPandas().to_dict(orient="records"))
    print("sample truth:", truth.limit(5).toPandas().to_dict(orient="records"))
    # -------------------------------------

    # save something tangible
    if not args.skip_recs_json:
        recs_mapped.write.mode("overwrite").json(recs_dir)

    # ---- CLEANUP (order matters!) ----
    if dfi.is_cached:
        dfi.unpersist(blocking=False)
    spark.stop()



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--user", dest="user_id", help="Print top-K recs for this user from the saved model")
    p.add_argument("--k", type=int, help="Override K for recs/eval")
    p.add_argument("--exclude-seen", action="store_true",
                   help="Filter out items the user has already interacted with in TRAIN")
    p.add_argument("--format", choices=["text", "csv", "json"], default="text",
                   help="Output format for --user preview (default: text)")
    p.add_argument("--outfile", help="Write --user output to this file (csv|json|text)")
    p.add_argument("--metrics-out", help="Write run metrics JSON to this local path")
    p.add_argument("--sweep", action="store_true",
                   help="Run a small hyperparam sweep and pick best by NDCG@K")
    p.add_argument("--biz", help="Path to businesses parquet with metadata")
    p.add_argument("--batch-out-csv", help="Write enriched batch recs to one CSV")
    p.add_argument("--city", help="Optional city filter for batch export")
    p.add_argument("--categories", help="Comma-list of category keywords to filter (case-insensitive)")
    p.add_argument("--min-score", type=float, help="Min predicted score to keep in batch export")
    p.add_argument("--run-dir", default="data/runs",
                   help="Parent directory for run artifacts (default: data/runs)")
    p.add_argument("--run-name",
                   help="Optional run name; if omitted, a timestamp will be used")
    # === ADD under other add_argument(...) lines ===
    p.add_argument("--per-user-topk", type=int,
                   help="When exporting batch CSV, keep only top-K items per user")
    p.add_argument("--split-seed", type=int, default=42,
                   help="Seed for deterministic per-user splits")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load model from run folder and evaluate/export")
    p.add_argument("--split-mode", choices=["peruser", "global"], default="global",
                   help="Train/test splitting mode. Use 'global' for big datasets (default).")
    p.add_argument("--global-split", type=float, default=0.8,
                   help="Train fraction for --split-mode=global (default: 0.8)")
    p.add_argument("--disable-batch", action="store_true",
                   help="Skip writing batch CSV (useful on big datasets).")
    p.add_argument("--skip-recs-json", action="store_true",
                   help="Skip writing full recs_json directory.")
    p.add_argument("--metrics-sample-frac", type=float, default=0.0,
                   help="If 0<frac<1, compute metrics on a sampled subset of users to reduce shuffle.")
    p.add_argument("--pop-alpha", type=float, default=0.05,
                   help="Popularity blend weight; 0 disables (default 0.05).")

    main(p.parse_args())
