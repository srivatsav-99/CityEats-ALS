# CityEats-CA: Aspect-Aware Restaurant Recommender (Spark + GCP)

A hybrid recommender for Canadian cities (Toronto/Montreal/Vancouver) that combines:
- Collaborative filtering (ALS) over user–restaurant interactions
- Unsupervised topic modeling over review text (LDA)
- Weakly/semi-supervised sentiment to weight aspects
- Optional user/restaurant clustering for taste segments

## Quickstart (Local, Small Sample)
1) Create and activate a virtual env (or use conda), then:
```
pip install -r requirements.txt
```
2) Run a smoke test (uses a tiny synthetic sample):
```
python jobs/ingest_yelp_local.py --in data/sample_reviews.json --out data/bronze_reviews.parquet
python jobs/topic_model_local.py --in data/bronze_reviews.parquet --out data/topics
python jobs/train_als_local.py --in data/bronze_reviews.parquet --out data/als_model
```
Artifacts will be saved under `data/` to keep this demo self-contained.

## Project layout
- `src/common/spark_utils.py`: SparkSession helper and common conf
- `jobs/ingest_yelp_local.py`: Reads JSON reviews -> cleans -> parquet
- `jobs/topic_model_local.py`: LDA topic model pipeline
- `jobs/train_als_local.py`: ALS recommender baseline + eval@K
- `conf/config.yaml`: Centralized config (paths, params)
- `scripts/gcp_setup.sh`: Template for GCP & Dataproc
- `Makefile`: Handy shortcuts

## GCP (Outline)
- Create GCS bucket, upload Yelp JSON, spin up Dataproc (v2+), then submit jobs:
```
gcloud dataproc clusters create ceats-cluster --region=us-central1 --single-node --image-version=2.2-debian12
gcloud dataproc jobs submit pyspark jobs/ingest_yelp_local.py --cluster ceats-cluster --region us-central1 --   --in gs://<your-bucket>/yelp/review.json --out gs://<your-bucket>/bronze/reviews
```
(Replace paths; see `scripts/gcp_setup.sh` for a fuller template.)

## Train + serve + compare

# 1) Base run
python jobs/train_als_local.py --in data/bronze_reviews.parquet --out data/als_model --run-dir data/runs --run-name base --split-seed 7

# 2) Another run
python jobs/train_als_local.py --in data/bronze_reviews.parquet --out data/als_model --run-dir data/runs --run-name k5 --k 5 --split-seed 7

# 3) Build index
python tools/collect_runs.py
type data\runs\runs_index.csv

# 4) (optional) sweeps
python jobs/train_als_local.py --in data/bronze_reviews.parquet --out data/als_model --run-dir data/runs --run-name sweepA --sweep --split-seed 7

# 5) Pick best run
python tools/pick_best_run.py

# 6) Serve a user (from a run)
python jobs/serve_user_recs.py --run-dir data/runs --run-name base --user u3 --k 10 --format text


## License
MIT



#25-10-2025
# CityEats Recommender (PySpark ALS)

Top-N restaurant recommendations using Spark MLlib (ALS) with ranking metrics (Precision@K, Recall@K, NDCG@K).  
**Local-only folders are gitignored:** `data/`, `artifacts/`.

## Quickstart
1) Create and activate a venv, then:
