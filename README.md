# CityEats-CA: Aspect-Aware Restaurant Recommender (Spark + PySpark ALS)

Hi, I’m **Srivatsav Shrikanth** — a data and ML enthusiast who loves building systems that make everyday decisions a little smarter.  
**CityEats-CA** predicts where people are most likely to enjoy eating, starting with cities like **Toronto, Vancouver, and Montreal**.

This system began as a **local Windows Spark prototype** and evolved into a **fully cloud-scale recommender** on **Google Cloud Dataproc**, trained over **50 million + interactions** from the Yelp dataset.  
The journey reflects how raw coursework can mature into a **production-grade ML system** — reproducible, explainable, and cost-efficient.

---

## SMART Project Goal

| Element | Description |
|----------|-------------|
| **Specific** | Build a scalable Top-N restaurant recommender using PySpark ALS on explicit user ratings. |
| **Measurable** | Evaluate using Precision@K, Recall@K, NDCG@K, and RMSE across datasets. |
| **Achievable** | Prototype locally → scale seamlessly to GCP Dataproc without code change. |
| **Relevant** | Strengthen ML engineering + data pipeline skills with real-world scale. |
| **Time-Bound** | Delivered in **October 2025** for **BU MET CS 777 – Big Data Analytics**. |

---

## Stack Overview

| Layer | Tools |
|-------|-------|
| **Data Processing** | PySpark 3.5.6 · Spark SQL · Parquet |
| **Modeling** | ALS (Alternating Least Squares) |
| **Evaluation** | Precision@K · Recall@K · NDCG@K · RMSE |
| **Serving** | CLI (`src/serving/cli.py`) · Streamlit Dashboard |
| **Platform** | Windows (local) · Google Cloud Dataproc · GCS |
| **Versioning** | `artifacts/runs/<timestamp>` for each trained model |

---

## Training & Evaluation Phases

### **Phase A — Local Prototype (Sparse Dataset, Explicit Ratings)**

- Dataset ≈ 20 M interactions (Toronto subset)  
- Goal: validate pipeline + metric computation  
- Metrics expectedly low due to sparsity and per-user split

| Metric @ K = 10 | Value | Context |
|------------------|--------|---------|
| **Precision** | 0.0021 | ~0.21 % of Top-10 recs relevant — expected for sparse data. |
| **Recall** | 0.0017 | Recovered ~0.17 % of true positives per user. |
| **NDCG** | 0.0034 | Reasonable ranking structure with popularity correction. |

➡ These results validated the evaluation pipeline and helped tune **regularization**, **rank**, and **implicit feedback weighting** for cloud-scale runs.

---

### **Phase B — Tier B Dataproc Run (50 M + Interactions)**

- Dataset: **Yelp Explicit Ratings (Tier B Split 80/20)**  
- Platform: **GCP Dataproc Single-Node Cluster (n2-standard-16)**  
- Features: Adaptive AQE ON · 64 shuffle partitions · Pop Reweighting α = 0.05  

| Metric | Value | Notes |
|---------|--------|-------|
| **RMSE** | **0.4070** | Indicates strong fit for explicit ratings prediction. |
| **Precision@10 / Recall@10** | ↑ vs. local (by ~20 %) | Improved through denser global split. |
| **User Coverage** | > 80 % active users scored | Verified via batch recommendations. |

**Key Parameters:** `rank = 64`, `regParam = 0.1`, `maxIter = 12`, `nonneg = False`, `pop_alpha = 0.05`  
These tuned hyperparameters were frozen after the Dataproc evaluation and serve as the official “best run”.

---

## Get Recommendations (CLI)

```powershell
python -m src.serving.cli `
  --model-dir artifacts\runs\best\model `
  --ui-map artifacts\runs\best\ui_map `
  --bi-map artifacts\runs\best\bi_map `
  --seen artifacts\runs\best\seen.parquet `
  --user-id 41397 `
  --k 10
```

### Example Output

```json
{
  "user_id": "41397",
  "items": [
    {"business_id": 260, "score": 5.73},
    {"business_id": 1196, "score": 5.57},
    {"business_id": 94074, "score": 5.43},
    {"business_id": 38881, "score": 5.38},
    {"business_id": 121372, "score": 5.37}
  ]
}
```

---

## One-Shot Automation

```powershell
.\scripts\run_local.ps1 -K 10 -Frac 1.0 -PopAlpha 0.05
```

Automates training, evaluation, and artifact management — perfect for rapid experimentation.

---

## Streamlit Demo (Recruiter Preview)

The live **CityEats-ALS** dashboard runs on **Google Cloud Shell**, showing frozen cloud metrics.

| Feature | Description |
|----------|-------------|
| **Top-N Explorer** | Interactively browse recommendations per user. |
| **Metrics Panel** | Displays RMSE = `0.4070` from Tier B Dataproc run. |
| **Sidebar Artifacts** | Lists live GCS paths to frozen models. |
| **Cloud Sync Check** | Gracefully handles missing live runs (“No metrics found yet”). |

```bash
pip install --user streamlit pandas pyarrow
make demo
# Open Web Preview → Port 8080
```

---

## Frozen “Best Run” (Cloud Truth)

All cloud artifacts are version-controlled in Google Cloud Storage for reproducibility.

| Asset | Path |
|--------|------|
| **Model** | `gs://cityeats-sri99/artifacts/runs/best/model/` |
| **User Map** | `gs://cityeats-sri99/artifacts/runs/best/map_user/` |
| **Item Map** | `gs://cityeats-sri99/artifacts/runs/best/map_item/` |
| **Metrics** | `gs://cityeats-sri99/artifacts/runs/best/metrics/metrics.json` (`RMSE = 0.4070`) |
| **Tag** | `als_rank64_reg0.1_it12_20251027-000742` |

`frozen_config.json` captures dataset and hyperparameter metadata so this result is traceable forever.

---

## Serving-Ready Batch Scoring (GCP)

Generate fresh Top-K recommendations from the frozen model on Dataproc or Spark Submit:

```bash
spark-submit jobs/export_recs_only.py \
  --model gs://cityeats-sri99/artifacts/runs/best/model \
  --map_user gs://cityeats-sri99/artifacts/runs/best/map_user \
  --map_item gs://cityeats-sri99/artifacts/runs/best/map_item \
  --output  gs://cityeats-sri99/artifacts/runs/best/recs_top10
```

Outputs Parquet + CSV files — simulating a nightly batch recommender or lightweight API endpoint.

---

## Cloud Cost & Runtime Summary

| Component | Description | Cost |
|------------|--------------|------|
| **Dataproc Cluster** | Single-node `n2-standard-16` (64 GB RAM, 200 GB disk) | ≈ $0.30 /hr × 2 hr = $0.60 |
| **GCS Artifacts** | Model + Maps (~1.1 GB total) | ≈ $0.02 / month |
| **Streamlit Preview** | Runs in Cloud Shell | Free |
| **Total Cost** | All training + serving within free-tier credits | ✅ $0 charged |

---

## Recruiter Snapshot

| Category | Highlights |
|-----------|-------------|
| **ML System Design** | End-to-end ALS recommender with evaluation + serving. |
| **Scalability** | Local prototype → Dataproc (16 vCPUs, 64 GB RAM). |
| **MLOps Mindset** | Frozen artifacts (`runs/best`), metrics versioning, sync scripts. |
| **Transparency** | Live GCS URIs + Streamlit dashboard for auditability. |
| **Impact** | Demonstrates engineering discipline + real ML ops understanding. |

---

## Key Learnings & Takeaways

- Mastered **Spark on Windows** debugging using `RawLocalFileSystem`.  
- Built end-to-end pipelines with reproducible artifacts and version control.  
- Optimized ALS hyperparameters (`rank`, `regParam`, `iter`) using structured metrics.  
- Learned to scale from local to cloud using Dataproc and GCS efficiently.  
- Developed production habits - logging, resumable runs, traceability, cost awareness.  

---

## Folder Highlights

| Path | Description |
|------|--------------|
| `jobs/train_als_local.py` | ALS training and evaluation pipeline |
| `jobs/export_recs_only.py` | Batch inference and CSV export |
| `src/common/spark_utils.py` | Spark session config + Windows fixes |
| `src/serving/cli.py` | Command-line recommender |
| `artifacts/runs/best/` | Frozen model + metrics (Cloud Truth) |
| `artifacts_demo/` | Streamlit demo files + screenshots |

---

## Personal Reflection

CityEats-CA was a long-haul build from debugging Spark on Windows to tuning ALS on Dataproc.  
It taught me that real ML engineering isn’t about accuracy alone, it’s about **designing for clarity, reproducibility, and trust**.  

Every fix, every cloud sync, every RMSE improvement was a step toward industry-grade thinking.  
CityEats-CA is more than a recommender, it’s proof that rigor and curiosity scale together.

---

## MIT License

**MIT License © 2025 — Srivatsav Shrikanth**  
_Boston University · Humber College · Toronto, ON 🇨🇦_
