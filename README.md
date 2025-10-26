# CityEats-CA: Aspect-Aware Restaurant Recommender (Spark + PySpark ALS)

Hi, I’m **Srivatsav Shrikanth** — a data and ML enthusiast who loves building systems that make everyday decisions a little smarter.  
**CityEats-CA** predicts where people are most likely to enjoy eating, starting with cities like **Toronto, Vancouver, and Montreal**.

This project applies **Spark MLlib’s ALS (Alternating Least Squares)** for collaborative filtering, structured evaluation, and a full end-to-end serving flow — all built on a **local Windows Spark setup**.

---

## SMART Project Goal

| Element | Description |
|----------|-------------|
| **Specific** | Build a scalable Top-N restaurant recommender using PySpark ALS on explicit user ratings. |
| **Measurable** | Evaluate model performance using Precision@K, Recall@K, and NDCG@K. |
| **Achievable** | Run efficiently on local Spark (Windows) without any cluster dependencies. |
| **Relevant** | Strengthen my ML systems and data engineering skills through a real-world recommender scenario. |
| **Time-Bound** | Completed in **October 2025** as part of **MET CS 777 – Big Data Analytics Term Project**. |

---

## Stack Overview

| Layer | Tools |
|-------|-------|
| **Data Processing** | PySpark 3.5.6 · Spark SQL · Parquet |
| **Modeling** | ALS (Alternating Least Squares) |
| **Evaluation** | Precision@K · Recall@K · NDCG@K |
| **Serving** | Command-line interface (`src/serving/cli.py`) |
| **Platform** | Local Windows · Optional GCP Dataproc |
| **Versioning** | `artifacts/runs/<timestamp>` for each trained model |

---

## Train & Evaluate

```powershell
$env:PYTHONPATH="$PWD"
python .\jobs\train_als_local.py `
  --in data\silver_explicit.parquet `
  --out artifacts\tmp `
  --run-dir artifacts\runs `
  --split-mode peruser `
  --k 10 `
  --exclude-seen `
  --pop-alpha 0.05 `
  --metrics-out artifacts\metrics.json `
  --metrics-sample-frac 1.0 `
  --disable-batch `
  --skip-recs-json
```

Artifacts and metrics are saved under `artifacts\runs\<timestamp>\`.  
The best-performing model is stored at `artifacts\runs\best\`.

---

## Final Results (October 2025)

| Metric @ K = 10 | Value | Interpretation |
|------------------|--------|----------------|
| **Precision** | 0.0021 | Around 0.21% of top-10 recommendations were relevant — expected for large, sparse data. |
| **Recall** | 0.0017 | The model recovered about 0.17% of all relevant items per user. |
| **NDCG** | 0.0034 | Shows good ranking differentiation after popularity re-weighting. |
| **User Coverage** | 1.8% (≈ 2.5K of 135K users) | Indicates reasonable reach given ~20M total interactions. |

**Key parameters:**  
`rank = 32`, `regParam = 0.02`, `maxIter = 15`, `nonneg = False`, `pop_alpha = 0.05`, `pos_thresh = 3.5`

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
    {"business_id": 121372, "score": 5.37},
    {"business_id": 117444, "score": 5.34},
    {"business_id": 26762, "score": 5.32},
    {"business_id": 1, "score": 5.31},
    {"business_id": 1210, "score": 5.30},
    {"business_id": 89056, "score": 5.28}
  ]
}
```

---

## One-Shot Run

```powershell
.\scripts\run_local.ps1 -K 10 -Frac 1.0 -PopAlpha 0.05
```

This PowerShell script automates training, metrics logging, and run management — perfect for quick experiments.

---

## Key Learnings & Takeaways

- Learned to **debug Spark on Windows** using `RawLocalFileSystem` — an underrated but vital trick for local dev.  
- Understood why **Precision@K can seem small yet remain meaningful** for extremely sparse recommendation data.  
- Built a **reproducible pipeline** with proper artifact management, CLI serving, and version control.  
- Realized that **clean engineering practices** (scripts, structure, logging) transform a course project into a **portfolio-grade ML system**.

---

## Folder Highlights

| Path | Description |
|------|--------------|
| `src/common/spark_utils.py` | Spark session config + Windows compatibility fixes |
| `jobs/train_als_local.py` | ALS training and evaluation pipeline |
| `src/serving/cli.py` | Lightweight command-line recommender |
| `scripts/run_local.ps1` | One-shot local run automation |
| `artifacts/runs/best/` | Frozen model, mappings, and metrics |

---

## Personal Reflection

What started as a class project quickly became an exploration of how **real-world ML systems come together**.  
I learned that it’s not just about building a model — it’s about designing for reproducibility, traceability, and clarity.  

Every bug fixed and warning suppressed felt like a small win toward becoming a better engineer.  
CityEats-CA isn’t just a recommender system — it’s a reminder of progress, persistence, and curiosity.

---

**MIT License © 2025 — Srivatsav Shrikanth**  
_Boston University · Humber College · Toronto, ON 🇨🇦_
