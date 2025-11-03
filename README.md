# CityEats‑ALS — Scalable Food Recommender System  
**Developed by [Srivatsav Shrikanth](https://github.com/srivatsav-99)**  
*Machine Learning Engineer | Humber College & Boston University MET*  

---

## Overview

**CityEats-ALS** is a production‑grade, scalable **food recommendation system** built with **PySpark ALS** on **Google Cloud Dataproc** and deployed through a modern **Streamlit UI**.  
It demonstrates end‑to‑end machine learning system design - from distributed model training on 50M+ interactions to an explainable, server‑less user interface powered entirely by a frozen offline bundle (no Spark required).  

This project represents my ability to design, optimize, and deliver a **cloud‑scale ML system** that is both technically sound and presentation‑ready.

---

## Architecture Overview

### **Cloud Training Pipeline (Google Cloud Dataproc)**
- Engineered an **ALS (Alternating Least Squares)** recommender using PySpark 3.5  
- Optimized for **50M+ Yelp explicit ratings**, training in distributed mode (Tier‑B scale)  
- Generated **Parquet artifacts** for user/item factors, mappings, and evaluation metrics  
- Metrics and artifacts versioned on **GCS (`gs://cityeats-sri99/artifacts/`)**

### **Offline Serving Layer (Python/NumPy)**
- Designed a **Spark‑free NumPy inference layer** (`src/serving/offline_recs.py`)  
- Loads compact `itemFactors` and `userFactors` Parquet parts directly into memory  
- Enables real‑time cosine‑like scoring for any user without cloud runtime dependency  

### **Interactive Streamlit Dashboard**
- Built the entire front‑end in **Streamlit**, allowing users to:
  - Select any `user_id` and visualize **Top‑K recommendations**
  - Upload **custom user-item CSVs** to test pair scoring interactively
  - Inspect **model metrics** (RMSE, Precision@10, MAP@10)
  - Sync cloud metrics from GCS directly via `gcloud storage cp`
- Fully self‑contained works even when cloud access is disabled

---

## Cloud Deployment & Scale

| Component | Technology | Purpose |
|------------|-------------|----------|
| Compute | **GCP Dataproc (single node, auto‑scalable)** | Train ALS efficiently on distributed parquet data |
| Storage | **Google Cloud Storage (GCS)** | Artifact & metric persistence |
| Job Orchestration | **gcloud dataproc jobs submit pyspark** | Automated training runs |
| Versioning | **Git + Git LFS** | Bundle large artifacts and models |
| Visualization | **Streamlit Cloud** | Live deployment of offline demo |

---

## Offline Bundle & Serving

The frozen serving bundle - **`CityEats‑ALS_best_bundle.zip`** - contains everything required to run inference locally:

```
model/
├── itemFactors/     # ALS item latent factors
├── userFactors/     # ALS user latent factors
├── metadata/        # Spark model metadata
map_user/
map_item/
metrics.json
```

It’s tracked via **Git LFS** to handle its 500+ MB size safely.  
In Streamlit, users can download this same bundle or run the CLI manually:

```bash
python -m src.serving.cli \
  --model-dir artifacts_demo/CityEats-ALS_best_bundle/model \
  --ui-map    artifacts_demo/CityEats-ALS_best_bundle/map_user \
  --bi-map    artifacts_demo/CityEats-ALS_best_bundle/map_item \
  --user-id lU0x0khkn8g7ZeDmkWA --k 10
```

---

## Model Evaluation

| Metric | Description | Best Run Value |
|:-------|:-------------|:---------------|
| **RMSE** | Root Mean Square Error on explicit ratings | **0.407** |
| **Precision@10** | Top‑10 relevance precision | 0.321 |
| **MAP@10** | Mean Average Precision (ranking) | 0.284 |

> All metrics frozen from the final Dataproc run (`gs://cityeats-sri99/artifacts/runs/best/metrics/metrics.json`).

---

## Engineering Highlights

- **Algorithm Optimization:** Tuned ALS hyperparameters (`rank=64, regParam=0.1, maxIter=12`) with adaptive AQE for faster Spark shuffles.  
- **Resilient Architecture:** Portable model serving independent of Spark or Java runtime.  
- **Interactive Visualization:** Built with Streamlit + Altair for real‑time analytics and user interactivity.  
- **Artifact Versioning:** Implemented reproducible model freezing workflow via Git LFS.  
- **Explainability:** Human‑readable item mapping allows users to see not just IDs but actual restaurant names.  

---

## Repository Map

```
CityEats-ALS/
│
├── artifacts_demo/
│   ├── CityEats-ALS_best_bundle.zip   # Git LFS bundle
│   ├── metrics.json                   # Frozen metrics
│
├── src/
│   └── serving/
│       ├── cli.py                     # Local CLI inference
│       └── offline_recs.py            # Pure‑Python offline recommender
│
├── streamlit_app.py                   # Interactive dashboard
└── README.md
```

---

## Streamlit Demo Highlights

- Select user IDs dynamically (100+ available)  
- Generate **Top‑K recommendations** instantly using offline NumPy inference  
- Toggle **Show internal IDs** for transparency  
- Upload **custom CSVs** to test user‑item pairs manually  
- Visualize metrics directly from `metrics.json`  
- Sync latest GCS metrics if authenticated via Google Cloud CLI  

> The app mirrors how a lightweight serving layer can coexist with enterprise‑scale cloud training.

---

## Quickstart (Local)

```bash
# Clone repository
git clone https://github.com/srivatsav-99/CityEats-ALS.git
cd CityEats-ALS

# Initialize Git LFS and pull large bundle
git lfs install
git lfs pull

# Install dependencies
pip install -r requirements.txt

# Run the demo locally
streamlit run streamlit_app.py
```

Open your browser at **http://localhost:8501**.  

---

## Project Impact

CityEats‑ALS bridges the gap between **academic ML** and **real‑world data engineering**.  
It demonstrates:  
- Scalable Spark training workflows on real big data  
- Efficient artifact management and offline reproducibility  
- Hands‑on understanding of **end‑to‑end ML systems** - from Dataproc to Streamlit deployment  

> This project showcases my practical engineering mindset - focusing on reproducibility, clarity, and real‑world deployability of ML systems.

---

## License

MIT License © 2025 [Srivatsav Shrikanth](https://github.com/srivatsav-99)

