# streamlit_app.py
import os
import io
import re
import glob
import json
import zipfile
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

# --- Offline recommender (pure Python/NumPy) ---
# Assumes you have src/serving/offline_recs.py with load_bundle() and recommend()
from src.serving.offline_recs import load_bundle, recommend

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="CityEats-ALS — Demo", layout="wide")
st.title("CityEats-ALS — Scalable Food Recommender (Demo)")
st.caption("Streamlit demo using the frozen offline bundle (no Spark). Built by Srivatsav Shrikanth.")

# ---------------- PATHS / CONSTANTS ----------------
# Local, human-readable maps (optional; used only to show readable names if present)
MAP_ITEM = "map_item/map_item.csv"

# GCS bucket + artifacts (sidebar cloud browsing + metrics sync)
USER_ID_ENV = os.getenv("USER", "sri99")
PROJECT_ID = "sri99-cs777"
BUCKET = f"gs://cityeats-{USER_ID_ENV}"
ARTIFACTS = f"{BUCKET}/artifacts"

# Public GitHub LFS URL for the frozen serving bundle
BUNDLE_URL = (
    "https://media.githubusercontent.com/media/"
    "srivatsav-99/CityEats-ALS/main/artifacts_demo/CityEats-ALS_best_bundle.zip"
)

# Local bundle locations
BUNDLE_ZIP = "artifacts_demo/CityEats-ALS_best_bundle.zip"
BUNDLE_ROOT_A = "artifacts_demo/CityEats-ALS_best_bundle"   # typical unzip root
BUNDLE_ROOT_B = "artifacts_demo"                             # fallback root (zip extracted flat)

# ---------------- HELPERS ----------------
def bundle_user_csv_path() -> str:
    """Return the path to map_user.csv regardless of how the zip extracted."""
    candidates = [
        os.path.join(BUNDLE_ROOT_A, "map_user", "map_user.csv"),
        os.path.join(BUNDLE_ROOT_B, "map_user", "map_user.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""

def ensure_bundle_extracted():
    """Unzip the bundle so we have artifacts_demo/CityEats-ALS_best_bundle/..."""
    os.makedirs("artifacts_demo", exist_ok=True)
    root = BUNDLE_ROOT_A  # artifacts_demo/CityEats-ALS_best_bundle
    # if model already present, we're done
    if os.path.isdir(os.path.join(root, "model")):
        return

    # If the zip isn’t present (e.g. Streamlit Cloud didn’t pull LFS), download it.
    if not os.path.exists(BUNDLE_ZIP):
        try:
            import requests
            url = BUNDLE_URL
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(BUNDLE_ZIP, "wb") as f:
                f.write(r.content)
        except Exception as e:
            st.warning(f"Bundle zip not found and download failed: {e}")
            return

    # Extract zip into artifacts_demo/
    try:
        import zipfile
        with zipfile.ZipFile(BUNDLE_ZIP, "r", allowZip64=True) as zf:
            zf.extractall("artifacts_demo")
    except Exception as e:
        st.warning(f"Could not unzip bundle: {e}")

@st.cache_resource
def _load_bundle():
    """Load the frozen offline bundle once (no Spark)."""
    # Try both possible roots
    for root in [BUNDLE_ROOT_A, BUNDLE_ROOT_B]:
        if os.path.isdir(os.path.join(root, "model")):
            return load_bundle(root)
    # Fallback to A (will raise if not present)
    return load_bundle(BUNDLE_ROOT_A)

@st.cache_data
def get_available_users(n=100):
    """Sample a set of user_ids from the bundle for the dropdown."""
    try:
        b = _load_bundle()
        pool = b["map_user"]["user_id"].dropna().astype(str).unique().tolist()
        if not pool:
            return []
        k = min(n, len(pool))
        return sorted(pd.Series(pool).sample(k, random_state=42).tolist())
    except Exception:
        # Fallback attempts: CSV from bundle or parquet
        try:
            csv_path = bundle_user_csv_path()
            if csv_path:
                dfc = pd.read_csv(csv_path, usecols=["user_id"])
                pool = dfc["user_id"].dropna().astype(str).unique().tolist()
                if pool:
                    k = min(n, len(pool))
                    return sorted(pd.Series(pool).sample(k, random_state=42).tolist())
        except Exception:
            pass
        # Last resort: empty
        return []

def _run_gcloud(args):
    full = ["gcloud", "--quiet", "--project", PROJECT_ID] + args
    p = subprocess.run(full, capture_output=True, text=True)
    return (p.returncode == 0, p.stdout.strip())

def _ensure_demo_dir():
    os.makedirs("artifacts_demo", exist_ok=True)

def render_metrics_from_local(local_path="artifacts_demo/metrics.json"):
    p = Path(local_path)
    if not p.exists():
        st.info("No local metrics found yet. Use the section below to sync a metrics file from GCS.")
        return
    try:
        m = json.loads(p.read_text())
    except Exception as e:
        st.warning(f"Could not parse metrics.json: {e}")
        return

    st.subheader("Model metrics")
    cols = st.columns(3)
    shown = False

    # Ratings metric
    if isinstance(m, dict) and m.get("rmse") is not None:
        cols[0].metric("RMSE (ratings)", f"{float(m['rmse']):.3f}")
        st.caption("Lower RMSE indicates better reconstruction of user–item ratings.")
        shown = True

    # Ranking metrics (if present)
    if isinstance(m, dict) and m.get("p_at_10") is not None:
        cols[1].metric("Precision@10", f"{float(m['p_at_10']):.3f}")
        shown = True
    if isinstance(m, dict) and m.get("map_at_10") is not None:
        cols[2].metric("MAP@10", f"{float(m['map_at_10']):.3f}")
        shown = True

    if not shown:
        st.caption("Raw metrics payload (no known keys like rmse/p_at_10/map_at_10 detected):")
        st.json(m)

    with st.expander("Run details / hyperparameters"):
        for k in ["rank", "regParam", "alpha", "maxIter", "implicitPrefs", "coldStartStrategy"]:
            if k in m:
                st.write(f"- **{k}**: {m[k]}")

# ---------------- SIDEBAR: CLOUD HOOKS ----------------
st.sidebar.header("Cloud Storage (GCS)")
if st.sidebar.button("List cloud artifacts"):
    try:
        ok, out = _run_gcloud(["storage", "ls", "-r", f"{ARTIFACTS}/"])
        if not ok or not out.strip():
            st.sidebar.info("(no output — ensure gcloud is installed & authenticated)")
        else:
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            for path in lines:
                if path.startswith("gs://"):
                    st.sidebar.markdown(f"- [{path}]({path})")
    except Exception as e:
        st.sidebar.error(str(e))

# ---------------- BUNDLE INIT ----------------
ensure_bundle_extracted()
try:
    bundle = _load_bundle()
except Exception as e:
    st.error(f"Failed to load offline bundle: {e}")
    st.stop()

# ---------------- MAIN LAYOUT ----------------
colL, colR = st.columns([1, 2], gap="large")

with colL:
    st.subheader("Pick a user")

    available_users = get_available_users(n=100)
    if not available_users:
        st.error("No users found in the bundle’s map_user. Make sure the zip extracted correctly.")
        st.stop()

    user = st.selectbox("User ID", options=available_users, index=0)
    k = st.slider("Top-K", min_value=3, max_value=20, value=10, step=1)
    show_idx = st.toggle("Show internal IDs", value=False)

    st.markdown("**Exclude seen items (optional)**")
    seen_up = st.file_uploader(
        "Upload a CSV with columns: user_id,item_id",
        type=["csv"],
        accept_multiple_files=False
    )
    seen_df = None
    if seen_up is not None:
        try:
            s = pd.read_csv(seen_up)
            need = {"user_id", "item_id"}
            if not need.issubset(set(s.columns)):
                st.warning(f"Seen CSV must have columns {need}. Got {list(s.columns)}.")
            else:
                seen_df = s.astype({"user_id": str, "item_id": str})
        except Exception as e:
            st.warning(f"Could not read seen CSV: {e}")

with colR:
    st.subheader("Top-K Recommendations")

    # Compute recommendations with the pure-NumPy offline engine
    res = recommend(bundle, user_id=str(user), k=k, seen_df=seen_df)
    items = pd.DataFrame(res.get("items", []))

    if items.empty:
        st.info("No recommendations for this user (or all were excluded as seen). Try another user.")
        st.stop()

    # Try to attach a readable item name from optional MAP_ITEM CSV
    readable_col = None
    try:
        if os.path.exists(MAP_ITEM):
            mi = pd.read_csv(MAP_ITEM)
            for c in ["name", "business_name", "title", "item_name"]:
                if c in mi.columns:
                    readable_col = c
                    break
            if readable_col:
                items = items.merge(mi[["item_id", readable_col]], on="item_id", how="left")
    except Exception:
        pass

    # Choose presentation
    if show_idx or readable_col is None:
        table = items[["item_id", "score"]].rename(columns={"item_id": "item_id (internal)"})
        y_field = "item_id (internal)"
        y_title = "Item ID (internal)"
        df_for_chart = table.rename(columns={"item_id (internal)": "label_for_chart"})
    else:
        table = items[[readable_col, "score"]].rename(columns={readable_col: "recommended_item"})
        y_field = "recommended_item"
        y_title = "Recommended item"
        df_for_chart = table.rename(columns={"recommended_item": "label_for_chart"})

    # Table
    st.dataframe(table, width="stretch", hide_index=True)

    # Chart
    chart = (
        alt.Chart(df_for_chart)
        .mark_bar()
        .encode(
            x=alt.X("score:Q", title="Predicted score"),
            y=alt.Y("label_for_chart:N", sort="-x", title=y_title),
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

    # Download
    st.download_button(
        "Download this Top-K CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"topk_user_{user}.csv",
        mime="text/csv",
    )

if show_idx:
    st.caption("Internal results (debug)")
    with st.expander("Show raw recommendation rows"):
        st.dataframe(items, width="stretch")

# ---------------- METRICS PANEL ----------------
st.divider()
render_metrics_from_local()

# ---------------- OFFLINE BUNDLE SECTION ----------------
st.divider()
st.subheader("Offline serving bundle")
st.caption(
    "Download the frozen model + maps to run recommendations locally without GCS. "
    "This is the exact snapshot frozen from the best cloud run."
)
st.link_button("⬇️ Download CityEats-ALS_best_bundle.zip", BUNDLE_URL)

with st.expander("How to use the bundle (local quickstart)"):
    st.markdown(
        """
1. Unzip the file. You will get `model/`, `map_user/`, and `map_item/`.
2. This Streamlit app already uses the offline bundle automatically once unzipped.
3. (Optional) CLI usage example for the same bundle:

   ```bash
   python -m src.serving.cli \
     --model-dir /path/to/CityEats-ALS_best_bundle/model \
     --ui-map    /path/to/CityEats-ALS_best_bundle/map_user \
     --bi-map    /path/to/CityEats-ALS_best_bundle/map_item \
     --user-id 41397 --k 10
   ```
        """
    )

# ---------------- SYNC LATEST CLOUD METRICS ----------------
st.subheader("Sync latest cloud metrics")
HAS_GCLOUD = shutil.which("gcloud") is not None
BEST_METRICS = f"{BUCKET}/artifacts/runs/best/metrics/metrics.json"

if not HAS_GCLOUD:
    st.info(
        "This environment doesn’t have Google Cloud CLI or credentials, "
        "so syncing from GCS is disabled. The metrics shown above are "
        "the latest frozen metrics bundled with the repo."
    )
else:
    col_sync1, col_sync2 = st.columns([1,2])
    with col_sync1:
        if st.button("Pull newest metrics.json from GCS"):
            _ensure_demo_dir()
            ok, _ = _run_gcloud(["storage", "cp", BEST_METRICS, "artifacts_demo/metrics.json"])
            if ok:
                st.success("Pulled frozen BEST metrics.json from GCS.")
                st.caption(BEST_METRICS)
                render_metrics_from_local()
            else:
                # Fallback: look for latest part file
                ok_ls, out_ls = _run_gcloud(["storage", "ls", "--recursive", f"{BUCKET}/artifacts/"])
                if not ok_ls or not out_ls:
                    st.warning("No metrics found in GCS (yet) or no access.")
                else:
                    candidates = [
                        s.strip() for s in out_ls.splitlines()
                        if s.strip().endswith(".json") and "/metrics/" in s and "part-" in s
                    ]
                    if not candidates:
                        st.warning("No metrics part file found in GCS.")
                    else:
                        part = sorted(candidates)[-1]
                        ok2, _ = _run_gcloud(["storage", "cp", part, "artifacts_demo/metrics.json"])
                        if ok2:
                            st.success("Pulled latest Spark metrics part file from GCS.")
                            st.caption(part)
                            render_metrics_from_local()
                        else:
                            st.error("Failed to pull metrics from GCS. Ensure you’re authenticated (gcloud auth login).")

# ---------------- ABOUT / MAPPING ----------------
st.divider()
st.markdown(
    """
**How this demo maps to the real system**

- This page uses a **frozen offline bundle** (model + maps) for instant, Spark-free recommendations.
- The full pipeline runs Spark ALS on **GCP Dataproc**, writing Parquet + JSON metrics to **GCS**:
  - Runs: `gs://cityeats-<user>/artifacts/runs/...`
  - Metrics: `gs://cityeats-<user>/artifacts/metrics/...`
- Use the sidebar to confirm your cloud artifacts exist.
"""
)

st.divider()
st.markdown(
    """
### About this Demo

This demo showcases the **CityEats-ALS** recommender system, a scalable ML pipeline built with **PySpark ALS** and deployed on **Google Cloud Dataproc**.

- **Dataset:** 50M+ explicit Yelp ratings (Tier B scale)
- **Model:** ALS (e.g., rank=64, regParam=0.1, maxIter=12)
- **Artifacts:** Stored on `gs://cityeats-sri99/artifacts/runs/best/`
- **Purpose:** Visualize top-N recommendations, metrics, and artifact structure

This Streamlit version uses a compact, frozen bundle to mirror the behavior of the full cloud pipeline—**lightweight and explainable**, no Spark required.
"""
)
