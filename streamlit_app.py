import os, subprocess, io
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="CityEats-ALS — Demo", layout="wide")

#Paths
LOCAL_RECS = "recs_top10_sample_csv/sample.csv"
MAP_USER = "map_user/map_user.csv"
MAP_ITEM = "map_item/map_item.csv"
BUCKET = f"gs://cityeats-{os.getenv('USER','sri99')}"
ARTIFACTS = f"{BUCKET}/artifacts"
#Public GitHub LFS URL for the frozen serving bundle
BUNDLE_URL = (
    "https://media.githubusercontent.com/media/"
    "srivatsav-99/CityEats-ALS/main/artifacts_demo/CityEats-ALS_best_bundle.zip"
)

BUNDLE_ZIP = "artifacts_demo/CityEats-ALS_best_bundle.zip"
BUNDLE_ROOT_A = "artifacts_demo/CityEats-ALS_best_bundle" 
BUNDLE_ROOT_B = "artifacts_demo"
USER_POOL = "artifacts_demo/user_pool.csv"

def bundle_user_csv_path() -> str:
    """Return the path to map_user.csv regardless of how the zip extracted"""
    candidates = [
        os.path.join(BUNDLE_ROOT_A, "map_user", "map_user.csv"),
        os.path.join(BUNDLE_ROOT_B, "map_user", "map_user.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""





st.title("CityEats-ALS - Scalable Food Recommender (Demo)")
st.caption("Small CSV demo while the full Spark pipeline runs in cloud. Built by Srivatsav Shrikanth.")

#Sidebar : Cloud hooks
st.sidebar.header("Cloud Storage")

if st.sidebar.button("List cloud artifacts"):
    try:
        out = subprocess.check_output(["gcloud", "storage", "ls", "-r", f"{ARTIFACTS}/"], text=True)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if not lines:
            st.sidebar.info("(empty)")
        else:
            for path in lines:
                if path.startswith("gs://"):
                    st.sidebar.markdown(f"- [{path}]({path})")
    except subprocess.CalledProcessError as e:
        st.sidebar.error((e.output or str(e)).strip())


#Data loaders
@st.cache_data
def load_demo():
    recs = pd.read_csv(LOCAL_RECS)
    u = pd.read_csv(MAP_USER)
    i = pd.read_csv(MAP_ITEM)
    # Join readable ids (idxs stand in for titles in this tiny demo)
    recs = recs.merge(u, on="user_id", how="left").merge(i, on="item_id", how="left")
    return recs, u, i

import zipfile

def ensure_bundle_extracted():

    if os.path.isdir(os.path.join(BUNDLE_ROOT_A, "map_user")) or \
       os.path.isdir(os.path.join(BUNDLE_ROOT_B, "map_user")):
        return

    try:
        os.makedirs("artifacts_demo", exist_ok=True)
        if os.path.exists(BUNDLE_ZIP):
            with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
                zf.extractall("artifacts_demo")
        else:
            st.warning(f"Bundle zip not found at: {BUNDLE_ZIP}")
    except Exception as e:
        st.warning(f"Could not unzip bundle: {e}")



ensure_bundle_extracted()


import glob

@st.cache_data
def get_available_users(n=12):


    #Curated pool
    try:
        if os.path.exists(USER_POOL):
            dfp = pd.read_csv(USER_POOL, usecols=["user_id"])
            pool = dfp["user_id"].dropna().astype(str).unique().tolist()
            if pool:
                k = min(n, len(pool))
                return sorted(pd.Series(pool).sample(k, random_state=42).tolist())
    except Exception:
        pass

    #Bundle CSV
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

    #Bundle Parquet
    try:
        parts = sorted(glob.glob(os.path.join(BUNDLE_ROOT_A, "map_user", "part-*.parquet")))
        if not parts:
            parts = sorted(glob.glob(os.path.join(BUNDLE_ROOT_B, "map_user", "part-*.parquet")))
        if parts:
            dfp = pd.read_parquet(parts[0], columns=["user_id"], engine="pyarrow")
            pool = dfp["user_id"].dropna().astype(str).unique().tolist()
            if pool:
                k = min(n, len(pool))
                return sorted(pd.Series(pool).sample(k, random_state=42).tolist())
    except Exception:
        pass

    #Fallback
    try:
        dfm = pd.read_csv(MAP_USER, usecols=["user_id"])
        pool = dfm["user_id"].dropna().astype(str).unique().tolist()
        return sorted(pool)
    except Exception:
        return []






try:
    recs, users, items = load_demo()

    RECS_USER_IDS = set(recs["user_id"].astype(str).unique())
    
except FileNotFoundError as e:
    st.error(f"Missing demo file: {e}")
    st.stop()

colL, colR = st.columns([1,2], gap="large")

with colL:
    st.subheader("Pick a user")
    available_users = get_available_users(n=15)
    st.caption("Select any user ID to refresh recommendations (loaded from the frozen bundle when available).")
    user = st.selectbox("User ID", options=available_users, index=0)

    k = st.slider("Top-K", min_value=3, max_value=10, value=10, step=1)
    show_idx = st.toggle("Show internal indices", value=False)




with colR:
    st.subheader("Top-K Recommendations")

    #Top-K for the selected user
    df = (
        recs.loc[recs["user_id"].astype(str).eq(str(user))]
            .sort_values("score", ascending=False)
            .head(k)
            .copy()
    )
    if df.empty:
        st.info(
            "No demo recommendations for this user in the small sample CSV. "
            "Pick another user ID from the dropdown."
        )
        st.stop()

    readable_item_col = next(
        (c for c in ["item_name", "name", "business_name", "title", "item"]
         if c in df.columns),
        "item_id"  #fallback to internal id if no readable name exists
    )

    if show_idx:
        #explicitly showing ids
        table = (
            df[["user_id", "item_id", "score"]]
            .rename(columns={
                "user_id": "user_id (internal)",
                "item_id": "item_id (internal)"
            })
        )
        y_field = "item_id"      #chart labels
        y_title = "Item ID (internal)"
    else:
        #showing a readable item label if available
        table = (
            df[[readable_item_col, "score"]]
            .rename(columns={readable_item_col: "recommended_item"})
        )
        y_field = readable_item_col
        y_title = "Recommended item"

    #Table
    st.dataframe(table, width="stretch", hide_index=True)

    #Chart
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("score:Q", title="Predicted score"),
            y=alt.Y(f"{y_field}:N", sort="-x", title=y_title),
        )
        .properties(height=300)
    )
    st.altair_chart(chart)

    #download
    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download this Top-K CSV",
        data=csv_bytes,
        file_name="topk.csv",
        mime="text/csv",
    )



if show_idx:
    st.caption("Internal indices preview")
    with st.expander("Show merged demo frame"):
        st.dataframe(df, width="stretch")





def render_metrics_from_local(local_path="artifacts_demo/metrics.json"):
    import json, pathlib
    p = pathlib.Path(local_path)
    if not p.exists():
        st.info("No local metrics found yet. Use the steps below to sync a metrics file from GCS.")
        return

    try:
        m = json.loads(p.read_text())
    except Exception as e:
        st.warning(f"Could not parse metrics.json: {e}")
        return

    # Header
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

#Metrics panel

render_metrics_from_local()

#offline bundle section
st.divider()
st.subheader("Offline serving bundle")

st.caption(
    "Download the frozen model + maps to run recommendations locally without GCS. "
    "This is the exact snapshot we froze from the best cloud run."
)

st.link_button("⬇️ Download CityEats-ALS_best_bundle.zip", BUNDLE_URL)

with st.expander("How to use the bundle (local quickstart)"):
    st.markdown(
        """
        1. Unzip the file. You will get `model/`, `map_user/`, and `map_item/`.
        2. Run the CLI pointing to those folders (example):

           ```bash
           python -m src.serving.cli \
             --model-dir /path/to/CityEats-ALS_best_bundle/model \
             --ui-map    /path/to/CityEats-ALS_best_bundle/map_user \
             --bi-map    /path/to/CityEats-ALS_best_bundle/map_item \
             --user-id 41397 --k 10
           ```
        3. You should see Top-K items with scores printed for the given user id.
        """
    )



import re

#Sync latest cloud metrics
import shutil

st.subheader("Sync latest cloud metrics")

HAS_GCLOUD = shutil.which("gcloud") is not None
PROJECT_ID = "sri99-cs777"
USER_ID = os.getenv("USER", "sri99")
BEST_METRICS = f"gs://cityeats-{USER_ID}/artifacts/runs/best/metrics/metrics.json"

def _run_gcloud(args):
    full = ["gcloud", "--quiet", "--project", PROJECT_ID] + args
    p = subprocess.run(full, capture_output=True, text=True)
    return (p.returncode == 0, p.stdout.strip())

def _ensure_demo_dir():
    os.makedirs("artifacts_demo", exist_ok=True)

def _latest_part_metrics():
    ok, out = _run_gcloud(["storage", "ls", "--recursive", f"gs://cityeats-{USER_ID}/artifacts/"])
    if not ok or not out:
        return ""
    candidates = [s.strip() for s in out.splitlines()
                  if s.strip().endswith(".json") and "/metrics/" in s and "part-" in s]
    return sorted(candidates)[-1] if candidates else ""

if not HAS_GCLOUD:
    st.info(
        "This hosted demo doesn’t have Google Cloud CLI or credentials, "
        "so syncing from GCS is disabled. The metrics shown above are the "
        "latest frozen metrics bundled with the repo."
    )
else:
    if st.button("Pull newest metrics.json from GCS"):
        _ensure_demo_dir()
        ok, _ = _run_gcloud(["storage", "cp", BEST_METRICS, "artifacts_demo/metrics.json"])
        if ok:
            st.success("Pulled frozen BEST metrics.json from GCS.")
            st.caption(BEST_METRICS)
            render_metrics_from_local()
        else:
            part = _latest_part_metrics()
            if not part:
                st.warning("No metrics part file found in GCS (yet).")
            else:
                ok2, _ = _run_gcloud(["storage", "cp", part, "artifacts_demo/metrics.json"])
                if ok2:
                    st.success("Pulled latest Spark metrics part file from GCS.")
                    st.caption(part)
                    render_metrics_from_local()
                else:
                    st.error("Failed to pull metrics from GCS. Make sure you’re authenticated in Cloud Shell.")




st.divider()

st.markdown(
"""
**How this demo maps to the real system**

- This page reads tiny CSVs checked into git.
- The real pipeline runs Spark ALS on GCP Dataproc, writes Parquet + JSON metrics to **GCS**:
  - Runs: `gs://cityeats-<user>/artifacts/runs/...`
  - Metrics: `gs://cityeats-<user>/artifacts/metrics/...`
- Use the sidebar button to confirm your cloud artifacts exist.
"""
)

st.divider()

st.markdown("""
### About this Demo

This demo showcases the **CityEats-ALS** recommender system, a scalable ML pipeline built with **PySpark ALS** and deployed on **Google Cloud Dataproc**.

- **Dataset:** 50M+ explicit Yelp ratings (Tier B scale)
- **Model:** ALS (rank=64, regParam=0.1, maxIter=12)
- **Metrics:** RMSE ≈ 0.407 (explicit ratings)
- **Artifacts:** Stored on `gs://cityeats-sri99/artifacts/runs/best/`
- **Purpose:** Visualize top-N recommendations, metrics, and artifact structure

This Streamlit version uses compact CSV samples to mirror the behavior of the full cloud pipeline : lightweight and explainable.
""")

