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

try:
    recs, users, items = load_demo()
except FileNotFoundError as e:
    st.error(f"Missing demo file: {e}")
    st.stop()

colL, colR = st.columns([1,2], gap="large")

with colL:
    st.subheader("Pick a user")
    user = st.selectbox("User ID", users["user_id"].unique(), index=0)
    k = st.slider("Top-K", min_value=3, max_value=10, value=10, step=1)
    show_idx = st.toggle("Show internal indices", value=False)




with colR:
    st.subheader("Top-K Recommendations")

    #Top-K for the selected user
    df = (
        recs.loc[recs["user_id"].eq(user)]
            .sort_values("score", ascending=False)
            .head(k)
            .copy()
    )

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




import re

#Sync latest cloud metrics
st.subheader("Sync latest cloud metrics")

PROJECT_ID = "sri99-cs777"
USER_ID = os.getenv("USER", "sri99")
BEST_METRICS = f"gs://cityeats-{USER_ID}/artifacts/runs/best/metrics/metrics.json"

def _run_gcloud(args):
    """Run gcloud with --project and --quiet; return (ok, stdout, stderr)."""
    full = ["gcloud", "--quiet", "--project", PROJECT_ID] + args
    p = subprocess.run(full, capture_output=True, text=True)
    return (p.returncode == 0, (p.stdout or "").strip(), (p.stderr or "").strip())

def _ensure_demo_dir():
    os.makedirs("artifacts_demo", exist_ok=True)

def _latest_part_metrics():
    """
    Return (status, path_or_msg). If status is True, value is the latest part-*.json path.
    If False, value is an error message from gcloud (stderr/stdout).
    """
    ok, out, err = _run_gcloud(["storage", "ls", "--recursive",
                                f"gs://cityeats-{USER_ID}/artifacts/"])
    if not ok:
        return False, (err or out or "gcloud ls failed (no details)")
    candidates = []
    for line in out.splitlines():
        s = line.strip()
        if s.endswith(".json") and "/metrics/" in s and "part-" in s:
            candidates.append(s)
    if not candidates:
        return False, "No metrics part file found in GCS (yet)."
    return True, sorted(candidates)[-1]

if st.button("Pull newest metrics.json from GCS"):
    _ensure_demo_dir()

    #frozen best metrics
    ok, _, err = _run_gcloud(["storage", "cp", BEST_METRICS, "artifacts_demo/metrics.json"])
    if ok:
        st.success("Pulled frozen BEST metrics.json from GCS.")
        st.caption(BEST_METRICS)
        render_metrics_from_local()
    else:
        #Fall back to latest Spark part-*.json
        status, val = _latest_part_metrics()
        if status:
            part = val
            ok2, _, err2 = _run_gcloud(["storage", "cp", part, "artifacts_demo/metrics.json"])
            if ok2:
                st.success("Pulled latest Spark metrics part file from GCS.")
                st.caption(part)
                render_metrics_from_local()
            else:
                st.error("Failed to copy metrics from GCS (part-*.json).")
                st.caption(err2 or "No error text from gcloud.")
        else:
            
            if "No metrics part file" in val:
                st.warning(val)
            else:
                st.error("Failed to list metrics in GCS.")
                st.caption(val)




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
