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
    st.dataframe(table, use_container_width=True, hide_index=True)

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
    st.altair_chart(chart, use_container_width=True)

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
        st.dataframe(df, use_container_width=True)






#Metrics panel
st.subheader("Model metrics")

import json, os, subprocess

def load_metrics(local_path="artifacts_demo/metrics.json"):
    if not os.path.exists(local_path):
        return {}
    try:
        with open(local_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not parse metrics.json: {e}")
        return {}

m = load_metrics()

if not m:
    st.info("No local metrics found yet. Use the steps below to sync a metrics file from GCS.")
else:
    cols = st.columns(3)
    shown = False

    #Ratings style metric
    if isinstance(m, dict) and m.get("rmse") is not None:
        cols[0].metric("RMSE (ratings)", f"{float(m['rmse']):.3f}")
        st.caption("Lower RMSE indicates better reconstruction of user–item ratings.")

        shown = True

    #ranking-style metrics
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




import re

st.subheader("Sync latest cloud metrics")
if st.button("Pull newest metrics.json from GCS"):
    user = os.getenv("USER", "sri99")
    best_metrics = f"gs://cityeats-{user}/artifacts/runs/best/metrics/metrics.json"

    try:
        #trying frozen best metrics first
        _ = subprocess.check_output(["gcloud","storage","ls", best_metrics], text=True)
        subprocess.check_call(["gcloud","storage","cp", best_metrics, "artifacts_demo/metrics.json"])
        st.success("Pulled frozen BEST metrics.json from GCS.")
        st.caption(best_metrics)
    except subprocess.CalledProcessError:
        #fallback - looking for a Spark part-*.json produced by a run
        try:
            ls_out = subprocess.check_output(
                ["bash","-lc",
                 f"gcloud storage ls -r gs://cityeats-{user}/artifacts/**/metrics/ | grep -E 'part-.*\\.json$' | sort | tail -n1"],
                text=True
            ).strip()

            if not ls_out:
                st.warning("No metrics part file found in GCS (yet).")
            else:
                subprocess.check_call(["gcloud","storage","cp", ls_out, "artifacts_demo/metrics.json"])
                st.success("Pulled latest Spark metrics part file from GCS.")
                st.caption(ls_out)
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to pull metrics from GCS: {e}")

#show metrics
try:
    import json, pathlib
    mj = json.loads(pathlib.Path("artifacts_demo/metrics.json").read_text())
    colA, colB = st.columns([1,5])
    with colA:
        st.metric("RMSE (ratings)", f"{mj.get('rmse', 0):.3f}")
    with colB:
        st.caption("Lower RMSE indicates better reconstruction of user–item ratings.")
    with st.expander("Run details / hyperparameters"):
        st.write(f"**rank:** {mj.get('rank','?')}")
        st.write(f"**regParam:** {mj.get('regParam','?')}")
        st.write(f"**maxIter:** {mj.get('maxIter','?')}")
except Exception:
    pass


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
