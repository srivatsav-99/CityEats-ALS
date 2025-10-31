import os, subprocess, io
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="CityEats-ALS — Demo", layout="wide")

# --- Paths ---
LOCAL_RECS = "recs_top10_sample_csv/sample.csv"
MAP_USER = "map_user/map_user.csv"
MAP_ITEM = "map_item/map_item.csv"
BUCKET = f"gs://cityeats-{os.getenv('USER','sri99')}"
ARTIFACTS = f"{BUCKET}/artifacts"

st.title("CityEats-ALS — Scalable Food Recommender (Demo)")
st.caption("Small CSV demo while the full Spark pipeline runs in cloud. Built by Srivatsav Shrikanth.")

# --- Sidebar: Cloud hooks ---
st.sidebar.header("Cloud Storage")
if st.sidebar.button("List cloud artifacts"):
    try:
        out = subprocess.check_output(["gcloud", "storage", "ls", "-r", f"{ARTIFACTS}/"], text=True)
        st.sidebar.text(out if out.strip() else "(empty)")
    except subprocess.CalledProcessError as e:
        st.sidebar.error((e.output or str(e)).strip())

# --- Data loaders ---
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
    df = recs.loc[recs["user_id"].eq(user)].sort_values("score", ascending=False).head(k).copy()
    pretty = df[["item_id","score"]].copy()
    pretty.rename(columns={"item_id":"recommended_item"}, inplace=True)
    st.dataframe(pretty, use_container_width=True, hide_index=True)

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("score:Q", title="Predicted score"),
        y=alt.Y("item_id:N", sort="-x", title="Item ID")
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download this Top-K CSV", data=csv_bytes, file_name="topk.csv", mime="text/csv")

if show_idx:
    st.caption("Internal indices preview")
    with st.expander("Show merged demo frame"):
        st.dataframe(df, use_container_width=True)

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
