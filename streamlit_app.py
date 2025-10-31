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

    # Top-K for the selected user
    df = (
        recs.loc[recs["user_id"].eq(user)]
            .sort_values("score", ascending=False)
            .head(k)
            .copy()
    )

    # Try to find a human-readable item name column if we have one after merges
    # (common fallbacks you might use in other datasets are included)
    readable_item_col = next(
        (c for c in ["item_name", "name", "business_name", "title", "item"]
         if c in df.columns),
        "item_id"  # fallback to internal id if no readable name exists
    )

    if show_idx:
        # INTERNAL view: explicitly show ids
        table = (
            df[["user_id", "item_id", "score"]]
            .rename(columns={
                "user_id": "user_id (internal)",
                "item_id": "item_id (internal)"
            })
        )
        y_field = "item_id"      # chart labels
        y_title = "Item ID (internal)"
    else:
        # PRETTY view: show a readable item label if we have it
        table = (
            df[[readable_item_col, "score"]]
            .rename(columns={readable_item_col: "recommended_item"})
        )
        y_field = readable_item_col
        y_title = "Recommended item"

    # Table
    st.dataframe(table, use_container_width=True, hide_index=True)

    # Chart
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

    # Download (respect the same columns we show in the table)
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
