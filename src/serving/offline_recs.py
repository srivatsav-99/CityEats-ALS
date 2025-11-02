import os, glob, json
import numpy as np
import pandas as pd

def _read_all_parquet_parts(dir_path, expected_cols=None):
    local = dir_path.replace("file:///", "").replace("file://", "")
    parts = sorted(glob.glob(os.path.join(local, "*.parquet")))
    if not parts:
        raise FileNotFoundError(f"No parquet files in {local}")
    dfs = [pd.read_parquet(p, engine="pyarrow") for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns {missing}; got {list(df.columns)}")
    return df

def _normalize_map_cols(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    # kind: "user" or "item"
    rename = {}
    if kind == "user":
        if "userIndex" in df.columns and "user_idx" not in df.columns:
            rename["userIndex"] = "user_idx"
        if "user_id" not in df.columns and "userId" in df.columns:
            rename["userId"] = "user_id"
    else:
        if "itemIndex" in df.columns and "item_idx" not in df.columns:
            rename["itemIndex"] = "item_idx"
        if "business_id" in df.columns and "item_id" not in df.columns:
            rename["business_id"] = "item_id"
        if "itemId" in df.columns and "item_id" not in df.columns:
            rename["itemId"] = "item_id"

    df = df.rename(columns=rename)
    for c in ("user_idx", "item_idx"):
        if c in df.columns and pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("int64")
    if kind == "user":
        need = {"user_id", "user_idx"}
    else:
        need = {"item_id", "item_idx"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns {missing}; got {list(df.columns)}")
    return df

def _read_map_dir(dir_path: str, kind: str) -> pd.DataFrame:
    """
    Read a map directory that may contain parquet parts (preferred) OR a CSV
    named map_user.csv / map_item.csv.
    """
    local = dir_path.replace("file:///", "").replace("file://", "")
    # 1) parquet
    parts = sorted(glob.glob(os.path.join(local, "*.parquet")))
    if parts:
        df = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in parts], ignore_index=True)
        return _normalize_map_cols(df, kind)
    # 2) csv
    csv_name = "map_user.csv" if kind == "user" else "map_item.csv"
    csv_path = os.path.join(local, csv_name)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return _normalize_map_cols(df, kind)
    raise FileNotFoundError(f"No parquet or {csv_name} found in {local}")

def load_bundle(bundle_dir):
    # Factors (always parquet written by Spark ALS)
    item_f = _read_all_parquet_parts(os.path.join(bundle_dir, "model", "itemFactors"),
                                     expected_cols=["id","features"])
    user_f = _read_all_parquet_parts(os.path.join(bundle_dir, "model", "userFactors"),
                                     expected_cols=["id","features"])
    # Maps (parquet OR csv)
    map_user = _read_map_dir(os.path.join(bundle_dir, "map_user"), kind="user")
    map_item = _read_map_dir(os.path.join(bundle_dir, "map_item"), kind="item")

    # Convert features list< double > to numpy arrays
    item_idx = item_f["id"].to_numpy(dtype=np.int64)
    item_mat = np.vstack(item_f["features"].to_numpy())
    user_idx = user_f["id"].to_numpy(dtype=np.int64)
    user_mat = np.vstack(user_f["features"].to_numpy())

    # Fast lookups
    uidx_to_row = {int(i): r for r, i in enumerate(user_idx)}
    iidx_to_itemid = dict(zip(map_item["item_idx"].astype(int), map_item["item_id"].astype(str)))

    return {
        "item_mat": item_mat,
        "item_idx": item_idx,
        "user_mat": user_mat,
        "uidx_to_row": uidx_to_row,
        "map_user": map_user[["user_id","user_idx"]],
        "iidx_to_itemid": iidx_to_itemid,
    }

def recommend(bundle, user_id, k=10, seen_df=None):
    row = bundle["map_user"].loc[bundle["map_user"]["user_id"].astype(str) == str(user_id)]
    if row.empty:
        return {"user_id": user_id, "items": []}
    uidx = int(row.iloc[0]["user_idx"])
    ur = bundle["uidx_to_row"].get(uidx)
    if ur is None:
        return {"user_id": user_id, "items": []}
    uvec = bundle["user_mat"][ur]

    scores = bundle["item_mat"].dot(uvec)
    exclude_set = set()
    if seen_df is not None:
        s = seen_df.loc[seen_df["user_id"].astype(str) == str(user_id), "item_id"]
        exclude_set = set(s.astype(str).tolist())

    top = []
    need = max(k*3, k)
    cand = np.argpartition(scores, -need)[-need:]
    cand = cand[np.argsort(scores[cand])[::-1]]
    for ci in cand:
        item_index = int(bundle["item_idx"][ci])
        item_id = bundle["iidx_to_itemid"].get(item_index)
        if not item_id or item_id in exclude_set:
            continue
        top.append({"item_id": item_id, "score": float(scores[ci])})
        if len(top) == k:
            break
    return {"user_id": user_id, "items": top}
