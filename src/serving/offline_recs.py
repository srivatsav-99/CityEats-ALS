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

def load_bundle(bundle_dir):
    item_f = _read_all_parquet_parts(os.path.join(bundle_dir, "model", "itemFactors"),
                                     expected_cols=["id","features"])
    user_f = _read_all_parquet_parts(os.path.join(bundle_dir, "model", "userFactors"),
                                     expected_cols=["id","features"])
    # maps
    map_user = _read_all_parquet_parts(os.path.join(bundle_dir, "map_user"))
    map_item = _read_all_parquet_parts(os.path.join(bundle_dir, "map_item"))

    # normalize expected names
    map_user = map_user.rename(columns={"userIndex":"user_idx","user_id":"user_id"})
    map_item = map_item.rename(columns={"itemIndex":"item_idx","business_id":"item_id"})

    # convert features (list< double >) to numpy arrays
    item_idx = item_f["id"].to_numpy(dtype=np.int64)
    item_mat = np.vstack(item_f["features"].to_numpy())
    user_idx = user_f["id"].to_numpy(dtype=np.int64)
    user_mat = np.vstack(user_f["features"].to_numpy())

    # fast lookups
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
    # find user_idx
    row = bundle["map_user"].loc[bundle["map_user"]["user_id"] == user_id]
    if row.empty:
        return {"user_id": user_id, "items": []}
    uidx = int(row.iloc[0]["user_idx"])

    # get user vector
    ur = bundle["uidx_to_row"].get(uidx, None)
    if ur is None:
        return {"user_id": user_id, "items": []}
    uvec = bundle["user_mat"][ur]

    # scores = item_mat dot uvec
    scores = bundle["item_mat"].dot(uvec)

    # optional exclude seen
    exclude_set = set()
    if seen_df is not None:
        s = seen_df.loc[seen_df["user_id"] == user_id, "item_id"]
        exclude_set = set(s.astype(str).tolist())

    # pick top-k (ignoring seen)
    top = []
    # get more than k to account for exclusions
    candidate_idx = np.argpartition(scores, -max(k*3, k))[-max(k*3, k):]
    candidate_sorted = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]

    for ci in candidate_sorted:
        item_index = int(bundle["item_idx"][ci])
        item_id = bundle["iidx_to_itemid"].get(item_index)
        if item_id is None or item_id in exclude_set:
            continue
        top.append({"item_id": item_id, "score": float(scores[ci])})
        if len(top) == k:
            break

    return {"user_id": user_id, "items": top}
