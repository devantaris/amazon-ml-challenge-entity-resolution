"""
Phase 3b — train the v6 matcher on generated pair shards.

Pipeline:
  1. Load all pairs_*.parquet shards (f0..f41, s1_id, cand_id, label).
  2. Entity-grouped holdout split (5% of entities) for early stopping.
  3. XGBoost (GPU if available) on the full training pairs.
  4. LightGBM on a subsample (CPU budget).
  5. GroupKFold(3) OOF stacking -> LogisticRegression meta-learner
     (grouped by s1_id — fixes the v5 fold-leakage bug).
  6. Save ensemble_v6.pkl with validation metrics.

Run on a Kaggle GPU session (T4): ~1-2 h. CPU-only works but XGBoost falls
back to hist (much slower) — reduce --n-rows if so.
"""

import os
import gc
import glob
import json
import time
import pickle
import argparse

import numpy as np
import polars as pl

WATERMARK_GB = 25.0  # Kaggle GPU sessions have 30 GB RAM


def check_rss(label):
    try:
        import psutil
        gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        print(f"    [rss {gb:.2f} GB] {label}", flush=True)
    except Exception:
        pass


def load_shards(pairs_dir):
    files = sorted(glob.glob(os.path.join(pairs_dir, "pairs_*.parquet")))
    assert files, f"no pairs_*.parquet shards in {pairs_dir}"
    dfs = []
    for f in files:
        df = pl.read_parquet(f)
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df):,} pairs ({df['label'].mean():.2%} pos)", flush=True)
    df = pl.concat(dfs)
    del dfs
    gc.collect()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default="/kaggle/working/cache/pairs")
    ap.add_argument("--models-dir", default="/kaggle/working/models")
    ap.add_argument("--n-rows-lgb", type=int, default=10_000_000)
    ap.add_argument("--n-rows-oof", type=int, default=8_000_000)
    ap.add_argument("--n-est", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    t_start = time.time()

    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit, GroupKFold
    from sklearn.metrics import roc_auc_score

    print("Loading shards...", flush=True)
    df = load_shards(args.pairs_dir)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    print(f"Total: {len(df):,} pairs x {len(feat_cols)} features, "
          f"{df['label'].mean():.2%} positive", flush=True)
    check_rss("after load")

    # entity-grouped holdout for early stopping
    entities = df["s1_id"].unique().to_list()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=args.seed)
    tr_idx, va_idx = next(gss.split(df, groups=df["s1_id"].to_numpy()))
    va_ids = set(df["s1_id"][va_idx].to_list())
    train_df = df.filter(~pl.col("s1_id").is_in(va_ids))
    val_df = df.filter(pl.col("s1_id").is_in(va_ids))
    print(f"Train: {len(train_df):,} pairs | Val(held-out entities): {len(val_df):,} pairs", flush=True)

    X_tr = train_df.select(feat_cols).to_numpy().astype(np.float32)
    y_tr = train_df["label"].to_numpy().astype(np.int32)
    X_va = val_df.select(feat_cols).to_numpy().astype(np.float32)
    y_va = val_df["label"].to_numpy().astype(np.int32)
    del df, train_df, val_df
    gc.collect()
    check_rss("after split")

    pos_w = max(1.0, (y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"scale_pos_weight: {pos_w:.2f}", flush=True)

    use_gpu = os.environ.get("ER_XGB_CPU", "0") != "1"
    xgb_params = dict(
        n_estimators=args.n_est, max_depth=8, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_lambda=1.5, scale_pos_weight=pos_w, random_state=args.seed,
        eval_metric="auc", early_stopping_rounds=60,
    )
    if use_gpu:
        try:
            m = xgb.XGBClassifier(tree_method="hist", device="cuda", **xgb_params)
            m.fit(X_tr[:100_000], y_tr[:100_000])
            use_gpu = True
        except Exception as e:
            print(f"GPU unavailable ({e}) — falling back to CPU hist", flush=True)
            use_gpu = False
    device = "cuda" if use_gpu else "hist"
    print(f"XGBoost device: {device}", flush=True)

    print("\n--- XGBoost ---", flush=True)
    xgb_model = xgb.XGBClassifier(tree_method="hist", device=device, **xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)
    auc_x = roc_auc_score(y_va, xgb_model.predict_proba(X_va)[:, 1])
    print(f"XGB val AUC: {auc_x:.5f} (best iter {xgb_model.best_iteration})", flush=True)
    check_rss("after xgb")

    print("\n--- LightGBM (subsample) ---", flush=True)
    if len(y_tr) > args.n_rows_lgb:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(y_tr), args.n_rows_lgb, replace=False)
        Xl, yl = X_tr[idx], y_tr[idx]
    else:
        Xl, yl = X_tr, y_tr
    lgb_model = lgb.LGBMClassifier(
        n_estimators=600, num_leaves=127, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=40,
        scale_pos_weight=pos_w, random_state=args.seed, verbose=-1,
    )
    lgb_model.fit(Xl, yl, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    auc_l = roc_auc_score(y_va, lgb_model.predict_proba(X_va)[:, 1])
    print(f"LGBM val AUC: {auc_l:.5f}", flush=True)
    del Xl, yl
    gc.collect()

    print("\n--- GroupKFold OOF stacking ---", flush=True)
    if len(y_tr) > args.n_rows_oof:
        rng = np.random.RandomState(args.seed + 1)
        idx = rng.choice(len(y_tr), args.n_rows_oof, replace=False)
        Xo, yo = X_tr[idx], y_tr[idx]
        go = train_df["s1_id"].to_numpy()[idx]
    else:
        Xo, yo, go = X_tr, y_tr, train_df["s1_id"].to_numpy()

    gkf = GroupKFold(n_splits=3)
    oof_x = np.zeros(len(yo), dtype=np.float64)
    oof_l = np.zeros(len(yo), dtype=np.float64)
    for fold, (a, b) in enumerate(gkf.split(Xo, yo, groups=go)):
        px = dict(xgb_params)
        px["n_estimators"] = max(400, args.n_est // 2)
        px.pop("early_stopping_rounds", None)
        fx = xgb.XGBClassifier(tree_method="hist", device=device, **px)
        fx.fit(Xo[a], yo[a])
        oof_x[b] = fx.predict_proba(Xo[b])[:, 1]
        fl = lgb.LGBMClassifier(
            n_estimators=400, num_leaves=127, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, min_child_samples=40,
            scale_pos_weight=pos_w, random_state=args.seed, verbose=-1,
        )
        fl.fit(Xo[a], yo[a])
        oof_l[b] = fl.predict_proba(Xo[b])[:, 1]
        print(f"  fold {fold + 1}/3 done", flush=True)
        del fx, fl
        gc.collect()

    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(np.column_stack([oof_x, oof_l]), yo)
    print(f"meta weights: xgb={meta.coef_[0][0]:.3f}, lgb={meta.coef_[0][1]:.3f}", flush=True)

    # validation metrics through the stack
    p_x = xgb_model.predict_proba(X_va)[:, 1]
    p_l = lgb_model.predict_proba(X_va)[:, 1]
    p_meta = meta.predict_proba(np.column_stack([p_x, p_l]))[:, 1]
    auc_m = roc_auc_score(y_va, p_meta)
    print(f"STACK val AUC: {auc_m:.5f}", flush=True)

    out = os.path.join(args.models_dir, "ensemble_v6.pkl")
    with open(out, "wb") as f:
        pickle.dump({
            "xgb_model": xgb_model,
            "lgb_model": lgb_model,
            "meta_model": meta,
            "val_auc": {"xgb": auc_x, "lgb": auc_l, "stack": auc_m},
            "n_pairs": int(len(y_tr) + len(y_va)),
            "trained": time.strftime("%Y-%m-%d %H:%M"),
        }, f)
    print(f"\nSaved {out}", flush=True)
    print(f"ALL DONE in {(time.time() - t_start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
