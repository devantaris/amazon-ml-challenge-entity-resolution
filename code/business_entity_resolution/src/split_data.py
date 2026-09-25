"""
Data Splitting & Sanity Verification.

Creates a reproducible, stratified train/validation split on Source 1 entities:
- Stratified on (country, is_singleton).
- Ensures no data leakage between train and val transforms.
- Outputs summary statistics: row counts, missing rates, singleton ratio, country distributions.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def analyze_and_split(
    data_dir: str = "student_resource/dataset/train",
    output_dir: str = "cache/splits",
    val_size: int = 50000,
    random_state: int = 42,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    s1_path = os.path.join(data_dir, "train_source1.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")

    print(f"Loading {s1_path}...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"Source 1 records: {len(df_s1):,}")

    print(f"Loading {gt_path}...")
    df_gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"Ground truth records: {len(df_gt):,}")

    # Merge on entity_id
    df_merged = df_s1[["entity_id", "country"]].rename(columns={"entity_id": "source1_entity_id"})
    df_merged = df_merged.merge(df_gt[["source1_entity_id", "matched_entity_ids"]], on="source1_entity_id", how="left")

    # Characterize matches
    df_merged["matched_entity_ids"] = df_merged["matched_entity_ids"].fillna("")
    df_merged["is_singleton"] = (df_merged["matched_entity_ids"].str.strip() == "").astype(int)

    def count_matches(m_str: str) -> int:
        s = m_str.strip()
        if not s:
            return 0
        return len(s.split(","))

    df_merged["num_matches"] = df_merged["matched_entity_ids"].apply(count_matches)

    print("\n--- Summary Statistics ---")
    print(f"Total S1 entities: {len(df_merged):,}")
    print(f"Country distribution:\n{df_merged['country'].value_counts(dropna=False)}")
    print(f"\nSingleton distribution:\n{df_merged['is_singleton'].value_counts(normalize=True)}")
    print(f"Average matches per non-singleton: {df_merged[df_merged['is_singleton'] == 0]['num_matches'].mean():.2f}")

    # Stratification key: country + is_singleton
    df_merged["strata"] = df_merged["country"] + "_" + df_merged["is_singleton"].astype(str)

    val_fraction = min(val_size / len(df_merged), 0.2)
    print(f"\nSplitting validation set with val_size={val_size} ({val_fraction:.2%})...")

    train_df, val_df = train_test_split(
        df_merged,
        test_size=val_fraction,
        random_state=random_state,
        stratify=df_merged["strata"],
    )

    print(f"Train S1 entities: {len(train_df):,}")
    print(f"Val S1 entities: {len(val_df):,}")

    print("\nVal Strata distribution:")
    print(val_df["strata"].value_counts(normalize=True))

    train_ids_path = os.path.join(output_dir, "train_s1_ids.txt")
    val_ids_path = os.path.join(output_dir, "val_s1_ids.txt")

    with open(train_ids_path, "w", encoding="utf-8") as f:
        for eid in train_df["source1_entity_id"]:
            f.write(f"{eid}\n")

    with open(val_ids_path, "w", encoding="utf-8") as f:
        for eid in val_df["source1_entity_id"]:
            f.write(f"{eid}\n")

    # Also save a metadata parquet for quick lookups
    val_df[["source1_entity_id", "country", "is_singleton", "num_matches"]].to_parquet(
        os.path.join(output_dir, "val_meta.parquet"), index=False
    )
    print(f"\nSaved IDs to {train_ids_path} and {val_ids_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="student_resource/dataset/train")
    parser.add_argument("--output-dir", default="cache/splits")
    parser.add_argument("--val-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analyze_and_split(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        val_size=args.val_size,
        random_state=args.seed,
    )
