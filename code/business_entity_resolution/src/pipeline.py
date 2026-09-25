"""
End-to-End Business Entity Resolution Pipeline.

Unified entry point:
- Training: normalize -> block -> extract features -> train LightGBM -> sweep threshold.
- Inference: index test targets -> block test S1 -> extract features -> predict -> global assign -> validate.
"""

import os
import sys
import argparse
import subprocess

from split_data import analyze_and_split
from train import train_matching_model
from predict import run_inference


def run_pipeline(
    mode: str = "all",
    train_dir: str = "student_resource/dataset/train",
    test_dir: str = "student_resource/dataset/test",
    output_dir: str = "output",
    train_size: int = 15000,
    val_size: int = 3000,
    threshold: float = None,
) -> None:
    if mode in ("split", "all"):
        print("\n===============================")
        print(" STEP 1: Creating Data Splits")
        print("===============================\n")
        analyze_and_split(
            data_dir=train_dir,
            output_dir="cache/splits",
            val_size=50000,
        )

    if mode in ("train", "all"):
        print("\n===============================")
        print(" STEP 2: Training Matcher Model")
        print("===============================\n")
        train_matching_model(
            data_dir=train_dir,
            splits_dir="cache/splits",
            models_dir="cache/models",
            n_train_entities=train_size,
            n_val_entities=val_size,
        )

    if mode in ("predict", "all"):
        print("\n===============================")
        print(" STEP 3: Running Test Inference")
        print("===============================\n")
        run_inference(
            test_dir=test_dir,
            model_path="cache/models/lgb_matcher.pkl",
            output_dir=output_dir,
            override_threshold=threshold,
        )

        print("\n===============================")
        print(" STEP 4: Validating Submission")
        print("===============================\n")
        validator_script = os.path.join("student_resource", "utils", "validate_submission.py")
        if os.path.exists(validator_script):
            cmd = [
                sys.executable,
                validator_script,
                "--matching", os.path.join(output_dir, "matching_results.tsv"),
                "--candidate", os.path.join(output_dir, "candidate_pairs.tsv"),
                "--test-dir", test_dir,
            ]
            print("Running:", " ".join(cmd))
            ret = subprocess.run(cmd)
            if ret.returncode == 0:
                print("\n>>> SUBMISSION VALIDATION PASSED (Exit 0) <<<")
            else:
                print(f"\n>>> SUBMISSION VALIDATION FAILED with code {ret.returncode} <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["split", "train", "predict", "all"], default="all")
    parser.add_argument("--train-dir", default="student_resource/dataset/train")
    parser.add_argument("--test-dir", default="student_resource/dataset/test")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--train-size", type=int, default=15000)
    parser.add_argument("--val-size", type=int, default=3000)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        threshold=args.threshold,
    )
