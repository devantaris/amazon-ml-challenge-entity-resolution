"""
Submission Packaging & Final Verification Script.

Creates the official submission archive per ML Challenge 2026 specifications:
<team_name>_submission.zip
  output/
    matching_results.tsv
    candidate_pairs.tsv
  code/
    business_entity_resolution/
      src/
      README.md
      requirements.txt
  Documentation_template.md
"""

import os
import sys
import zipfile
import argparse
import subprocess


def package_submission(
    team_name: str = "wanheda",
    output_dir: str = "output",
    code_dir: str = "code/business_entity_resolution",
    doc_path: str = "Documentation_template.md",
    zip_output_dir: str = ".",
    skip_validation: bool = False,
) -> str:
    # 1. Run validator first
    if not skip_validation:
        validator = os.path.join("student_resource", "utils", "validate_submission.py")
        matching_file = os.path.join(output_dir, "matching_results.tsv")
        candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")

        if not os.path.exists(matching_file):
            print(f"Error: {matching_file} does not exist. Run inference first!")
            sys.exit(1)

        print("--- Running Submission Validator ---")
        cmd = [
            sys.executable,
            validator,
            "--matching", matching_file,
            "--candidate", candidate_file,
            "--test-dir", "student_resource/dataset/test",
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"Validation FAILED with code {ret.returncode}. Aborting packaging.")
            sys.exit(1)
        print("Validation PASSED (Exit 0)!\n")

    # 2. Build ZIP archive
    zip_filename = f"{team_name}_submission.zip"
    zip_path = os.path.join(zip_output_dir, zip_filename)
    print(f"Packaging submission into {zip_path}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add output files
        for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                zf.write(fpath, arcname=f"output/{fname}")
                print(f"  Added output/{fname}")

        # Add documentation template
        if os.path.exists(doc_path):
            zf.write(doc_path, arcname="Documentation_template.md")
            print(f"  Added Documentation_template.md")

        # Add code/business_entity_resolution files
        for root, dirs, files in os.walk(code_dir):
            for file in files:
                if file.endswith((".py", ".txt", ".md")) and not file.startswith("."):
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, start=os.path.dirname(os.path.dirname(code_dir)))
                    zf.write(full_p, arcname=rel_p.replace("\\", "/"))
                    print(f"  Added {rel_p.replace(chr(92), '/')}")

    print(f"\nSuccessfully generated {zip_path} ({os.path.getsize(zip_path) / (1024 * 1024):.2f} MB)")
    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-name", default="wanheda")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    package_submission(
        team_name=args.team_name,
        output_dir=args.output_dir,
        skip_validation=args.skip_validation,
    )
