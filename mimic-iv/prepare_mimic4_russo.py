#!/usr/bin/env python3

import argparse
import gzip
import shutil
from pathlib import Path


# MIMIC-IV 1.0 -> filenames expected by mimic-iv-benchmarks
FILES = {
    "core/patients.csv.gz": "PATIENTS.csv",
    "core/admissions.csv.gz": "ADMISSIONS.csv",

    "icu/icustays.csv.gz": "ICUSTAYS.csv",
    "icu/chartevents.csv.gz": "CHARTEVENTS.csv",
    "icu/outputevents.csv.gz": "OUTPUTEVENTS.csv",

    "hosp/labevents.csv.gz": "LABEVENTS.csv",

    # Russo's extract_subjects.py reads diagnoses even if we only care about IHM.
    "hosp/diagnoses_icd.csv.gz": "DIAGNOSES_ICD.csv",
    "hosp/d_icd_diagnoses.csv.gz": "D_ICD_DIAGNOSES.csv",
}


def decompress_copy(src: Path, dst: Path):
    print(f"{src} -> {dst}")

    with gzip.open(src, "rb") as f_in:
        with open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mimic_root",
        type=Path,
        help="Path to the MIMIC-IV 1.0 directory containing core/, hosp/, icu/",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to create for Russo mimic-iv-benchmarks",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    src_root = args.mimic_root
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    missing = []

    for relative_src, target_name in FILES.items():
        src = src_root / relative_src
        dst = out / target_name

        if not src.exists():
            missing.append(src)
            continue

        if dst.exists() and not args.overwrite:
            print(f"Skipping existing: {dst}")
            continue

        decompress_copy(src, dst)

    if missing:
        print("\nWARNING: Missing input files:")
        for path in missing:
            print(f"  {path}")

    print("\nFinished.")
    print(f"Russo-compatible MIMIC directory: {out}")


if __name__ == "__main__":
    main()
