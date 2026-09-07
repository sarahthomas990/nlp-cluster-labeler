"""
Basic cleaning of the raw CFPB complaint pull, run after data_pull.py.

Produces a de-duplicated, filtered CSV small enough to commit to the
repo as a self-contained working sample (so the project runs without
anyone having to re-hit the API).
"""

import json
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "complaints_raw.json"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

MIN_NARRATIVE_LENGTH = 40  # characters; drops near-empty narratives

KEEP_COLS = [
    "complaint_id",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "company",
    "state",
    "date_received",
    "complaint_what_happened",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    with path.open() as f:
        records = json.load(f)
    return pd.DataFrame(records)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[[c for c in KEEP_COLS if c in df.columns]]
    df = df.rename(columns={"complaint_what_happened": "narrative"})

    df["narrative"] = df["narrative"].astype(str).str.strip()
    df = df[df["narrative"].str.len() >= MIN_NARRATIVE_LENGTH]

    df = df.drop_duplicates(subset="complaint_id")
    df = df.drop_duplicates(subset="narrative")

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw()
    print(f"Loaded {len(raw_df)} raw records")

    clean_df = clean(raw_df)
    print(f"{len(clean_df)} records after cleaning")

    out_path = PROCESSED_DIR / "complaints_clean.csv"
    clean_df.to_csv(out_path, index=False)
    print(f"Saved cleaned data to {out_path}")
