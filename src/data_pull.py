"""
Pull the BANKING77 dataset: ~13,000 real banking customer service queries
labeled with 77 fine-grained intents.

Source: PolyAI-LDN/task-specific-datasets (GitHub), accompanying the paper
"Efficient Intent Detection with Dual Sentence Encoders" (Casanueva et al.,
2020). No API key or account required.

Note: this project originally targeted the CFPB Consumer Complaint
Database, but CFPB pulled down its live search API and dropped the
narrative field from its published schema in late August 2026 (see
README). BANKING77 is a more stable substitute -- and arguably a closer
match to real banking message-intent work anyway.
"""

import io
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def pull_banking77() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for split, filename in [("train", "train.csv"), ("test", "test.csv")]:
        resp = requests.get(f"{BASE_URL}/{filename}", timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df["split"] = split
        frames.append(df)
        print(f"Pulled {len(df)} rows from {filename}")

    full_df = pd.concat(frames, ignore_index=True)
    out_path = RAW_DIR / "banking77_raw.csv"
    full_df.to_csv(out_path, index=False)
    print(f"Saved {len(full_df)} total rows to {out_path}")
    return full_df


if __name__ == "__main__":
    pull_banking77()
