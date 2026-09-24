"""
Basic cleaning of the BANKING77 pull, run after data_pull.py.

The data is already fairly clean (short, structured customer service
messages), so this mostly standardizes column names and drops exact
duplicates. The `true_intent` column is kept but should NOT be used for
clustering -- it's ground truth to validate discovered clusters against
in a later session.
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "banking77_raw.csv"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={"text": "message", "category": "true_intent"})

    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() > 0]

    df = df.drop_duplicates(subset="message")
    df = df.reset_index(drop=True)
    df.insert(0, "message_id", df.index)
    return df


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw()
    print(f"Loaded {len(raw_df)} raw records")

    clean_df = clean(raw_df)
    print(f"{len(clean_df)} records after cleaning")
    print(
        f"{clean_df['true_intent'].nunique()} distinct true intents "
        "(held out for later validation, not used in clustering)"
    )

    out_path = PROCESSED_DIR / "messages_clean.csv"
    clean_df.to_csv(out_path, index=False)
    print(f"Saved cleaned data to {out_path}")
