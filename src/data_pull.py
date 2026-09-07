"""
Pull a sample of consumer complaint narratives from the CFPB Consumer
Complaint Database public API. No API key required.

Docs: https://cfpb.github.io/api/ccdb/
Field reference: https://cfpb.github.io/api/ccdb/fields.html

Note: CFPB already redacts PII in narratives (dates, account numbers,
names, etc. show up as "XX/XX" or "XXXX"), so this data is safe to
commit to a public repo as-is.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import requests

BASE_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
PAGE_SIZE = 100  # API max results per request
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_page(
    frm: int,
    size: int,
    date_min: str,
    date_max: str,
    product: Optional[str],
) -> dict:
    params = {
        "format": "json",
        "has_narrative": "true",
        "frm": frm,
        "size": size,
        "sort": "created_date_desc",
        "date_received_min": date_min,
        "date_received_max": date_max,
        "no_aggs": "true",
        "no_highlight": "true",
    }
    if product:
        params["product"] = product

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def pull_complaints(
    target_count: int = 3000,
    date_min: str = "2024-01-01",
    date_max: str = "2026-01-01",
    product: Optional[str] = None,
) -> list:
    """Pull complaint narratives, paginating through the API.

    The underlying search index has a deep-pagination limit, so if you
    need more records than this returns, narrow the date range or add
    a product filter rather than paging further back.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    frm = 0

    while len(records) < target_count:
        size = min(PAGE_SIZE, target_count - len(records))
        payload = fetch_page(frm, size, date_min, date_max, product)
        hits = payload.get("hits", {}).get("hits", [])
        if not hits:
            print(f"No more results after {len(records)} records.")
            break

        for hit in hits:
            records.append(hit["_source"])

        frm += size
        print(f"Pulled {len(records)} / {target_count}")
        time.sleep(0.5)  # be polite to a public government API

    out_path = RAW_DIR / "complaints_raw.json"
    with out_path.open("w") as f:
        json.dump(records, f)
    print(f"Saved {len(records)} records to {out_path}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull CFPB complaint narratives.")
    parser.add_argument("--n", type=int, default=3000, help="Target number of records")
    parser.add_argument("--date-min", default="2024-01-01")
    parser.add_argument("--date-max", default="2026-01-01")
    parser.add_argument(
        "--product",
        default=None,
        help="Optional product filter, e.g. 'Credit card' or 'Checking or savings account'",
    )
    args = parser.parse_args()

    pull_complaints(
        target_count=args.n,
        date_min=args.date_min,
        date_max=args.date_max,
        product=args.product,
    )
