# Complaint Cluster Labeler

Unsupervised topic clustering of consumer financial complaints, with an LLM
in the loop to turn raw clusters into human-readable labels — explored
through an interactive app.

## Business question

When you have thousands of unlabeled text records and no existing taxonomy,
how do you find the natural groupings *and* make them usable by someone who
isn't going to read all thousand records themselves?

## Finding
*(fill in once the clustering is built — what topics actually emerged?)*

## Insight
*(fill in once labeling is built — what surprised you, or what would you do
differently with more data/time?)*

---

## Why this project

Text like this shows up constantly in analytics work — support tickets,
survey free-text, complaint logs, marketing message content — and almost
never comes pre-labeled. This project builds a small, repeatable pipeline
for going from "pile of raw text" to "structured, explorable topics" using:

- **Clustering** to find structure with no labels required
- **An LLM (Claude API)** to translate cluster centroids into plain-language
  labels and rationale, instead of a human eyeballing every cluster
- **Streamlit** to make the result explorable instead of a static notebook

## Data

[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) —
public, free, no API key required. Complaint narratives are already
PII-redacted by CFPB before publication, so this is safe to use in a public
repo. A cleaned sample lives in `data/processed/` so the project runs without
anyone re-pulling from the API.

## Project structure

```
nlp-cluster-labeler/
├── data/
│   ├── raw/            # gitignored — regenerate with data_pull.py
│   └── processed/      # small, committed sample
├── src/
│   ├── data_pull.py    # pulls from the CFPB API
│   └── clean.py        # dedupes, filters, trims to modeling columns
├── notebooks/          # exploration
├── app/                # Streamlit app (later session)
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Roadmap (session by session)

- [x] **Session 1 — Data pull + cleaning** (this one)
- [ ] **Session 2 — Clustering:** TF-IDF or embeddings + KMeans, pick k,
      sanity-check clusters against `product`/`issue` fields
- [ ] **Session 3 — AI auto-labeling:** send representative docs per
      cluster to the Claude API, generate a label + one-line rationale
- [ ] **Session 4 — Streamlit app:** pick a cluster, see its label, size,
      and representative complaints
- [ ] **Session 5 — Polish:** README findings section, screenshots, deploy

## Usage (session 1)

```bash
python src/data_pull.py --n 3000 --date-min 2024-01-01 --date-max 2026-01-01
python src/clean.py
```

This pulls ~3,000 narratives into `data/raw/complaints_raw.json`, then
cleans and writes a modeling-ready CSV to `data/processed/complaints_clean.csv`.
