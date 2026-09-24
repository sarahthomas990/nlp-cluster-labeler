# Message Cluster Labeler

Unsupervised topic clustering of banking customer messages, with an LLM
in the loop to turn raw clusters into human-readable labels explored
through an interactive app.

## Business question

When you have thousands of unlabeled text records and no existing taxonomy,
how do you find the natural groupings *and* make them usable by someone who
isn't going to read all thousand records themselves?

## Finding

Three approaches to unsupervised topic discovery were compared against
BANKING77's true intent labels (held out, never used for clustering):

| Method | NMI | ARI |
|---|---|---|
| TF-IDF + KMeans | 0.50 | 0.12 |
| Embeddings + KMeans | 0.71 | 0.36 |
| BERTopic | 0.82 | 0.47 |

Quality rose consistently across both metrics as each method incorporated
more context — word overlap, then meaning, then automatic topic discovery.
BERTopic's advantage was large enough to change the plan: instead of the
originally intended fixed k=35, its own topic assignment — 181 topics
discovered without specifying k, plus an explicit "doesn't fit anywhere"
bucket for 12.7% of messages — is what moved forward to labeling.

## Insight

Picking k turned out to be a harder, more interesting problem than picking
an algorithm. Neither the elbow method nor silhouette score gave
TF-IDF+KMeans a clean answer. Inertia declined smoothly with no elbow out
to k=75, and silhouette stayed below 0.11 throughout, indicating weak
cluster separation at any k. That absence of a clean signal is what
actually motivated moving to richer representations, rather than tuning k
harder on a weak baseline.

The labeling step surfaced one more finding almost by accident: even after
BERTopic's own automatic topic-reduction step, a handful of distinct
topics ended up with identical LLM-generated labels (two separate topics
both labeled "Card Decline Issues," for instance), evidence the reduction
didn't fully converge. With more time, I'd re-run it with an explicit
topic count instead of "auto," or merge same-labeled topics as a
post-processing pass.

---

## Why this project

Text like this shows up constantly in analytics work. For example: support tickets,
survey free-text, complaint logs, marketing message content and it is almost
never pre-labeled. This project builds a small, repeatable pipeline
for going from "pile of raw text" to "structured, explorable topics" using:

- **Clustering** to find structure with no labels required
- **An LLM (Claude API)** to translate cluster centroids into plain-language
  labels and rationale, instead of a human eyeballing every cluster
- **Streamlit** to make the result explorable instead of a static notebook (see app folder)

## Data

**BANKING77** — ~13,000 real banking customer service messages, each
labeled with one of 77 fine-grained intents (e.g. `card_arrival`,
`wrong_amount_of_cash_received`). Source: [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets),
from Casanueva et al. 2020. No API key or account required.

**A note on the pivot:** this project originally targeted the CFPB Consumer
Complaint Database. On August 27, 2026, CFPB announced it was scaling back
the public complaint database and ceasing publication of complaint
narratives; its live search API and narrative field were pulled down
shortly after. Rather than build on a government data source in flux,
this project moved to BANKING77 which turns out to be an even closer
analog to real banking message-intent work, since it's messages about
banking issues rather than post-hoc complaint narratives. The true intent
labels are kept in the data but *not* used for clustering, instead they're held
out to validate the discovered clusters later.

## Project structure

nlp-cluster-labeler/
├── app/                # Streamlit app
├── data/
│   ├── raw/            # gitignored — regenerate with data_pull.py
│   └── processed/      # small, committed sample
├── src/
│   ├── data_pull.py    # pulls BANKING77 from GitHub
│   └── clean.py        # dedupes, standardizes columns
├── notebooks/          # exploration
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python src/data_pull.py
python src/clean.py
```

This pulls ~13,000 messages into `data/raw/banking77_raw.csv`, then cleans
and writes a modeling-ready CSV to `data/processed/messages_clean.csv`.

Clustering, validation, and AI labeling happen in the notebook under `notebooks/`.

### Explore the results

```bash
streamlit run app/app.py
```

Opens an interactive explorer over the labeled topics — an Overview tab
with corpus-level stats and topic-size/purity distributions, and a Topic
Detail tab to drill into any single topic's sample messages and
true-intent breakdown.
