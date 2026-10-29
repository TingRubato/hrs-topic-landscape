# HRS Topic Landscape

What do researchers study with the Health and Retirement Study (HRS)? This repository maps the
research topics in more than 6,000 journal articles that use HRS data, by grouping the articles'
abstracts by meaning.

**Authors:** Ting Xu, Ziming Wang

## Approach

Topic models can be tuned to give almost any number of topics, and the result depends heavily on
which text-embedding model and which clustering settings are used. Instead of picking one setup by
eye, the pipeline tries many and judges them against an outside yardstick:

1. **Collect and clean the abstracts.** Scrape the HRS bibliography, then repair text damaged
   during scraping (deleted symbols, copyright lines, garbled characters) by re-fetching abstracts
   from PubMed and other databases.
2. **Represent each abstract as a vector** with ten models: general-purpose sentence encoders
   (MPNet, MiniLM, E5, BGE), models trained on scientific papers (SPECTER2, SciNCL, SciBERT),
   a biomedical model (BioMedBERT) and a word-count baseline (TF-IDF).
3. **Cluster every representation many ways.** UMAP maps the vectors to two dimensions and
   HDBSCAN finds dense groups (topics); abstracts in sparse regions are left as outliers.
   12 UMAP x 30 HDBSCAN settings x 10 models = 3,600 clusterings.
4. **Build a yardstick.** 800 pairs of abstracts are rated for whether they address the same
   research question. A good topic model should put strongly related pairs in the same topic and
   keep unrelated pairs apart. Balanced accuracy measures both at once.
5. **Check stability.** The best settings are refit with other random seeds and on random 80%
   subsets of the corpus. Topics that vanish under small changes are not trustworthy.
6. **Describe the chosen model.** Topic sizes, top terms, trends over time, a 2-D map, and
   similarities between topics.

## Result

The final model uses `all-mpnet-base-v2` embeddings, UMAP (n_neighbors 15, min_dist 0) and
HDBSCAN (min_cluster_size 50, min_samples 5, cluster_selection_epsilon 0), seed 42.

| | |
|---|---|
| Abstracts | 6,303 |
| Topics | 38 |
| Outliers | 709 (11.2%) |
| Balanced accuracy against the rated pairs | 0.784 (95% CI 0.731–0.835) |
| Best of all 3,600 settings, cross-validated | 0.786 (95% CI 0.735–0.834) |
| Agreement across UMAP seeds (ARI) | 0.72 |
| Agreement on 80% subsamples (ARI) | 0.77 |
| Topics recovered across seeds (Jaccard ≥ 0.5) | 82% |

The cross-validated accuracy shows that picking the best of 3,600 settings is not just luck: a
setting chosen on part of the pairs does as well on the pairs held out.

## Repository layout

```
scraper/       step 0: collect article metadata and abstracts from the HRS bibliography
scripts/       steps 1-9: the pipeline, one script per step, run in numerical order
hrs_topics/    code shared by the steps: settings, models, clustering, topic words, metrics
data/          inputs and outputs of the steps; the scraped, repaired and final corpus are included
tests/         checks of the evaluation metrics
```

Each folder has its own README with details. `hrs_topics/config.py` holds every setting.

## Getting started

Requires Python 3.12. A GPU (CUDA or Apple MPS) is recommended for step 3 only.

```bash
git clone https://github.com/TingRubato/hrs-topic-landscape.git
cd hrs-topic-landscape
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests
```

Run every command from the repository root. Each step writes the inputs of the next one into
`data/`; set `HRS_DATA_DIR` to use another folder. Steps skip work whose output already exists,
so an interrupted step can be started again.

**Quick start from the included data.** The repository includes the scraped articles
(`data/raw/articles.csv`), the repaired version with its change log (`articles_repaired.csv`,
`repair_log.csv`) and the corpus (`data/corpus.csv`). You can skip scraping and repair and begin at
step 3:

```bash
python scripts/03_embed.py
python scripts/04_grid.py
python scripts/05_build_pairs.py
python scripts/06_rate_pairs.py      # needs an LLM API, see step 6
python scripts/07_evaluate.py
python scripts/08_stability.py
python scripts/09_final_model.py
```

## Steps in detail

**0. Collect the articles** (`scraper/`). Crawls the HRS bibliography in a visible Chrome window,
saving each journal article's metadata and abstract to a SQLite database, then exports it to CSV.
It pauses if a site check appears and resumes after interruptions. Keep the request rate low and
follow the site's terms of use.
```bash
mkdir -p data/raw && cd scraper
python hrs_scraper.py crawl
python hrs_scraper.py --csv ../data/raw/articles.csv export
cd ..
```

**1. Repair abstracts.** Abstracts damaged during scraping would distort both the embeddings and
the topic words. A first pass replaces flagged abstracts with their PubMed version. A second pass
looks up abstracts that still have deleted symbols in Europe PMC, OpenAlex, Crossref and Semantic
Scholar, and accepts a result only if its title and most of its wording match, so a different
paper is never swapped in. Deterministic fixes then remove copyright lines, split glued section
headers and decode garbled characters. Every change is logged; the scraped file is never modified.
```bash
python scripts/01_repair_abstracts.py            # --no-fetch: text fixes only, no web requests
```

**2. Build the corpus.** Removes duplicates, the 148 records without an abstract, and one
off-domain paper (document 7281, a parasitology study), leaving 6,303 abstracts. The text the
models read is the title, a blank line and the abstract.
```bash
python scripts/02_build_corpus.py
```

**3. Embed.** Each model gets the input format and pooling its authors recommend; for example
SPECTER2 reads "title [SEP] abstract" and E5 expects a "passage: " prefix. Long abstracts are cut
at each model's token limit. All vectors are scaled to unit length.
```bash
python scripts/03_embed.py --device cuda          # or mps / cpu; --models mpnet,minilm for a subset
```

**4. Grid.** For each model: 12 UMAP settings (n_neighbors 15/30/50 x min_dist 0/0.1/0.25/0.5)
and on each map 30 HDBSCAN settings (min_cluster_size 5/10/20/30/50 x min_samples 5/10/20 x
cluster_selection_epsilon 0/0.1). Larger values give fewer, broader topics.
```bash
python scripts/04_grid.py
```

**5. Relevance pairs.** 100 anchor abstracts are drawn evenly from 20 content areas of the HRS
(demographics, health, cognition, family, work and retirement, wealth, and so on), matched by
title keywords. For each anchor, eight partner abstracts are chosen from the nearest neighbours of
all ten models: three that most models agree on, two that only one or two models find, two
near-misses, and one random abstract as a sanity check. Drawing partners from every model keeps
the yardstick from favouring any one of them.
```bash
python scripts/05_build_pairs.py
```

**6. Rate the pairs.** A language model rates each pair: 2 = same research question (same main
exposure and outcome), 1 = same exposure or same outcome, 0 = different questions. The prompt is
fixed and saved with its hash, temperature is 0, and every raw response is kept so the ratings can
be audited or re-parsed. Any OpenAI-compatible endpoint works.
```bash
export RATER_BASE_URL=https://... RATER_MODEL=... RATER_API_KEY=...
python scripts/06_rate_pairs.py                  # --run r2 for a second rating, to check agreement
```

**7. Evaluate.** For every clustering: the share of 2-rated pairs placed in the same topic, the
share of 0-rated pairs kept apart, and their average (balanced accuracy). 1-rated and random pairs
are left out of the score. Confidence intervals come from resampling anchors. Five-fold
cross-validation over anchors estimates how optimistic it is to pick the best of 3,600. The script
also writes the shortlist for step 8: MPNet's three best settings, the final setting, and each other
model's best.
```bash
python scripts/07_evaluate.py
```

**8. Stability.** Each shortlisted setting is refit with UMAP seeds 0, 123, 456 and 2026 and on
three random 80% subsets. Reported: agreement between runs (adjusted Rand index, outliers left
out), how well each topic is found again (Jaccard overlap), the range of topic counts, and the
accuracy across seeds.
```bash
python scripts/08_stability.py
```

**9. Final model.** Writes to `data/final/`: the topic table (size, top 10 terms, years, example
titles), each article's topic, the 2-D map, topic centroids and their cosine similarities, and each
topic's yearly share with a 5-year moving average. Short topic names are assigned by hand from the
top terms and example titles.
```bash
python scripts/09_final_model.py
```

## Reproducibility

- All randomness uses seed 42. With a fixed seed UMAP runs single-threaded, and with the library
  versions in `requirements.txt` its output is identical from run to run.
- Embeddings computed on different hardware can differ in the last decimals. Everything after step
  3 is deterministic given the embeddings.
- A language model that reasons before answering does not give identical ratings on every run.
  Steps 7–9 give identical results when re-run on saved ratings.

## Citation and license

Citation details are in `CITATION.cff` (GitHub shows them under "Cite this repository").
The code is under the MIT license; see `LICENSE`. Abstracts in `data/` remain the copyright of
their publishers.
