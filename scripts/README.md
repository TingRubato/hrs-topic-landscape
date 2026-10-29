# scripts/

The pipeline. Each script is one step; run them in numerical order from the repository root
(`python scripts/03_embed.py`). Every step reads what the previous steps wrote into `data/` and
skips work whose output already exists, so an interrupted step can simply be started again.

| Step | Script | Reads | Writes | Run time* |
|---|---|---|---|---|
| 1 | `01_repair_abstracts.py` | `raw/articles.csv` | `raw/articles_repaired.csv`, `raw/repair_log.csv` | ~1 h (web requests) |
| 2 | `02_build_corpus.py` | `raw/articles_repaired.csv` | `corpus.csv` | seconds |
| 3 | `03_embed.py` | `corpus.csv` | `embeddings/<model>.npy`, `embeddings/tfidf.npz` | ~1 h on a GPU |
| 4 | `04_grid.py` | `embeddings/` | `grid/umap/`, `grid/labels/`, `grid/grid_summary.csv` | ~1 h on a CPU |
| 5 | `05_build_pairs.py` | `corpus.csv`, `embeddings/` | `validation/anchors.csv`, `pairs.csv`, `pairs_to_rate.csv` | ~1 min |
| 6 | `06_rate_pairs.py` | `validation/pairs_to_rate.csv` | `validation/ratings/` | depends on the API |
| 7 | `07_evaluate.py` | `grid/`, `validation/` | `validation/grid_scores.csv`, `summary.json`, `shortlist.json` | ~1 min |
| 8 | `08_stability.py` | `validation/shortlist.json`, `embeddings/`, `grid/` | `stability/stability_summary.csv` | ~5 min per setting |
| 9 | `09_final_model.py` | `corpus.csv`, `embeddings/`, `grid/` | `final/` | seconds |

\* Rough times for 6,303 abstracts on a recent laptop.

## What each step does and why

**01_repair_abstracts.py.** The scraped abstracts carry scraping damage: deleted symbols
(`P 0.05` for `P < 0.05`), copyright lines, section headers glued to the next word, and mojibake.
Damaged text changes embeddings and topic words, so it is repaired before anything else. The first
pass replaces flagged abstracts with their PubMed version. The second pass looks up abstracts that
still have deleted symbols in Europe PMC, OpenAlex, Crossref and Semantic Scholar, and accepts a
result only when the title and most of the wording match (so a wrong paper is never swapped in).
It then applies deterministic text fixes to every abstract. `--no-fetch` runs the text fixes only.

**02_build_corpus.py.** Drops exact duplicates, duplicate title + abstract pairs, records without
a title or abstract, and the papers listed in `config.OFF_DOMAIN_IDS`. Builds `embedding_text`
(title, blank line, abstract, HTML stripped, whitespace collapsed). The row order of `corpus.csv`
is the row order of every array written later.

**03_embed.py.** Encodes every abstract with nine pretrained models and fits a TF-IDF baseline.
Comparing several model families (general-purpose sentence encoders, scientific-paper encoders,
biomedical encoders, a lexical baseline) shows whether the topic structure depends on the choice
of model. Options: `--models mpnet,minilm`, `--device cuda|mps|cpu`, `--batch-size N`.

**04_grid.py.** UMAP reduces each embedding to 2-D, then HDBSCAN finds dense clusters (topics) and
leaves sparse points as outliers. Both have settings that change the result a lot, so every
combination is run rather than picking one by eye: 12 UMAP x 30 HDBSCAN = 360 per model.

**05_build_pairs.py.** Builds an outside yardstick for judging the 3,600 clusterings. Anchors are
spread over HRS content areas so no single field dominates. Candidates come from all ten models'
nearest neighbours, mixing easy, ambiguous and hard pairs, so the benchmark does not favour the
model that generated it.

**06_rate_pairs.py.** Asks a language model whether two abstracts address the same research
question (2), share an exposure or an outcome (1), or differ (0). Works with any
OpenAI-compatible endpoint; see the docstring for the environment variables. The raw responses
are saved, so the scores can be re-parsed or audited.

**07_evaluate.py.** A good topic model should put 2-rated pairs in the same topic and keep
0-rated pairs apart. Balanced accuracy averages those two rates. The script scores every
clustering, adds bootstrap confidence intervals, estimates how optimistic "pick the best of
3,600" is by cross-validation, and writes the shortlist for step 8.

**08_stability.py.** A setting that only works for one random seed is not trustworthy. Each
shortlisted setting is refit with four more seeds and on three 80% subsamples, and the
clusterings are compared (ARI, topic-level Jaccard, accuracy across seeds). `--only id1,id2`
limits the run to some shortlist entries.

**09_final_model.py.** Writes everything needed to describe and plot the chosen model
(`config.FINAL_SETTING`): topic sizes and top terms, which topic each paper is in, the 2-D map,
topic centroids and their similarities, and how each topic's share changes over the years.

## Why every script starts with `sys.path.insert`

The scripts import the shared package `hrs_topics/` from the repository root. Adding the root to
Python's search path lets them run as `python scripts/NN_name.py` without installing anything. The
`# noqa: E402` comments tell linters that the imports after that line are intentionally not at the
top of the file.
