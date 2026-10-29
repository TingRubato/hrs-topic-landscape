# HRS Topic Landscape

Topic modeling of research published with the Health and Retirement Study (HRS). The pipeline
collects HRS-based journal abstracts, repairs text damaged during scraping, embeds the abstracts
with ten document representations, clusters each one under 360 UMAP + HDBSCAN settings, scores
all 3,600 clusterings against 800 rated abstract pairs, tests the best ones for stability, and
writes the final topic model.

**Author:** Ting Xu

Final model: `all-mpnet-base-v2` embeddings, UMAP (n_neighbors 15, min_dist 0), HDBSCAN
(min_cluster_size 50, min_samples 5, cluster_selection_epsilon 0), seed 42. On 6,303 abstracts it
gives 38 topics and 709 outliers, with balanced accuracy 0.784 (95% CI 0.731–0.835) against the
rated pairs.

## Layout

```
scraper/hrs_scraper.py        collect article metadata and abstracts from the HRS bibliography
scripts/                      the pipeline, one script per step, run in order
  01_repair_abstracts.py      fix abstracts damaged during scraping
  02_build_corpus.py          deduplicate, drop records without an abstract -> corpus.csv
  03_embed.py                 ten embeddings per abstract
  04_grid.py                  UMAP x HDBSCAN grid: 3,600 clusterings
  05_build_pairs.py           100 anchors, 800 anchor-candidate pairs to rate
  06_rate_pairs.py            rate the pairs 0/1/2 with a language model
  07_evaluate.py              balanced accuracy of every clustering; stability shortlist
  08_stability.py             refits with other seeds and 80% subsamples
  09_final_model.py           topics, document assignments, map, centroids, trends
hrs_topics/                   code shared by the steps
  config.py                   paths, models, grids, seeds, final setting
  data.py                     loading the corpus, embeddings and grid outputs
  embeddings.py               the ten models and how each encodes an abstract
  clustering.py               UMAP and HDBSCAN with fixed settings
  topics.py                   topic words by c-TF-IDF
  metrics.py                  balanced accuracy, ARI, topic recovery
  hrs_stopwords.txt           HRS boilerplate words left out of topic words
tests/test_metrics.py
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests
```

Run every command from the repository root. Each step writes the inputs of the next one into
`data/` (set `HRS_DATA_DIR` to use another folder). No data is included in the repository.

## Steps

**0. Collect the articles.** Opens Chrome; resumes after an interruption. See `scraper/README.md`.
Keep the request rate low and follow the site's terms of use.
```bash
cd scraper && python hrs_scraper.py crawl && python hrs_scraper.py export --csv ../data/raw/articles.csv && cd ..
```

**1. Repair abstracts.** A PubMed pass, then Europe PMC / OpenAlex / Crossref / Semantic Scholar
for abstracts with deleted symbols, then deterministic local fixes (copyright tails, glued section
headers, mojibake). A fetched abstract is accepted only if its title and wording match the old one.
Every change is logged. `data/raw/articles.csv` is never modified.
```bash
python scripts/01_repair_abstracts.py
```

**2. Build the corpus.** Removes duplicates, records without an abstract (148) and one off-domain
paper (document 7281), leaving 6,303 abstracts.
```bash
python scripts/02_build_corpus.py
```

**3. Embed.** MPNet, MiniLM, E5-large, BGE-large, BGE-M3, SPECTER2, SciNCL, SciBERT, BioMedBERT
and TF-IDF. Each model gets the input format and pooling its authors recommend
(`hrs_topics/embeddings.py`). Needs a GPU for reasonable run time.
```bash
python scripts/03_embed.py --device cuda      # or mps / cpu
```

**4. Grid.** 12 UMAP settings x 30 HDBSCAN settings per model, seed 42.
```bash
python scripts/04_grid.py
```

**5. Relevance pairs.** 100 anchors drawn from 20 title-keyword strata covering the HRS content
areas; for each, 3 candidates most models retrieve, 2 that only one or two models retrieve, 2
semi-hard negatives and 1 random pair. Pooling over all ten models keeps the benchmark neutral.
```bash
python scripts/05_build_pairs.py
```

**6. Rate the pairs.** 0 = different research question, 1 = same exposure or same outcome,
2 = same question. Fixed prompt, temperature 0; raw responses are kept so a run can be re-parsed.
```bash
export RATER_BASE_URL=... RATER_MODEL=... RATER_API_KEY=...   # any OpenAI-compatible endpoint
python scripts/06_rate_pairs.py
```

**7. Evaluate.** Balanced accuracy of all 3,600 clusterings on the 700 non-random pairs, with
anchor-bootstrap confidence intervals, 5-fold cross-validated selection over anchors, and
sensitivity checks. Writes the stability shortlist: MPNet's three best settings, the final
setting, and each other model's best.
```bash
python scripts/07_evaluate.py
```

**8. Stability.** Four more UMAP seeds and three 80% subsamples per shortlisted setting: ARI,
topic recovery (Jaccard) and balanced accuracy across seeds.
```bash
python scripts/08_stability.py
```

**9. Final model.** Writes the topic table, document assignments, 2-D map, topic centroids and
their cosine similarities, and yearly topic shares to `data/final/`. Topic names are assigned
by hand from the top terms and example titles.
```bash
python scripts/09_final_model.py
```

## Reproducibility

- Seed 42 for UMAP, sampling and bootstraps. With a fixed seed UMAP runs single-threaded and its
  output is identical across runs with the library versions in `requirements.txt`.
- Embeddings computed on different hardware can differ in the last decimals; everything after
  step 3 is deterministic given the embeddings.
- The rater in step 6 reasons internally, so a new rating run will not reproduce every score.
  Re-running steps 7–9 on saved ratings gives identical results.

## License

MIT. See `LICENSE`.
