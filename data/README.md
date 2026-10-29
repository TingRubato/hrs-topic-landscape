# data/

Where the pipeline reads and writes its data. The scraped articles, their repaired version, the
repair log and the corpus are in the repository; everything else is created by the steps (and
ignored by git). Set the environment
variable `HRS_DATA_DIR` to keep the data somewhere else.

## corpus.csv (included)

The analysis corpus written by step 2: 6,303 journal articles from the HRS bibliography that
have an abstract. One row per article.

| Column | Meaning |
|---|---|
| `document_id` | Article id in the HRS bibliography; used as the key everywhere |
| `title` | Title as scraped |
| `abstract` | Abstract after repair (step 1) |
| `embedding_text` | Title + blank line + abstract, HTML removed and whitespace collapsed; the text the models read |
| `year` | Publication year (empty when the bibliography gives none) |
| `doi` | DOI, when available |
| `journal` | Journal name as given in the bibliography |

Abstracts remain the copyright of their publishers.

## raw/ (included)

| File | Contents |
|---|---|
| `raw/articles.csv` | The scraper export (step 0): 6,452 HRS bibliography records with `article_id`, `url`, `title`, `authors`, `journal`, `year`, `doi`, `pubmed_id`, `abstract` and other bibliographic fields. Never modified. |
| `raw/articles_repaired.csv` | The same records after step 1, with repaired abstracts. |
| `raw/repair_log.csv` | One row per abstract the repair touched: `step` (`pubmed` = first pass, `sources+local` = second pass), `status` (`replaced` / `not found` in PubMed; `changed` / `still damaged` after the second pass), `source` it was fetched from (empty for text fixes only) and `damage_left` (problems the repair could not fix). |

## Created by the pipeline

```
raw/articles.csv            step 0  scraper export (never modified)
raw/articles_repaired.csv   step 1  abstracts after repair
raw/repair_log.csv          step 1  one row per changed or still-damaged abstract
corpus.csv                  step 2
embeddings/                 step 3  <model>.npy (float32, one row per document), tfidf.npz (sparse)
grid/umap/                  step 4  <model>__<umap setting>.npy, 2-D coordinates
grid/labels/                step 4  <model>__<umap>__<hdbscan>.npy, topic labels (-1 = outlier)
grid/grid_summary.csv       step 4  topics and outlier share of every setting
validation/                 steps 5-7  anchors, pairs, ratings, scores of every setting, shortlist
stability/                  step 8  refitted labels, stability_summary.csv
final/                      step 9  the final topic model (see scripts/README.md)
```

Embeddings and grid outputs take about 250 MB.
