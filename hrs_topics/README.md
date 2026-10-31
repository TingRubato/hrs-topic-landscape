# hrs_topics/

Code shared by the pipeline steps in `scripts/`. Nothing here runs on its own.

| File | Contents |
|---|---|
| `config.py` | Every setting in one place: the `data/` layout, the list of models, the UMAP and HDBSCAN grids, seeds, the off-domain records removed from the corpus, the final setting and the stability settings. Change a parameter here, not in the scripts. |
| `data.py` | Loading the corpus, an embedding, the UMAP input for a model (TF-IDF is first reduced to 300 dimensions with truncated SVD), and the label files of a grid setting. |
| `embeddings.py` | The nine neural models with their Hugging Face ids, token limits and input formats, the encoder that runs them, and the TF-IDF baseline. |
| `clustering.py` | UMAP (cosine metric, 2-D, seeded) and HDBSCAN (Euclidean on the UMAP map, excess-of-mass selection) with the project's fixed options, and `fit()` for one grid setting. |
| `topics.py` | Topic words by class-based TF-IDF (c-TF-IDF): each topic's abstracts are pooled (outliers form one more class) and terms are weighted by their in-topic frequency times log(1 + A / their count over all topics), the formula of BERTopic's `ClassTfidfTransformer`, so terms common to every topic rank low. Unigrams and bigrams, English stop words plus `hrs_stopwords.txt`, numbers dropped. |
| `metrics.py` | Whether a pair of abstracts is in the same topic, balanced accuracy against the 0/1/2 ratings, adjusted Rand index between two clusterings with outliers left out, and per-topic Jaccard recovery. |
| `hrs_stopwords.txt` | Words that appear in almost every HRS abstract (`hrs`, `data`, `wave`, `respondent`, ...) and would otherwise top every topic's word list. Used only for topic words, not for embeddings. |

## Conventions

- Arrays of embeddings and labels are in the row order of `data/corpus.csv`.
- Topic label `-1` means outlier (HDBSCAN found no dense cluster for that abstract).
- Grid settings are named by their parameters: `nn15_md0` is UMAP with n_neighbors 15 and
  min_dist 0; `mcs50_ms5_cse0` is HDBSCAN with min_cluster_size 50, min_samples 5 and
  cluster_selection_epsilon 0.
- All randomness is seeded with `config.SEED` (42).
