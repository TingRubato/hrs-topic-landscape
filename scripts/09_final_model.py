"""Step 9: write the outputs of the final topic model (config.FINAL_SETTING).

Input : data/corpus.csv, data/embeddings/, data/grid/
Output (data/final/):
  topics.csv             per topic: size, share, top 10 c-TF-IDF terms, mean/min/max year, example titles
  document_topics.csv    document_id, topic_id (-1 = outlier), year, title
  umap.csv               document_id, x, y, topic_id (the 2-D map)
  topic_centroids.npy    one L2-normalised mean embedding per topic, in topic_id order
  cosine_matrix.csv      cosine similarity between topic centroids
  topic_year_shares.csv  each topic's share of assigned abstracts per year, raw and 5-year centred mean
  summary.json           setting, number of topics and outliers

Topic labels (short names) are written by hand from the top terms and example titles.

Usage: python scripts/09_final_model.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrs_topics import config  # noqa: E402
from hrs_topics.data import load_corpus, load_embedding, load_labels, umap_path  # noqa: E402
from hrs_topics.topics import top_words  # noqa: E402


def main():
    model, umap_id, hdbscan_id = config.FINAL_SETTING
    out = config.FINAL
    out.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    labels = load_labels(model, umap_id, hdbscan_id)
    years = pd.to_numeric(corpus.year, errors="coerce").to_numpy(dtype=float)
    assigned = labels != -1

    words = top_words(corpus.embedding_text.fillna("").astype(str).tolist(), labels)
    topics = sorted(words)
    rows = []
    for t in topics:
        mem = np.flatnonzero(labels == t)
        y = years[mem][~np.isnan(years[mem])]
        rows.append({"topic_id": t, "n_docs": len(mem), "share_of_assigned": len(mem) / assigned.sum(),
                     "share_of_corpus": len(mem) / len(labels), "top_words": " ".join(words[t]),
                     "mean_year": y.mean(), "min_year": y.min(), "max_year": y.max(),
                     "example_titles": " | ".join(corpus.title.iloc[mem[:3]].fillna(""))})
    pd.DataFrame(rows).sort_values("n_docs", ascending=False).to_csv(out / "topics.csv", index=False)

    pd.DataFrame({"document_id": corpus.document_id, "topic_id": labels, "year": years,
                  "title": corpus.title}).to_csv(out / "document_topics.csv", index=False)
    xy = np.load(umap_path(model, umap_id))
    pd.DataFrame({"document_id": corpus.document_id, "x": xy[:, 0], "y": xy[:, 1],
                  "topic_id": labels}).to_csv(out / "umap.csv", index=False)

    # centroid = mean of the topic's (unit-length) embeddings, rescaled to unit length
    X = load_embedding(model)
    C = np.vstack([X[labels == t].mean(axis=0) for t in topics]).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    np.save(out / "topic_centroids.npy", C)
    names = [f"t{t}" for t in topics]
    pd.DataFrame(C @ C.T, index=names, columns=names).to_csv(out / "cosine_matrix.csv")

    # yearly shares among assigned abstracts; years with no abstract in a topic count as 0
    ty = pd.DataFrame({"year": years[assigned], "topic_id": labels[assigned]}).dropna()
    ty["year"] = ty.year.astype(int)
    share = pd.crosstab(ty.year, ty.topic_id, normalize="index")
    smooth = share.rolling(5, min_periods=1, center=True).mean()
    pd.DataFrame({"share": share.stack(), "share_smoothed": smooth.stack()}).reset_index() \
        .sort_values(["year", "topic_id"]).to_csv(out / "topic_year_shares.csv", index=False)

    summary = {"model": model, "umap_id": umap_id, "hdbscan_id": hdbscan_id, "seed": config.SEED,
               "n_documents": int(len(labels)), "n_topics": len(topics),
               "n_outliers": int((~assigned).sum()), "outlier_share": float((~assigned).mean())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
