"""Settings shared by every step: data locations, models, parameter grids and seeds."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("HRS_DATA_DIR", REPO / "data"))

# data layout (each step writes the inputs of the next one)
RAW_ARTICLES = DATA / "raw" / "articles.csv"            # scraper export, never modified
REPAIRED_ARTICLES = DATA / "raw" / "articles_repaired.csv"
REPAIR_LOG = DATA / "raw" / "repair_log.csv"
CORPUS = DATA / "corpus.csv"
EMBEDDINGS = DATA / "embeddings"                        # <model>.npy, tfidf.npz
GRID = DATA / "grid"                                    # umap/, labels/, grid_summary.csv
VALIDATION = DATA / "validation"                        # anchors, pairs, ratings, scores
STABILITY = DATA / "stability"
FINAL = DATA / "final"

SEED = 42

# Records removed from the corpus after review: 7281 is a parasitology paper with no HRS content.
OFF_DOMAIN_IDS = {7281}

MODELS = ["mpnet", "minilm", "e5_large", "bge_large", "bge_m3",
          "specter2", "scincl", "scibert", "biomedbert", "tfidf"]

# UMAP: 3 x 4 = 12 settings per model. Cosine metric on the embeddings, 2-D output.
UMAP_FIXED = dict(metric="cosine", n_components=2, spread=1.0, init="spectral", low_memory=True)
UMAP_GRID = [{"id": f"nn{nn}_md{md:g}", "n_neighbors": nn, "min_dist": md}
             for nn in (15, 30, 50) for md in (0.0, 0.1, 0.25, 0.5)]

# HDBSCAN on the 2-D UMAP coordinates: 5 x 3 x 2 = 30 settings per UMAP setting.
HDBSCAN_FIXED = dict(metric="euclidean", cluster_selection_method="eom", allow_single_cluster=False)
HDBSCAN_GRID = [{"id": f"mcs{mcs}_ms{ms}_cse{cse:g}", "min_cluster_size": mcs, "min_samples": ms,
                 "cluster_selection_epsilon": cse}
                for mcs in (5, 10, 20, 30, 50) for ms in (5, 10, 20) for cse in (0.0, 0.1)]

# TF-IDF vectors are sparse and very wide; they are reduced with truncated SVD before UMAP.
TFIDF_SVD_COMPONENTS = 300

# The topic model reported in the paper (chosen in step 7, see README).
FINAL_SETTING = ("mpnet", "nn15_md0", "mcs50_ms5_cse0")

# Stability: extra UMAP seeds and 80% subsamples.
STABILITY_SEEDS = (42, 0, 123, 456, 2026)
SUBSAMPLE_FRAC = 0.8
SUBSAMPLE_REPS = 3


def umap_params(umap_id):
    return next(g for g in UMAP_GRID if g["id"] == umap_id)


def hdbscan_params(hdbscan_id):
    return next(g for g in HDBSCAN_GRID if g["id"] == hdbscan_id)
