"""Step 4: cluster every embedding under every UMAP x HDBSCAN setting.

For each model: 12 UMAP settings (n_neighbors x min_dist) give 2-D coordinates; on each of them
30 HDBSCAN settings (min_cluster_size x min_samples x cluster_selection_epsilon) give topic
labels. 10 models x 360 settings = 3,600 clusterings. Seed 42 throughout. Existing outputs are
reused, so the step can be restarted.

Input : data/corpus.csv, data/embeddings/
Output: data/grid/umap/<model>__<umap_id>.npy                     2-D coordinates
        data/grid/labels/<model>__<umap_id>__<hdbscan_id>.npy     labels (-1 = outlier)
        data/grid/grid_summary.csv                                topics and outlier share per setting

Usage: python scripts/04_grid.py [--models mpnet,minilm,...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrs_topics import config  # noqa: E402
from hrs_topics.clustering import cluster, reduce  # noqa: E402
from hrs_topics.data import labels_path, load_features, umap_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(config.MODELS))
    a = ap.parse_args()
    (config.GRID / "umap").mkdir(parents=True, exist_ok=True)
    (config.GRID / "labels").mkdir(parents=True, exist_ok=True)

    for model in a.models.split(","):
        X = None
        for u in config.UMAP_GRID:
            up = umap_path(model, u["id"])
            if up.exists():
                coords = np.load(up)
            else:
                X = load_features(model) if X is None else X
                coords = reduce(X, u["n_neighbors"], u["min_dist"])
                np.save(up, coords)
            for h in config.HDBSCAN_GRID:
                lp = labels_path(model, u["id"], h["id"])
                if not lp.exists():
                    np.save(lp, cluster(coords, h["min_cluster_size"], h["min_samples"],
                                        h["cluster_selection_epsilon"]))
            print(f"{model} {u['id']}: 30 clusterings", flush=True)

    rows = []
    for model in config.MODELS:
        for u in config.UMAP_GRID:
            for h in config.HDBSCAN_GRID:
                lp = labels_path(model, u["id"], h["id"])
                if lp.exists():
                    lab = np.load(lp)
                    rows.append({"model": model, "umap_id": u["id"], "hdbscan_id": h["id"],
                                 "n_topics": int(len(np.unique(lab[lab != -1]))),
                                 "outlier_share": float((lab == -1).mean())})
    pd.DataFrame(rows).to_csv(config.GRID / "grid_summary.csv", index=False)
    print(f"{len(rows)} settings -> {config.GRID / 'grid_summary.csv'}")


if __name__ == "__main__":
    main()
