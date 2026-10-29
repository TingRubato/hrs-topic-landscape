"""Step 8: how stable is each shortlisted setting?

Seeds: UMAP + HDBSCAN refit with seeds 0, 123, 456 and 2026 next to the seed-42 run from step 4.
  seed_ARI     mean adjusted Rand index over all seed pairs (outliers excluded)
  topic_jacc   for each seed-42 topic, its best Jaccard match in another seed, averaged
  topics_j50   share of seed-42 topics recovered with Jaccard >= 0.5 (mean over seeds)
  BA_*         balanced accuracy on the rated pairs at seed 42, mean and worst over seeds
Subsamples: three random 80% subsets (seed 42), refit at seed 42.
  sub_ARI      mean ARI between subset labels and the full-corpus labels on the same documents

Input : data/validation/shortlist.json, data/validation/pairs_rated.csv, data/embeddings/, data/grid/
Output: data/stability/<id>__seed<seed>.npy, <id>__sub<rep>.npy   refitted labels (cached)
        data/stability/stability_summary.csv                       one row per setting

Usage: python scripts/08_stability.py [--only mpnet_final,...]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrs_topics import config  # noqa: E402
from hrs_topics.clustering import fit  # noqa: E402
from hrs_topics.data import load_corpus, load_features, load_labels  # noqa: E402
from hrs_topics.metrics import ari_without_outliers, balanced_accuracy, together, topic_recovery  # noqa: E402


def cached(path, compute):
    if path.exists():
        return np.load(path).astype(int)
    lab = compute()
    np.save(path, lab)
    return lab.astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default=None, help="comma-separated shortlist ids")
    a = ap.parse_args()
    S = config.STABILITY
    S.mkdir(parents=True, exist_ok=True)
    shortlist = json.loads((config.VALIDATION / "shortlist.json").read_text())
    if a.only:
        shortlist = [c for c in shortlist if c["id"] in a.only.split(",")]

    pos = {d: i for i, d in enumerate(load_corpus().document_id)}
    pairs = pd.read_csv(config.VALIDATION / "pairs_rated.csv")
    pairs = pairs[pairs.stratum != "random"]
    a_pos, c_pos = pairs.anchor_document_id.map(pos).to_numpy(), pairs.candidate_document_id.map(pos).to_numpy()
    y = pairs.score.to_numpy()
    n = len(pos)

    rows = []
    for c in shortlist:
        cid, m, u, h = c["id"], c["model"], c["umap_id"], c["hdbscan_id"]
        X = load_features(m)
        labs = {42: load_labels(m, u, h)}
        for s in config.STABILITY_SEEDS[1:]:
            labs[s] = cached(S / f"{cid}__seed{s}.npy", lambda: fit(X, u, h, seed=s))
            print(cid, "seed", s, flush=True)

        rng = np.random.default_rng(config.SEED)
        sub_ari = []
        for rep in range(config.SUBSAMPLE_REPS):
            idx = np.sort(rng.choice(n, size=int(config.SUBSAMPLE_FRAC * n), replace=False))
            lab = cached(S / f"{cid}__sub{rep}.npy", lambda: fit(X[idx], u, h))
            sub_ari.append(ari_without_outliers(labs[42][idx], lab))
            print(cid, "subsample", rep, flush=True)

        ba = [balanced_accuracy(together(l, a_pos, c_pos), y) for l in labs.values()]
        jacc = pd.DataFrame([topic_recovery(labs[42], labs[s]) for s in config.STABILITY_SEEDS[1:]]).mean()
        n_topics = [len(np.unique(l[l != -1])) for l in labs.values()]
        rows.append({"id": cid, "setting": f"{m} {u} {h}", "topics": n_topics[0],
                     "seed_topics": f"{min(n_topics)}-{max(n_topics)}",
                     "BA": ba[0], "BA_seeds_mean": np.mean(ba), "BA_seeds_min": min(ba),
                     "seed_ARI": np.nanmean([ari_without_outliers(labs[p], labs[q])
                                             for p, q in itertools.combinations(labs, 2)]),
                     "sub_ARI": np.nanmean(sub_ari), "topic_jacc": jacc.mean(), "topics_j50": (jacc >= 0.5).mean(),
                     "outlier_share": float((labs[42] == -1).mean())})

    out = pd.DataFrame(rows).sort_values("seed_ARI", ascending=False)
    out.to_csv(S / "stability_summary.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
