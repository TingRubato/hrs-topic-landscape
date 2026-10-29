"""Step 7: score all 3,600 clusterings against the rated pairs and pick the stability shortlist.

Balanced accuracy (BA) = mean of (share of 2-rated pairs put in the same topic) and (share of
0-rated pairs kept apart), on the high / low / semi pairs. 1-rated pairs and the random sanity
pairs are left out (the random pairs' co-assignment rate is reported separately).

Uncertainty: 2,000 bootstrap resamples of anchors (an anchor's pairs move together).
Selection bias: 5-fold cross-validation over anchors; the setting with the best BA on four folds
is scored on the fifth, giving an honest BA for "pick the best of 3,600".
Sensitivity: BA with 1-rated pairs counted as related or as unrelated, and BA on pairs a model
retrieved itself vs pairs only other models retrieved.

Input : data/grid/, data/validation/pairs.csv, data/validation/ratings/scores_r1.csv
Output: data/validation/grid_scores.csv   every setting: BA, TPR, TNR, co-assignment rates
        data/validation/cv_selection.csv  setting chosen per fold and its held-out BA
        data/validation/summary.json      best setting, final setting, per-model best, held-out BA
        data/validation/shortlist.json    settings for the stability step

Usage: python scripts/07_evaluate.py [--ratings scores_r1.csv]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrs_topics import config  # noqa: E402
from hrs_topics.data import load_corpus, load_labels  # noqa: E402

FOLDS, BOOT = 5, 2000
MAIN = ("high", "low", "semi")
KEEP = ["model", "umap_id", "hdbscan_id", "n_topics", "outlier_share"]


def ba(co, y, w=None):
    """Vectorised BA. co: (settings x pairs) 0/1; y: ratings; w: pair weights, (pairs,) or (reps x pairs)."""
    pos, neg = (y == 2).astype(float), (y == 0).astype(float)
    w = np.ones(len(y)) if w is None else w
    wp, wn = w * pos, w * neg
    tpr = (co @ wp.T) / wp.sum(-1)
    tnr = 1 - (co @ wn.T) / wn.sum(-1)
    return (tpr + tnr) / 2, tpr, tnr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ratings", default="scores_r1.csv")
    a = ap.parse_args()
    V = config.VALIDATION
    pairs = pd.read_csv(V / "pairs.csv")
    scores = pd.read_csv(V / "ratings" / a.ratings).set_index("pair_id").score
    assert scores.notna().all() and len(scores) == len(pairs), "missing ratings"
    pairs = pairs.join(scores, on="pair_id")
    pairs["score"] = pairs.score.astype(int)
    pairs.to_csv(V / "pairs_rated.csv", index=False)

    pos = {d: i for i, d in enumerate(load_corpus().document_id)}
    a_pos, c_pos = pairs.anchor_document_id.map(pos).to_numpy(), pairs.candidate_document_id.map(pos).to_numpy()
    main_m = pairs.stratum.isin(MAIN).to_numpy()
    y = pairs.score.to_numpy()[main_m]

    grid = pd.read_csv(config.GRID / "grid_summary.csv")
    co_all = np.zeros((len(grid), len(pairs)))
    for i, r in enumerate(grid.itertuples()):
        lab = load_labels(r.model, r.umap_id, r.hdbscan_id)
        co_all[i] = (lab[a_pos] == lab[c_pos]) & (lab[a_pos] != -1)
    co = co_all[:, main_m]

    grid["BA"], grid["TPR"], grid["TNR"] = ba(co, y)
    for s in (0, 1, 2):
        grid[f"co_rate_{s}"] = co[:, y == s].mean(1)
    grid["BA_1_as_related"] = ba(co, np.where(y >= 1, 2, 0))[0]
    grid["BA_1_as_unrelated"] = ba(co, np.where(y == 2, 2, 0))[0]
    grid["co_rate_random"] = co_all[:, ~main_m].mean(1)
    grid.to_csv(V / "grid_scores.csv", index=False)

    # anchor bootstrap: weight each pair by how often its anchor is drawn
    anchor = pd.factorize(pairs.anchor_document_id[main_m])[0]
    n_anchor = anchor.max() + 1
    draws = np.random.default_rng(config.SEED).integers(0, n_anchor, size=(BOOT, n_anchor))
    W = np.apply_along_axis(np.bincount, 1, draws, minlength=n_anchor)[:, anchor].astype(float)

    def ci(k):
        b = ba(co[[k]], y, W)[0][0]
        return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)], float(b.std())

    best = int(grid.BA.idxmax())
    final = int(grid.index[(grid.model == config.FINAL_SETTING[0]) & (grid.umap_id == config.FINAL_SETTING[1])
                           & (grid.hdbscan_id == config.FINAL_SETTING[2])][0])

    # 5-fold cross-validation over anchors, overall and within each model
    order = np.random.default_rng(config.SEED).permutation(n_anchor)
    fold_of_anchor = np.empty(n_anchor, int)
    fold_of_anchor[order] = np.arange(n_anchor) % FOLDS
    fold = fold_of_anchor[anchor]
    held = {scope: np.zeros(len(y)) for scope in ["all", *grid.model.unique()]}
    cv_rows = []
    for f in range(FOLDS):
        train, test = fold != f, fold == f
        ba_train = ba(co[:, train], y[train])[0]
        for scope in held:
            idx = np.arange(len(grid)) if scope == "all" else np.flatnonzero(grid.model.eq(scope))
            k = int(idx[np.argmax(ba_train[idx])])
            held[scope][test] = co[k, test]
            cv_rows.append({"fold": f, "scope": scope, **grid.loc[k, KEEP].to_dict(), "BA_train": float(ba_train[k]),
                            "BA_heldout": float(ba(co[[k]][:, test], y[test])[0][0])})
    pd.DataFrame(cv_rows).to_csv(V / "cv_selection.csv", index=False)

    def heldout(v):
        b = ba(v[None], y, W)[0][0]
        return {"BA": round(float(ba(v[None], y)[0][0]), 3),
                "CI95": [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]}

    per_model = []
    own_lists = pairs.retrieved_by.fillna("").str.split(";")[main_m]
    for m, g in grid.groupby("model"):
        k = int(g.BA.idxmax())
        own = own_lists.apply(lambda l: m in l).to_numpy()
        other = (pairs.hits_top10.to_numpy()[main_m] >= 1) & ~own
        per_model.append({"model": m, **grid.loc[k, ["umap_id", "hdbscan_id", "n_topics", "BA", "TPR", "TNR",
                                                     "BA_1_as_related", "BA_1_as_unrelated", "co_rate_random"]].to_dict(),
                          "BA_CI95": ci(k)[0], "BA_heldout_cv": heldout(held[m])["BA"],
                          "BA_own_retrieved": float(ba(co[[k]][:, own], y[own])[0][0]),
                          "BA_other_retrieved": float(ba(co[[k]][:, other], y[other])[0][0])})
    pm = pd.DataFrame(per_model).sort_values("BA", ascending=False)

    best_ci, best_se = ci(best)
    summary = {
        "ratings": a.ratings,
        "score_by_stratum": pairs.groupby("stratum").score.value_counts().unstack(fill_value=0).to_dict("index"),
        "best_setting": {**grid.loc[best, KEEP + ["BA", "TPR", "TNR"]].to_dict(), "CI95": best_ci},
        "final_setting": {**grid.loc[final, KEEP + ["BA", "TPR", "TNR"]].to_dict(), "CI95": ci(final)[0]},
        "heldout_best_of_all_settings": heldout(held["all"]),
        "settings_within_one_SE_of_best": int((grid.BA >= grid.BA[best] - best_se).sum()),
        "per_model": pm.to_dict("records"),
        "kendall_tau_model_ranking": {f"{p} vs {q}": round(float(kendalltau(pm[p], pm[q]).statistic), 3)
                                      for p, q in (("BA", "BA_1_as_related"), ("BA", "BA_1_as_unrelated"))},
    }
    (V / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    # stability shortlist: MPNet's three best, the final setting, and every other model's best
    mp = grid[grid.model == "mpnet"].sort_values("BA", ascending=False).head(3)
    top2 = mp.head(2).sort_values("n_topics")
    picks = [("mpnet_coarse", top2.iloc[0]), ("mpnet_fine", top2.iloc[1]), ("mpnet_3rd", mp.iloc[2]),
             ("mpnet_final", grid.loc[final])]
    picks += [(m, g.loc[g.BA.idxmax()]) for m, g in grid[grid.model != "mpnet"].groupby("model")]
    shortlist = [{"id": cid, "model": r.model, "umap_id": r.umap_id, "hdbscan_id": r.hdbscan_id} for cid, r in picks]
    (V / "shortlist.json").write_text(json.dumps(shortlist, indent=1))

    print(json.dumps({k: summary[k] for k in ("best_setting", "final_setting", "heldout_best_of_all_settings")},
                     indent=1, default=float))
    print(pm[["model", "n_topics", "BA", "BA_CI95", "BA_heldout_cv"]].to_string(index=False))


if __name__ == "__main__":
    main()
