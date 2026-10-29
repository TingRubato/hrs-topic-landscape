"""Agreement with rated pairs, and stability between two clusterings.

A pair of abstracts counts as "put together" when both are in the same topic and neither is an
outlier. Balanced accuracy compares that with the ratings: pairs rated 2 (same research question)
should be together, pairs rated 0 (different questions) should not. Pairs rated 1 are left out.
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score


def together(labels, a_pos, b_pos):
    """Boolean per pair: same topic and not an outlier."""
    la, lb = labels[a_pos], labels[b_pos]
    return (la == lb) & (la != -1)


def balanced_accuracy(co, score):
    """Mean of the share of 2-rated pairs put together and the share of 0-rated pairs kept apart."""
    co, score = np.asarray(co, bool), np.asarray(score)
    tpr = co[score == 2].mean()
    tnr = 1 - co[score == 0].mean()
    return float((tpr + tnr) / 2)


def ari_without_outliers(a, b):
    """Adjusted Rand index over documents that are not outliers in either clustering."""
    a, b = np.asarray(a), np.asarray(b)
    m = (a != -1) & (b != -1)
    if len(np.unique(a[m])) < 2 or len(np.unique(b[m])) < 2:
        return float("nan")
    return float(adjusted_rand_score(a[m], b[m]))


def topic_recovery(ref, other):
    """For each topic of `ref`, the best Jaccard overlap with any topic of `other` (same documents)."""
    ref, other = np.asarray(ref), np.asarray(other)
    members = {t: set(np.flatnonzero(other == t)) for t in np.unique(other) if t != -1}
    out = {}
    for t in np.unique(ref):
        if t == -1:
            continue
        r = set(np.flatnonzero(ref == t))
        out[int(t)] = max((len(r & m) / len(r | m) for m in members.values()), default=0.0)
    return out
