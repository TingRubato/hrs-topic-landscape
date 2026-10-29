"""Checks for the evaluation metrics. Run: python -m pytest tests"""
import numpy as np

from hrs_topics.metrics import ari_without_outliers, balanced_accuracy, together, topic_recovery


def test_together_ignores_outliers():
    labels = np.array([0, 0, -1, -1, 1])
    assert together(labels, np.array([0, 2, 0]), np.array([1, 3, 4])).tolist() == [True, False, False]


def test_balanced_accuracy():
    co = np.array([1, 1, 0, 0, 1, 1])
    score = np.array([2, 2, 0, 0, 1, 0])        # 1-rated pair is ignored
    # TPR = 2/2, TNR = 2/3
    assert np.isclose(balanced_accuracy(co, score), (1 + 2 / 3) / 2)


def test_ari_without_outliers():
    a = np.array([0, 0, 1, 1, -1])
    b = np.array([5, 5, 7, 7, 3])               # same partition, outlier dropped
    assert ari_without_outliers(a, b) == 1.0
    assert np.isnan(ari_without_outliers(np.array([0, 0, -1]), np.array([1, 1, 1])))


def test_topic_recovery():
    ref = np.array([0, 0, 1, 1, -1])
    other = np.array([3, 3, 3, 4, 4])
    rec = topic_recovery(ref, other)
    assert rec == {0: 2 / 3, 1: 1 / 3}
