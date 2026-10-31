"""Checks for c-TF-IDF. Run: python -m pytest tests"""
import numpy as np
import scipy.sparse as sp

from hrs_topics.topics import ctfidf


def test_ctfidf_matches_bertopic_formula():
    # terms: apple, banana, cherry, durian; classes -1 / 0 / 1
    D = sp.csr_matrix([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0]])
    labels = np.array([0, 0, 1, -1])
    classes, w, _ = ctfidf(D, labels)
    assert classes == [-1, 0, 1]
    # classes hold 1 / 4 / 2 terms -> A = int(7 / 3) = 2; apple 3, banana 1, cherry 2, durian 1 overall
    idf = np.log(1 + 2 / np.array([3.0, 1.0, 2.0, 1.0]))
    expect = np.array([[0, 0, 1, 0], [.5, .25, .25, 0], [.5, 0, 0, .5]]) * idf
    assert np.allclose(w, expect)


def test_term_in_every_topic_is_down_weighted():
    # apple is the most frequent term of topic 0 but occurs in every class, so banana ranks first
    D = sp.csr_matrix([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0]])
    classes, w, _ = ctfidf(D, np.array([0, 0, 1, -1]))
    assert np.argmax(w[classes.index(0)]) == 1
